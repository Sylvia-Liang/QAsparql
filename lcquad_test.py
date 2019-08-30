from common.graph.graph import Graph
from common.query.querybuilder import QueryBuilder
from parser.lc_quad import LC_Qaud
from sklearn.model_selection import train_test_split
import os
import torch.optim as optim
from learning.treelstm.model import *
from learning.treelstm.vocab import Vocab
from learning.treelstm.trainer import Trainer
from learning.treelstm.dataset import QGDataset
import learning.treelstm.preprocess_lcquad as preprocess_lcquad
from common.container.uri import Uri
from common.container.linkeditem import LinkedItem
from parser.lc_quad import LC_QaudParser
import common.utility.utility as utility
from learning.classifier.svmclassifier import SVMClassifier
import ujson
import learning.treelstm.Constants as Constants
import numpy as np
import itertools
from parser.lc_quad_linked import LC_Qaud_Linked
from parser.qald import Qald
from common.container.sparql import SPARQL
from common.container.answerset import AnswerSet
from common.graph.graph import Graph
from common.utility.stats import Stats
from common.query.querybuilder import QueryBuilder
from linker.goldLinker import GoldLinker
from linker.earl import Earl
import json
import argparse
import logging
import sys
import os
import itertools
from collections import Counter


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_uris(answers):
    uris = set()
    if type(answers) is list and answers != []:
        for answer in answers:
            for answer_value in answer['results']['bindings']:
                uris.add(answer_value[answers['head']['vars'][0]]['value'])
    elif type(answers) is dict and 'results' in answers.keys():
        if 'bindings' in answers['results'].keys():
            if answers['results']['bindings'] != []:
                for answer_value in answers['results']['bindings']:
                    uris.add(answer_value[answers['head']['vars'][0]]['value'])
    elif type(answers) is dict and 'boolean' in answers.keys():
        uris.add(answers['boolean'])

    return uris


class Struct(object): pass


class Orchestrator:
    def __init__(self, logger, question_classifier, double_relation_classifer, parser, filepath, auto_train=True):
        self.logger = logger
        self.question_classifier = question_classifier
        self.double_relation_classifer = double_relation_classifer
        self.parser = parser
        self.kb = parser.kb
        self.filepath = filepath
        self.X_train, self.X_test, self.y_train, self.y_test = [], [], [], []

        # if auto_train and not question_classifier.is_trained:
        #     self.train_question_classifier()

        if auto_train and question_classifier is not None and not question_classifier.is_trained:
            self.train_question_classifier()

        if auto_train and double_relation_classifer is not None and not double_relation_classifer.is_trained:
            self.train_double_relation_classifier()

        self.dep_tree_cache_file_path = './caches/dep_tree_cache_lcquadtest.json'
        if os.path.exists(self.dep_tree_cache_file_path):
            with open(self.dep_tree_cache_file_path) as f:
                self.dep_tree_cache = ujson.load(f)
        else:
            self.dep_tree_cache = dict()

    def prepare_question_classifier_dataset(self, file_path=None):
        if file_path is None:
            ds = LC_Qaud()
        else:
            ds = LC_Qaud(self.filepath)
        ds.load()
        ds.parse()

        X = []
        y = []
        for qapair in ds.qapairs:
            X.append(qapair.question.text)
            if "COUNT(" in qapair.sparql.query:
                y.append(2)
            elif "ASK " in qapair.sparql.query:
                y.append(1)
            else:
                y.append(0)

        return X, y

    def prepare_double_relation_classifier_dataset(self, file_path=None):
        if file_path is None:
            ds = LC_Qaud()
        else:
            ds = LC_Qaud(self.filepath)
        ds.load()
        ds.parse()

        X = []
        y = []
        for qapair in ds.qapairs:
            X.append(qapair.question.text)
            relation_uris = [u for u in qapair.sparql.uris if u.is_ontology() or u.is_type()]
            if len(relation_uris) != len(set(relation_uris)):
                y.append(1)
            else:
                y.append(0)

        return X, y

    def train_question_classifier(self, file_path=None, test_size=0.2):
        X, y = self.prepare_question_classifier_dataset(self.filepath)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size,
                                                                                random_state=42)
        return self.question_classifier.train(self.X_train, self.y_train)

    def train_double_relation_classifier(self, file_path=None, test_size=0.2):
        X, y = self.prepare_double_relation_classifier_dataset(self.filepath)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size,
                                                                                random_state=42)
        return self.double_relation_classifer.train(self.X_train, self.y_train)

    def rank(self, args, question, generated_queries):
        if len(generated_queries) == 0:
            return []
        if 2 > 1:
            # try:
            # Load the model
            checkpoint_filename = '%s.pt' % os.path.join(args.save, args.expname)
            dataset_vocab_file = os.path.join(args.data, 'dataset.vocab')
            # metrics = Metrics(args.num_classes)
            vocab = Vocab(filename=dataset_vocab_file,
                          data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
            similarity = DASimilarity(args.mem_dim, args.hidden_dim, args.num_classes)
            model = SimilarityTreeLSTM(
                vocab.size(),
                args.input_dim,
                args.mem_dim,
                similarity,
                args.sparse)
            criterion = nn.KLDivLoss()
            optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wd)
            emb_file = os.path.join(args.data, 'dataset_embed.pth')
            if os.path.isfile(emb_file):
                emb = torch.load(emb_file)
            model.emb.weight.data.copy_(emb)
            checkpoint = torch.load(checkpoint_filename, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'])
            trainer = Trainer(args, model, criterion, optimizer)

            # Prepare the dataset
            json_data = [{"id": "test", "question": question,
                          "generated_queries": [{"query": " .".join(query["where"]), "correct": False} for query in
                                                generated_queries]}]
            output_dir = "./output/tmp"
            preprocess_lcquad.save_split(output_dir, *preprocess_lcquad.split(json_data, self.parser))

            if question in self.dep_tree_cache:
                preprocess_lcquad.parse(output_dir, dep_parse=False)

                cache_item = self.dep_tree_cache[question]
                with open(os.path.join(output_dir, 'a.parents'), 'w') as f_parent, open(
                        os.path.join(output_dir, 'a.toks'), 'w') as f_token:
                    for i in range(len(generated_queries)):
                        f_token.write(cache_item[0])
                        f_parent.write(cache_item[1])
            else:
                preprocess_lcquad.parse(output_dir)
                with open(os.path.join(output_dir, 'a.parents')) as f:
                    parents = f.readline()
                with open(os.path.join(output_dir, 'a.toks')) as f:
                    tokens = f.readline()
                self.dep_tree_cache[question] = [tokens, parents]

                with open(self.dep_tree_cache_file_path, 'w') as f:
                    ujson.dump(self.dep_tree_cache, f)
            test_dataset = QGDataset(output_dir, vocab, args.num_classes)

            test_loss, test_pred = trainer.test(test_dataset)
            return test_pred
        # except Exception as expt:
        #     self.logger.error(expt)
        #     return []

    def generate_query(self, question, entities, relations, h1_threshold=9999999, question_type=None):
        ask_query = False
        sort_query = False
        count_query = False

        if question_type is None:
            question_type = 0
            if self.question_classifier is not None:
                question_type = self.question_classifier.predict([question])
        if question_type == 2:
            count_query = True
        elif question_type == 1:
            ask_query = True

        type_confidence = self.question_classifier.predict_proba([question])[0][question_type]
        if isinstance(self.question_classifier.predict_proba([question])[0][question_type], (np.ndarray, list)):
            type_confidence = type_confidence[0]

        double_relation = False
        if self.double_relation_classifer is not None:
            double_relation = self.double_relation_classifer.predict([question])
            if double_relation == 1:
                double_relation = True

        graph = Graph(self.kb)
        query_builder = QueryBuilder()
        graph.find_minimal_subgraph(entities, relations, double_relation=double_relation, ask_query=ask_query,
                                    sort_query=sort_query, h1_threshold=h1_threshold)

        valid_walks = query_builder.to_where_statement(graph, self.parser.parse_queryresult, ask_query=ask_query,
                                                       count_query=count_query, sort_query=sort_query)

        if question_type == 0 and len(relations) == 1:
            double_relation = True
            graph = Graph(self.kb)
            query_builder = QueryBuilder()
            graph.find_minimal_subgraph(entities, relations, double_relation=double_relation, ask_query=ask_query,
                                        sort_query=sort_query, h1_threshold=h1_threshold)
            valid_walks_new = query_builder.to_where_statement(graph, self.parser.parse_queryresult,
                                                               ask_query=ask_query,
                                                               count_query=count_query, sort_query=sort_query)
            valid_walks.extend(valid_walks_new)

        args = Struct()
        base_path = "./learning/treelstm/"
        args.expname = "lc_quad,epoch=5,train_loss=0.08340245485305786"
        args.mem_dim = 150
        args.hidden_dim = 50
        args.num_classes = 2
        args.input_dim = 300
        args.sparse = False
        args.lr = 0.01
        args.wd = 1e-4
        args.data = os.path.join(base_path, "data/lc_quad/")
        args.cuda = False
        # args.cuda = True
        try:
            scores = self.rank(args, question, valid_walks)
        except:
            scores = [1 for _ in valid_walks]
        for idx, item in enumerate(valid_walks):
            if idx >= len(scores):
                item["confidence"] = 0.3
            else:
                item["confidence"] = float(scores[idx] - 1)

        return valid_walks, question_type, type_confidence

    def sort_query(self, linker, kb, parser, qapair, question_type_classifier, force_gold=True):
        logger.info(qapair.sparql)
        logger.info(qapair.question.text)

        # Get Answer from KB online
        status, raw_answer_true = kb.query(qapair.sparql.query.replace("https", "http"))
        answerset_true = AnswerSet(raw_answer_true, parser.parse_queryresult)
        qapair.answerset = answerset_true

        ask_query = False
        count_query = False

        question = qapair.question.text

        question_type = question_type_classifier.predict([question])

        if question_type == 2:
            count_query = True
        elif question_type == 1:
            ask_query = True

        type_confidence = question_type_classifier.predict_proba([question])[0][question_type]
        if isinstance(question_type_classifier.predict_proba([question])[0][question_type], (np.ndarray, list)):
            type_confidence = type_confidence[0]
            type_confidence = float(type_confidence)

        question_type = int(question_type)

        entities, ontologies = linker.do(qapair, force_gold=force_gold)
        precision = None
        recall = None

        if qapair.answerset is None or len(qapair.answerset) == 0:
            return "-Not_Applicable", [], question_type, type_confidence, precision, recall
        else:
            if entities is None or ontologies is None:
                recall = 0.0
                return "-Linker_failed", [], question_type, type_confidence, precision, recall

            logger.info("start finding the minimal subgraph")

            entity_list = []
            for L in range(1, len(entities) + 1):
                for subset in itertools.combinations(entities, L):
                    entity_list.append(subset)
            entity_list = entity_list[::-1]

            relation_list = []
            for L in range(1, len(ontologies) + 1):
                for subset in itertools.combinations(ontologies, L):
                    relation_list.append(subset)
            relation_list = relation_list[::-1]

            combination_list = [(x, y) for x in entity_list for y in relation_list]

            args = Struct()
            base_path = "./learning/treelstm/"
            args.save = os.path.join(base_path, "checkpoints/")
            args.expname = "lc_quad,epoch=5,train_loss=0.08340245485305786"
            args.mem_dim = 150
            args.hidden_dim = 50
            args.num_classes = 2
            args.input_dim = 300
            args.sparse = False
            args.lr = 0.01
            args.wd = 1e-4
            args.data = os.path.join(base_path, "data/lc_quad/")
            args.cuda = False

            generated_queries = []

            for comb in combination_list:
                if len(generated_queries) == 0:
                    generated_queries, question_type, type_confidence = self.generate_query(question, comb[0], comb[1])
                    if len(generated_queries) > 0:
                        ask_query = False
                        count_query = False

                        if int(question_type) == 2:
                            count_query = True
                        elif int(question_type) == 1:
                            ask_query = True
                else:
                    break

            generated_queries.extend(generated_queries)
            if len(generated_queries) == 0:
                recall = 0.0
                return "-without_path", [], question_type, type_confidence, precision, recall

            scores = []
            for s in generated_queries:
                scores.append(s['confidence'])

            scores = np.array(scores)
            inds = scores.argsort()[::-1]
            sorted_queries = [generated_queries[s] for s in inds]
            scores = [scores[s] for s in inds]

            used_answer = []
            uniqueid = []
            for i in range(len(sorted_queries)):
                if sorted_queries[i]['where'] not in used_answer:
                    used_answer.append(sorted_queries[i]['where'])
                    uniqueid.append(i)

            sorted_queries = [sorted_queries[i] for i in uniqueid ]
            scores = [scores[i] for i in uniqueid]

            s_counter = Counter(sorted(scores, reverse=True))
            s_ind = []
            s_i = 0
            for k, v in s_counter.items():
                s_ind.append(range(s_i, s_i + v))
                s_i += v

            output_where = [{"query": " .".join(item["where"]), "correct": False, "target_var": "?u_0"} for item in sorted_queries]
            for item in list(output_where):
                logger.info(item["query"])
            correct = False

            wrongd = {}

            for idx in range(len(sorted_queries)):
                where = sorted_queries[idx]

                if "answer" in where:
                    answerset = where["answer"]
                    target_var = where["target_var"]
                else:
                    target_var = "?u_" + str(where["suggested_id"])
                    raw_answer = kb.query_where(where["where"], target_var, count_query, ask_query)
                    answerset = AnswerSet(raw_answer, parser.parse_queryresult)

                output_where[idx]["target_var"] = target_var
                sparql = SPARQL(kb.sparql_query(where["where"], target_var, count_query, ask_query), ds.parser.parse_sparql)

                answereq = (answerset == qapair.answerset)
                try:
                    sparqleq = (sparql == qapair.sparql)
                except:
                    sparqleq = False

                if answereq != sparqleq:
                    print("error")

                if answerset == qapair.answerset:
                    correct = True
                    output_where[idx]["correct"] = True
                    output_where[idx]["target_var"] = target_var
                    recall = 1.0
                    precision = 1.0
                    correct_index = idx
                    break
                else:
                    if target_var == "?u_0":
                        target_var = "?u_1"
                    else:
                        target_var = "?u_0"
                    raw_answer = kb.query_where(where["where"], target_var, count_query, ask_query)
                    answerset = AnswerSet(raw_answer, parser.parse_queryresult)

                    sparql = SPARQL(kb.sparql_query(where["where"], target_var, count_query, ask_query), ds.parser.parse_sparql)

                    answereq = (answerset == qapair.answerset)
                    try:
                        sparqleq = (sparql == qapair.sparql)
                    except:
                        sparqleq = False

                    if answereq != sparqleq:
                        print("error")

                    if answerset == qapair.answerset:
                        correct = True
                        output_where[idx]["correct"] = True
                        output_where[idx]["target_var"] = target_var
                        recall=1.0
                        precision=1.0
                        correct_index = idx
                        break
                    else:
                        correct = False
                        output_where[idx]["correct"] = False
                        output_where[idx]["target_var"] = target_var
                        intersect = answerset.intersect(qapair.answerset)
                        recall= intersect/len(qapair.answerset)
                        precision= intersect/len(answerset)
                        wrongd[idx] = intersect

            if correct:
                # here the precision and recall is calculated based on the number of correct generated queries
                for si in s_ind:
                    if correct_index in si:
                        if len(si)>1:
                            c_answer = []
                            t_answer = []
                            for j in si:
                                where = sorted_queries[j]

                                if "answer" in where:
                                    answerset = where["answer"]
                                    target_var = where["target_var"]
                                else:
                                    target_var = "?u_" + str(where["suggested_id"])
                                    raw_answer = kb.query_where(where["where"], target_var, count_query, ask_query)
                                    answerset = AnswerSet(raw_answer, parser.parse_queryresult)

                                output_where[j]["target_var"] = target_var
                                sparql = SPARQL(kb.sparql_query(where["where"], target_var, count_query, ask_query),
                                                ds.parser.parse_sparql)

                                answereq = (answerset == qapair.answerset)
                                try:
                                    sparqleq = (sparql == qapair.sparql)
                                except:
                                    sparqleq = False

                                if answereq != sparqleq:
                                    print("error")

                                if len(answerset)>0:
                                    if answerset == qapair.answerset:
                                        c_answer.append(len(answerset))
                                        t_answer.append(len(answerset))
                                    else:
                                        if target_var == "?u_0":
                                            target_var = "?u_1"
                                        else:
                                            target_var = "?u_0"
                                        raw_answer = kb.query_where(where["where"], target_var, count_query, ask_query)
                                        answerset = AnswerSet(raw_answer, parser.parse_queryresult)

                                        sparql = SPARQL(kb.sparql_query(where["where"], target_var, count_query, ask_query),
                                                        ds.parser.parse_sparql)

                                        answereq = (answerset == qapair.answerset)
                                        try:
                                            sparqleq = (sparql == qapair.sparql)
                                        except:
                                            sparqleq = False

                                        if answereq != sparqleq:
                                            print("error")

                                        if answerset == qapair.answerset:
                                            c_answer.append(len(answerset))
                                            t_answer.append(len(answerset))
                                        else:
                                            intersect = answerset.intersect(qapair.answerset)
                                            c_answer.append(intersect)
                                            t_answer.append(len(answerset))
                            precision = sum(c_answer)/sum(t_answer)
                            recall = min(sum(c_answer)/len(qapair.answerset),1.0)
                            break
            else:
                mkey, mvalue = max(wrongd.items(), key=lambda x: x[1])
                for si in s_ind:
                    if mkey in si:
                        if len(si)>1:
                            c_answer = []
                            t_answer = []
                            for j in si:
                                where = sorted_queries[j]

                                if "answer" in where:
                                    answerset = where["answer"]
                                    target_var = where["target_var"]
                                else:
                                    target_var = "?u_" + str(where["suggested_id"])
                                    raw_answer = kb.query_where(where["where"], target_var, count_query, ask_query)
                                    answerset = AnswerSet(raw_answer, parser.parse_queryresult)

                                output_where[j]["target_var"] = target_var
                                sparql = SPARQL(kb.sparql_query(where["where"], target_var, count_query, ask_query),
                                                ds.parser.parse_sparql)

                                answereq = (answerset == qapair.answerset)
                                try:
                                    sparqleq = (sparql == qapair.sparql)
                                except:
                                    sparqleq = False

                                if answereq != sparqleq:
                                    print("error")

                                if len(answerset)>0:
                                    if answerset == qapair.answerset:
                                        c_answer.append(len(answerset))
                                        t_answer.append(len(answerset))
                                    else:
                                        if target_var == "?u_0":
                                            target_var = "?u_1"
                                        else:
                                            target_var = "?u_0"
                                        raw_answer = kb.query_where(where["where"], target_var, count_query, ask_query)
                                        answerset = AnswerSet(raw_answer, parser.parse_queryresult)

                                        sparql = SPARQL(kb.sparql_query(where["where"], target_var, count_query, ask_query),
                                                        ds.parser.parse_sparql)

                                        answereq = (answerset == qapair.answerset)
                                        try:
                                            sparqleq = (sparql == qapair.sparql)
                                        except:
                                            sparqleq = False

                                        if answereq != sparqleq:
                                            print("error")

                                        if answerset == qapair.answerset:
                                            c_answer.append(len(answerset))
                                            t_answer.append(len(answerset))
                                        else:
                                            intersect = answerset.intersect(qapair.answerset)
                                            c_answer.append(intersect)
                                            t_answer.append(len(answerset))
                            precision = sum(c_answer)/sum(t_answer)
                            recall = min(sum(c_answer)/len(qapair.answerset),1.0)
                            break

            return "correct" if correct else "-incorrect", output_where, question_type, type_confidence, precision, recall


def safe_div(x, y):
    if y == 0:
        return None
    return x / y


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    utility.setup_logging()

    if not os.path.isfile("./data/LC-QUAD/linked_test.json"):
        with open("./learning/treelstm/data/lc_quad/LCQuad_test.json", 'r') as f:
            test = json.load(f)

        with open('data/LC-QUAD/linked_answer.json', 'r') as f:
            data = json.load(f)

        test_id=[]
        for i in test:
            test_id.append(i['id'])
        test_data = []
        for i in data:
            if i['id'] in test_id:
                test_data.append(i)
        with open('data/LC-QUAD/linked_test.json', 'w') as f:
            json.dump(test_data, f)

    ds = LC_Qaud_Linked(path="./data/LC-QUAD/linked_test.json")
    ds.load()
    ds.parse()

    if not ds.parser.kb.server_available:
        logger.error("Server is not available. Please check the endpoint at: {}".format(ds.parser.kb.endpoint))
        sys.exit(0)

    output_file = 'lcquadtestanswer_output'
    linker = Earl(path="data/LC-QUAD/entity_lcquad_test.json")

    base_dir = "./output"
    question_type_classifier_path = os.path.join(base_dir, "question_type_classifier")
    double_relation_classifier_path = os.path.join(base_dir, "double_relation_classifier")
    utility.makedirs(question_type_classifier_path)
    utility.makedirs(double_relation_classifier_path)
    question_type_classifier = SVMClassifier(os.path.join(question_type_classifier_path, "svm.model"))
    double_relation_classifier = SVMClassifier(os.path.join(double_relation_classifier_path, "svm.model"))

    stats = Stats()

    parser = LC_QaudParser()
    kb = parser.kb

    o = Orchestrator(logger, question_type_classifier, double_relation_classifier, parser, question_type_classifier_path, True)

    tmp = []
    output = []
    na_list = []

    for qapair in ds.qapairs:
        stats.inc("total")
        output_row = {"question": qapair.question.text,
                      "id": qapair.id,
                      "query": qapair.sparql.query,
                      "answer": "",
                      "question_type": None,
                      "type_confidence": None,
                      "features": list(qapair.sparql.query_features()),
                      "generated_queries": [],
                      "precision": None,
                      "recall": None}

        if qapair.answerset is None or len(qapair.answerset) == 0:
            stats.inc("query_no_answer")
            output_row["answer"] = "-no_answer"
            na_list.append(output_row['id'])
        else:
            result, where, question_type, type_confidence, precision, recall = o.sort_query(linker, ds.parser.kb, ds.parser, qapair, question_type_classifier, True)
            stats.inc(result)
            output_row["answer"] = result
            output_row["generated_queries"] = where
            output_row["question_type"] = question_type
            output_row["type_confidence"] = type_confidence
            output_row["precision"] = precision
            output_row["recall"] = recall
            logger.info(result)

        logger.info(stats)
        output.append(output_row)

        if stats["total"] % 100 == 0:
            with open("output/{}.json".format(output_file), "w") as data_file:
                json.dump(output, data_file, sort_keys=True, indent=4, separators=(',', ': '), cls=NumpyEncoder)

    with open("output/{}.json".format(output_file), "w") as data_file:
        json.dump(output, data_file, sort_keys=True, indent=4, separators=(',', ': '), cls=NumpyEncoder)
    print('stats: ', stats)

    with open('na_list_lcquadtest.txt', 'w') as f:
        for i in na_list:
            f.write("{}\n".format(i))

