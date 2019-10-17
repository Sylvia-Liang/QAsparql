# QAsparql
Question-Answering system for Knowledge Graph (DBpedia)

Abstract:

The emergence of Linked Data in the form of knowledge graphs in Resource Description Framework (RDF) data models has been among the first developments since Semantic Web appeared in 2001. These knowledge graphs are typically enormous and are not easily accessible to end users because they require specialized knowledge in query languages (such as SPARQL) as well as deep understanding of content structure in the underlying knowledge graph. This led to the development of Question-Answering (QA) systems based on RDF data to allow end users to access the knowledge graphs and benefit from the information stored in them. While QA systems have progressed rapidly in recent years, there is still room for improvement. 
To make the knowledge graphs more accessible to end users, we propose a new QA system for translating natural language questions into SPARQL queries. The key idea is to use neural network models to automatically learn and translate a natural language question into a SPARQL query. Our QA system first predicts the types of questions and then constructs the SPARQL query by extracting, ranking and selecting triple patterns from the original question. The final SPARQL query is constructed by combining the selected triple pattern with the predicted question type. The performance of our proposed QA system is empirically evaluated using the two renowned benchmarks - the 7th Question Answering over Linked Data Challenge (QALD-7) and the Large-Scale Complex Question Answering Dataset (LC- QuAD). Experimental results show that our QA system out- performs the state-of-art systems by 15% on the QALD-7 dataset and by 48% on the LC-QuAD dataset, respectively. The advantage of our approach is that it is generically applicable since it does not require any domain-specific knowledge. 


Preprocess:
1. bash earning/treelstm/download.sh -- download the pre-trained word embedding models FastText and Glove


Whole Process:
1. python lcquad_dataset.py -- preprocess the LC-QuAD dataset, generate 'linked_answer.json' file as the LC-QuAD dataset with golden standard answers
2. python lcquad_answer.py -- generate the golden answers for LC-QuAD dataset, generate 'lcquad_gold.json' file as LC-QuAD dataset with generated SPARQL queries based on the entities and properties extracted from the correct standard SPARQL query
3. python learning/treelstm/preprocess_lcquad.py -- preprocess the LC-QuAD dataset for Tree-LSTM training, split the original Lc-QuAD dataset into 'LCQuad_train.json', 'LCQuad_trial.json', 'LCQuad_test.json' each with 70%\20%\10% of the original dataset. Generate the dependency parsing tree and the corresponding input and output required to train the Tree-LSTM model.
4. python learning/treelstm/main.py -- train Tree-LSTM. The generated checkpoints files are stored in \checkpoints folder and used in lcquad_test.py and qald_test.py
5. python entity_lcquad_test.py -- generate phrase mapping for LC-QuAD test dataset
6. python entity_qald.py -- generate phrase mapping for QALD-7 test dataset
7. python lcquad_test.py -- test the QA system on LC-QuAD test dataset
8. python lcquadall_test.py -- test the QA system on LC-QuAD whole dataset
8. python qald_test.py  -- test the QA system on QALD-7 dataset
9. python question_type_anlaysis.py  -- analyze the question type classification accuracy on LC-QuAD and QALD-7 dataset
10. python result_analysis.py  -- analyze the final result for LC-QuAD and QALD-7 dataset


