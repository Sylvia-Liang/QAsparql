import requests
import json
import pandas as pd
import numpy as np
import json
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
import itertools
import spotlight
import tagme
import inflect
import re
import sys
import requests
from nltk.stem.porter import *
stemmer = PorterStemmer()
p = inflect.engine()
tagme.GCUBE_TOKEN = ""
from parser.lc_quad_linked import LC_Qaud_Linked


def sort_dict_by_values(dictionary):
    keys = []
    values = []
    for key, value in sorted(dictionary.items(), key=lambda item: (item[1], item[0]), reverse=True):
        keys.append(key)
        values.append(value)
    return keys, values


def preprocess_relations(file, prop=False):
    relations = {}
    with open(file, encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            split_line = line.split()

            key = ' '.join(split_line[2:])[1:-3].lower()
            key = ' '.join([stemmer.stem(word) for word in key.split()])

            if key not in relations:
                relations[key] = []

            uri = split_line[0].replace('<', '').replace('>', '')

            if prop is True:
                uri_property = uri.replace('/ontology/', '/property/')
                relations[key].extend([uri, uri_property])
            else:
                relations[key].append(uri)
    return relations


def get_earl_entities(query):

    result = {}
    result['question'] = query
    result['entities'] = []
    result['relations'] = []

    THRESHOLD = 0.1

    response = requests.post('https://earldemo.sda.tech/earl/api/processQuery',
                             json={"nlquery": query, "pagerankflag": False})

    json_response = json.loads(response.text)
    type_list = []
    chunk = []
    for i in json_response['ertypes']:
        type_list.append(i)
    for i in json_response['chunktext']:
        chunk.append([i['surfacestart'], i['surfacelength']])

    keys = list(json_response['rerankedlists'].keys())
    reranked_lists = json_response['rerankedlists']
    for i in range(len(keys)):
        if type_list[i] == 'entity':
            entity = {}
            entity['uris'] = []
            entity['surface'] = chunk[i]
            for r in reranked_lists[keys[i]]:
                if r[0] > THRESHOLD:
                    uri = {}
                    uri['uri'] = r[1]
                    uri['confidence'] = r[0]
                    entity['uris'].append(uri)
            if entity['uris'] != []:
                result['entities'].append(entity)
        if type_list[i] == 'relation':
            relation = {}
            relation['uris'] = []
            relation['surface'] = chunk[i]
            for r in reranked_lists[keys[i]]:
                if r[0] > THRESHOLD:
                    uri = {}
                    uri['uri'] = r[1]
                    uri['confidence'] = r[0]
                    relation['uris'].append(uri)
            if relation['uris'] != []:
                result['relations'].append(relation)

    return result


def get_tag_me_entities(query):
    threshold = 0.1
    try:
        response = requests.get("https://tagme.d4science.org/tagme/tag?lang=en&gcube-token={}&text={}"
                                .format('1b4eb12e-d434-4b30-8c7f-91b3395b96e8-843339462', query))

        entities = []
        for annotation in json.loads(response.text)['annotations']:
            confidence = float(annotation['link_probability'])
            if confidence > threshold:
                entity = {}
                uris = {}
                uri = 'http://dbpedia.org/resource/' + annotation['title'].replace(' ', '_')
                uris['uri'] = uri
                uris['confidence'] = confidence
                surface = [annotation['start'], annotation['end']-annotation['start']]
                entity['uris'] = [uris]
                entity['surface'] = surface
                entities.append(entity)
    except:
        entities = []
        print('get_tag_me_entities: ', query)
    return entities


def get_nliwod_entities(query, hashmap):
    ignore_list = []
    entities = []
    singular_query = [stemmer.stem(word) if p.singular_noun(word) == False else stemmer.stem(p.singular_noun(word)) for
                      word in query.lower().split(' ')]

    string = ' '.join(singular_query)
    words = query.split(' ')
    indexlist = {}
    surface = []
    current = 0
    locate = 0
    for i in range(len(singular_query)):
        indexlist[current] = {}
        indexlist[current]['len'] = len(words[i])-1
        indexlist[current]['surface'] = [locate, len(words[i])-1]
        current += len(singular_query[i])+1
        locate += len(words[i])+1
    for key in hashmap.keys():
        if key in string and len(key) > 2 and key not in ignore_list:
            e_list = list(set(hashmap[key]))
            k_index = string.index(key)
            if k_index in indexlist.keys():
                surface = indexlist[k_index]['surface']
            else:
                for i in indexlist:
                    if k_index>i and k_index<(i+indexlist[i]['len']):
                        surface = indexlist[i]['surface']
                        break
            for e in e_list:
                r_e = {}
                r_e['surface'] = surface
                r_en = {}
                r_en['uri'] = e
                r_en['confidence'] = 0.3
                r_e['uris'] = [r_en]
                entities.append(r_e)
    return entities


def get_spotlight_entities(query):
    entities = []
    data = {
        'text': query,
        'confidence': '0.4',
        'support': '10'
    }
    headers = {"Accept": "application/json"}
    try:
        response = requests.post('http://model.dbpedia-spotlight.org/en/annotate', data=data, headers=headers)
        response_json = response.text.replace('@', '')
        output = json.loads(response_json)
        if 'Resources' in output.keys():
            resource = output['Resources']
            for item in resource:
                entity = {}
                uri = {}
                uri['uri'] = item['URI']
                uri['confidence'] = float(item['similarityScore'])
                entity['uris'] = [uri]
                entity['surface'] = [int(item['offset']), len(item['surfaceForm'])]
                entities.append(entity)
    except:
        print('Spotlight: ', query)
    return entities


def get_falcon_entities(query):

    entities = []
    relations = []
    headers = {
        'Content-Type': 'application/json',
    }
    params = (
        ('mode', 'long'),
    )
    data = "{\"text\": \"" + query + "\"}"
    response = requests.post('https://labs.tib.eu/falcon/api', headers=headers, params=params, data=data.encode('utf-8'))
    try:
        output = json.loads(response.text)
        for i in output['entities']:
            ent = {}
            ent['surface'] = ""
            ent_uri = {}
            ent_uri['confidence'] = 0.9
            ent_uri['uri'] = i[0]
            ent['uris'] = [ent_uri]
            entities.append(ent)
        for i in output['relations']:
            rel = {}
            rel['surface'] = ""
            rel_uri = {}
            rel_uri['confidence'] = 0.9
            rel_uri['uri'] = i[0]
            rel['uris'] = [rel_uri]
            relations.append(rel)
    except:
            print('get_falcon_entities: ', query)
    return entities, relations


def merge_entity(old_e, new_e):
    for i in new_e:
        exist = False
        for j in old_e:
            for k in j['uris']:
                if i['uris'][0]['uri'] == k['uri']:
                    k['confidence'] = max(k['confidence'], i['uris'][0]['confidence'])
                    exist = True
        if not exist:
            old_e.append(i)
    return old_e


def merge_relation(old_e, new_e):
    for i in range(len(new_e)):
        for j in range(len(old_e)):
            if new_e[i]['surface']==old_e[j]['surface']:
                for i1 in range(len(new_e[i]['uris'])):
                    notexist = True
                    for j1 in range(len(old_e[j]['uris'])):
                        if new_e[i]['uris'][i1]['uri']==old_e[j]['uris'][j1]['uri']:
                            old_e[j]['uris'][j1]['confidence'] = max(old_e[j]['uris'][j1]['confidence'], new_e[i]['uris'][i1]['confidence'])
                            notexist = False
                    if notexist:
                        old_e[j]['uris'].append(new_e[i]['uris'][i1])
    return old_e


if __name__ == "__main__":

    with open('data/LC-QUAD/data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    properties = preprocess_relations('dbpedia_3Eng_property.ttl', True)
    print('properties: ', len(properties))

    ds = LC_Qaud_Linked(path="./data/LC-QUAD/linked_answer.json")
    ds.load()
    ds.parse()

    linked_data = []
    na_entity = []
    for qapair in ds.qapairs:
        idx = qapair.id.__str__()
        query = qapair.question.text
        earl = dict()
        if qapair.answerset is None or len(qapair.answerset) == 0:
            earl['question'] = query
            earl['id'] = idx
            earl['entities'] = []
            earl['relations'] = []
            na_entity.append(idx)
        else:
            earl = get_earl_entities(query)
            tagme_e = get_tag_me_entities(query)
            if len(tagme_e) > 0:
                earl['entities'] = merge_entity(earl['entities'], tagme_e)

            nliwod = get_nliwod_entities(query, properties)
            if len(nliwod) > 0:
                earl['relations'] = merge_entity(earl['relations'], nliwod)

            spot_e = get_spotlight_entities(query)
            if len(spot_e) > 0:
                earl['entities'] = merge_entity(earl['entities'], spot_e)

            e_falcon, r_falcon = get_falcon_entities(query)
            if len(e_falcon) > 0:
                earl['entities'] = merge_entity(earl['entities'], e_falcon)
            if len(r_falcon) > 0:
                earl['relations'] = merge_entity(earl['relations'], r_falcon)

            esim = []
            for i in earl['entities']:
                i['uris'] = sorted(i['uris'], key=lambda k: k['confidence'], reverse=True)
                esim.append(max([j['confidence'] for j in i['uris']]))

            earl['entities'] = np.array(earl['entities'])
            esim = np.array(esim)
            inds = esim.argsort()[::-1]
            earl['entities'] = earl['entities'][inds]

            rsim = []
            for i in earl['relations']:
                i['uris'] = sorted(i['uris'], key=lambda k: k['confidence'], reverse=True)
                rsim.append(max([j['confidence'] for j in i['uris']]))

            earl['relations'] = np.array(earl['relations'])
            rsim = np.array(rsim)
            inds = rsim.argsort()[::-1]
            earl['relations'] = earl['relations'][inds]

            earl['entities'] = list(earl['entities'])
            earl['relations'] = list(earl['relations'])
            earl['id'] = idx

        linked_data.append(earl)

    with open('data/LC-QUAD/entity_lcquad.json', "w") as data_file:
        json.dump(linked_data, data_file, sort_keys=True, indent=4, separators=(',', ': '))

    with open('na_entity.txt', 'w') as f:
        for i in na_entity:
            f.write("{}\n".format(i))

