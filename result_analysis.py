import json
import argparse
import os
import re
import numpy as np


def analysis(fiename):
    with open("output/{}.json".format(fiename), 'r') as data_file:
        data = json.load(data_file)

    na = lf = correct = incorrect = no_path = no_answer =  0
    p = []
    r=[]
    n_list = 0
    n_count = 0
    n_ask = 0
    p_count = []
    p_list = []
    p_ask = []
    r_count = []
    r_list = []
    r_ask = []
    incor = 0
    sp = []
    sr = []
    cp=[]
    cr=[]
    svp=[]
    svr=[]
    mvp=[]
    mvr=[]

    for i in data:
        p.append(i['precision'])
        r.append(i['recall'])
        if i['precision'] == i['recall'] == 0.0:
            incor +=1

        if i['answer'] == "correct":
            correct +=1
        elif i['answer'] == "-Not_Applicable":
            na +=1
        elif i['answer'] == "-Linker_failed":
            lf +=1
        elif i['answer'] == "-incorrect":
            incorrect +=1
        elif i['answer'] == "-without_path":
            no_path +=1
        elif i['answer'] == "-no_answer":
            no_answer +=1

        if 'ASK' in i['query']:
            n_ask +=1
            p_ask.append(i['precision'])
            r_ask.append(i['recall'])
        elif 'COUNT(' in i['query']:
            n_count+=1
            p_count.append(i['precision'])
            r_count.append(i['recall'])
        else:
            n_list +=1
            p_list.append(i['precision'])
            r_list.append(i['recall'])

        if 'single' in i['features']:
            sp.append(i['precision'])
            sr.append(i['recall'])
        elif 'compound' in i['features']:
            cp.append(i['precision'])
            cr.append(i['recall'])

        if 'singlevar' in i['features']:
            svp.append(i['precision'])
            svr.append(i['recall'])
        elif 'multivar' in i['features']:
            mvp.append(i['precision'])
            mvr.append(i['recall'])


    print("-- Basic Stats --")
    print("-  Total Questions: %d" % (correct+incorrect+no_path+no_answer+na+lf))
    print("-  Correct: %d" % correct)
    print("-  Incorrect: %d" % incorrect)
    print("-  No-Path: %d" % no_path)
    print("-  No-Answer: %d" % no_answer)
    print("-  Not_Applicable: %d" % na)
    print("-  Linker_failed: %d" % lf)
    print('-  Wrong Answer: %d' % incor)

    print('None in precision: ',sum(i is None for i in p))
    print('None in recall: ', sum(i is None for i in r))

    p = np.array(p, dtype=np.float64)
    r = np.array(r, dtype=np.float64)
    mp = np.nanmean(p)
    mr = np.nanmean(r)

    print("-  Precision: %.4f" % mp)
    print("-  Recall: %.4f" % mr)
    print("-  F1: %.4f" % ((2*mp*mr)/(mp+mr)))

    p_count = np.array(p_count, dtype=np.float64)
    p_list = np.array(p_list, dtype=np.float64)
    p_ask = np.array(p_ask, dtype=np.float64)
    r_count = np.array(r_count, dtype=np.float64)
    r_list = np.array(r_list, dtype=np.float64)
    r_ask = np.array(r_ask, dtype=np.float64)
    print('List: ', n_list)
    a = np.nanmean(p_list)
    b = np.nanmean(r_list)
    print('precision: %.4f' % a)
    print('reacall: %.4f' % b)
    print('f1-score: %.4f'% ((2*a*b)/(a+b)))

    print('Count: ', n_count)
    a = np.nanmean(p_count)
    b = np.nanmean(r_count)
    print('precision: %.4f' % a)
    print('reacall: %.4f' % b)
    print('f1-score: %.4f'% ((2*a*b)/(a+b)))

    print('Ask: ', n_list)
    a = np.nanmean(p_ask)
    b = np.nanmean(r_ask)
    print('precision: %.4f' % a)
    print('reacall: %.4f' % b)
    print('f1-score: %.4f'% ((2*a*b)/(a+b)))

    sp = np.array(sp, dtype=np.float64)
    sr = np.array(sr, dtype=np.float64)
    cp=np.array(cp, dtype=np.float64)
    cr=np.array(cr, dtype=np.float64)
    print('Single: ', len(sp), len(sr))
    a = np.nanmean(sp)
    b = np.nanmean(sr)
    print('precision: %.4f' % a)
    print('reacall: %.4f' % b)
    print('f1-score: %.4f'% ((2*a*b)/(a+b)))

    print('Compound: ', len(cp), len(cr))
    a = np.nanmean(cp)
    b = np.nanmean(cr)
    print('precision: %.4f' % a)
    print('reacall: %.4f' % b)
    print('f1-score: %.4f'% ((2*a*b)/(a+b)))


    svp=np.array(svp, dtype=np.float64)
    svr=np.array(svr, dtype=np.float64)
    mvp=np.array(mvp, dtype=np.float64)
    mvr=np.array(mvr, dtype=np.float64)
    print('Single Var: ', len(svp), len(svr))
    a = np.nanmean(svp)
    b = np.nanmean(svr)
    print('precision: %.4f' % a)
    print('reacall: %.4f' % b)
    print('f1-score: %.4f' % ((2 * a * b) / (a + b)))

    print('Multiple Var: ', len(mvp), len(mvr))
    a = np.nanmean(mvp)
    b = np.nanmean(mvr)
    print('precision: %.4f' % a)
    print('reacall: %.4f' % b)
    print('f1-score: %.4f' % ((2 * a * b) / (a + b)))


if __name__ == "__main__":
    file = "lcquadtestanswer_output"
    print('LC-QUAD test: ')
    analysis(file)
    print('\n'*2)

    file = "qaldanswer_output"
    print('QALD-7: ')
    analysis(file)
    print('\n'*2)

    file = "lcquadanswer_output"
    print('LC-QUAD all: ')
    analysis(file)
    print('\n'*2)


