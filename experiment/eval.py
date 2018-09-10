#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import pickle
__author__ = 'Go Inoue'


tag_list = ["POS", "Abbr", "AdpType", "Aspect", "Case", "Definite",
            "Foreign", "Gender", "Mood", "Negative", "NumForm",
            "NumValue", "Number", "Person", "PronType",
            "VerbForm", "Voice"]

tex_list = ["POS", "Gender", "Number", "Case", "Mood", "Aspect", "Person",
            "Voice", "Definite", "Abbr", "AdpType", "Foreign", "Negative",
            "NumForm", "NumValue", "PronType", "VerbForm"]

def eval(pred_path, gold_path):
    '''
    :param pred_path: str of gold annotated conll format file
    :param gold_path: str of predicted conll format file
    '''

    with open(pred_path) as f1, open(gold_path) as f2:
        pos_pred, pos_gold = [], []
        for l1, l2 in zip(f1.readlines(), f2.readlines()):
            if len(l1) > 1 and len(l2) > 1:
                line1 = l1.strip().split('\t')
                line2 = l2.strip().split('\t')
                word1, pos1 = line1[1], line1[2:] # word, pos in gold
                word2, pos2 = line2[1], line2[2:] # word, pos in pred
                pos_pred.append(pos1)
                pos_gold.append(pos2)
        assert len(pos_pred) == len(pos_gold)
        n_samples = len(pos_pred)

        # each morphological faeture is correct of not
        acc_dict = {}
        cors = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for i in range(len(pos1)):
            for pred, gold in zip(pos_pred, pos_gold):
                if pred[i] == gold[i]:
                    cors[i] += 1
        cors = [100*i/len(pos_pred) for i in cors]
        for k,v in zip(tag_list, cors):
            acc_dict[k] = v
            print("{}:\t{:.2f}".format(k,v))

        # all features are correct or not
        cor = 0
        cor_all = 90.36
        for pred, gold in zip(pos_pred, pos_gold):
            if pred == gold:
                cor += 1
        print('All: \t{:.2f} ({}/{})'.format(100*cor/n_samples, cor, n_samples))
        acc_list = [acc_dict[tex_list[i]] for i in range(len(tex_list))]
        baseline_list = [95.92,97.96,96.69,94.60,99.67,99.50,99.45,99.21,96.67,99.99,99.85,99.47,99.99,99.90,99.98,99.81,99.78]
        acc = []
        dif = []
        a = []
        for feat, baseline in zip(acc_list, baseline_list):
            acc.append("{:.2f}".format(feat))
            if feat-baseline>0:
                dif.append("+{:.2f}".format(feat-baseline))
            else:
                dif.append("{:.2f}".format(feat-baseline))
        acc.append("{:.2f}".format(100*cor/n_samples))
        if 100*cor/n_samples-cor_all>0:
            dif.append("+{:.2f}".format(100*cor/n_samples-cor_all))
        else:
            dif.append("{:.2f}".format(100*cor/n_samples-cor_all))
        print('')

        print("For tex:\t{}\\\\".format("&".join([feat for feat in tex_list[:9]])))
        print("Tex Acc:\t{}\\\\".format("&".join(acc[:9])))
        print("Tex Diff:\t{}\\\\".format("&".join(dif[:9])))

        print('')

        print("For tex:\t{}&All\\\\".format("&".join([feat for feat in tex_list[9:]])))
        print("Tex Acc:\t{}\\\\".format("&".join(acc[9:])))
        print("Tex Diff:\t{}\\\\".format("&".join(dif[9:])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str)
    parser.add_argument("--gold", type=str)
    args = parser.parse_args()

    eval(args.pred, args.gold)
