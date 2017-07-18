#!/usr/bin/env python
# -*- coding: utf_8 -*-

from collections import defaultdict
import pickle
import re

tag_list = ['POS', 'PRC3', 'PRC2', 'PRC1', 'PRC0',\
            'PER', 'ASP', 'VOX', 'MOD', 'GEN',\
            'NUM', 'STT', 'CAS', 'ENC0']

SENT = re.compile("SENTENCE BREAK")
WORD = re.compile("(;;WORD )(.*)")
FEAT = re.compile("(diac:.+)")
FEAT_CAMEL = re.compile("\*[0-9].+(pos:.+)")
BW = re.compile("(bw:)([^ ]*)")
NA = re.compile("(NO-ANALYSIS \[)(.*)\]")

def load_all(in_file):
    d = defaultdict(list)
    with open(in_file) as f:
        word = ''
        tags = []
        for line in f.readlines():
            if WORD.search(line):
                word = WORD.search(line).group(2)
            elif FEAT.search(line):
                feat = FEAT.search(line).group(1)
                feats = []
                for f in feat.split():
                    feats.append(f)
                d[word].append(feats[:-5][-13:])
    return d

def load_camel(in_file):
    corpus = []
    with open(in_file) as f:
        word = ''
        tags = []
        sents, l = [], []
        for line in f.readlines():
            if line[:12] == ';;; SENTENCE':
                sents.append(l)
                l = []
            if WORD.search(line):
                word = WORD.search(line).group(2)
            elif FEAT_CAMEL.search(line):
                feat = FEAT_CAMEL.search(line).group(1)
                feats = []
                for f in feat.split():
                    feats.append(f)
                l.append([word]+feats[:14])
        sents.append(l)
    return sents[1:]

def save_conll(corpus, path):
    output = ""
    for sent in corpus:
        n = 1
        for word in sent:
            output += str(n) + "\t" + "\t".join(word) + "\n"
            n += 1
        output += "\n"
    with open(path, "w") as f:
        f.write(output[:-1])
    print("Saved as CoNLL format!")

def read_conll(fname):
    vocab = set()
    with open(fname) as f:
        wt_list = []
        for line in f.readlines():
            cols = line.rstrip().split("\t")
            if len(cols) > 1:
                # Extract word and tag from conll format
                word = cols[1]
                vocab.add(word)
    return vocab

def feature_dict(in_file):
    d = load_all(in_file)
    f_dict = defaultdict(dict)
    for word, feature_list in d.items():
        tags =  defaultdict(set)
        for v in feature_list:
            for name, tag in zip(tag_list, v):
                tags[name].add(tag)
        f_dict[word] = tags
    return f_dict

def load_conll_train(fname):
    d = defaultdict(list)
    with open(fname) as f:
        word = ''
        tags = []
        for line in f.readlines():
            cols = line.rstrip().split("\t")
            if len(cols) > 1:
                # Extract word and tag from conll format
                word = cols[1]
                word = NUM.sub("0", word)
                tag = []
                for name, feat in zip(tag_list, cols[2:-1]):
                    tag.append(feat)
                d[word].append(tag)
    return d

def feature_dict_conll(in_file):
    d = load_conll_train(in_file)
    f_dict = defaultdict(dict)
    for word, feature_list in d.items():
        tags =  defaultdict(set)
        for v in feature_list:
            for name, tag in zip(tag_list, v):
                tags[name].add(tag)
        f_dict[word] = tags
    return f_dict

def save(path, var):
    with open(path, "wb") as f:
        pickle.dump(var, f)
    return print('Saved!')
