#!/usr/bin/env python
# -*- coding: utf_8 -*-

"""
Indenpendent Prediction of Morphological Features for Fine-graind Arabic
Part-of-Speech Tagging
"""

__author__ = "Go Inoue"

import time
import random
import sys
import argparse
import pickle
import re
import os
import logging as L
from collections import Counter, defaultdict

import dynet as dy
import numpy as np

NUM = re.compile("[0-9]")
feat_list = ["POS", "PRC3", "PRC2", "PRC1", "PRC0", "PER", "ASP", "VOX", "MOD",
"GEN", "NUM", "STT", "CAS", "ENC0"]

class Vocab:
    def __init__(self, w2i=None):
        if w2i is None:
            w2i = defaultdict(lambda: len(w2i))
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.items()}
    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(lambda: len(w2i))
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)
    @classmethod
    def morphs(self):
        w2i = defaultdict(lambda: len(w2i))
        vts = dict()
        with open("./data/features.txt") as f:
            features_dict = defaultdict(set)
            lines = [line.strip().split("\t") for line in f.readlines()]
            for line in lines:
                features_dict[line[0].upper()] = set(line[1].split())
        for feat in feat_list:
            vts[feat] = self.from_corpus([list(features_dict[feat])])
        return vts
    def size(self):
        return len(self.w2i.keys())
    def vocabs(self):
        return self.i2w.values()

def read(fname):
    corpus = []
    with open(fname) as f:
        wt_list = []
        for line in f.readlines():
            cols = line.rstrip().split('\t')
            if len(cols) > 1:
                # Extract word and tag from conll format
                word = cols[1]
                word = NUM.sub('0', word)
                tags = dict()
                for name, tag in zip(feat_list, cols[2:-1]):
                    tags[name] = tag
                # Make tuple of word and tags, then append to the list "wt_list"
                wt_list.append((word, tags))
            else:
                corpus.append(wt_list)
                wt_list = []
        if wt_list:
            corpus.append(wt_list)
    return corpus

class Tagger(dy.Saveable):
    def __init__(self, n_words, n_chars, n_w_emb, n_c_emb, n_d_emb, n_hidden,
                 n_layer, n_mlp, vw, vc, vts, feat_dict, optimizer, feature):
        self.model = dy.Model()
        self.optimizer = optimizer
        self.WORDS_LOOKUP = self.model.add_lookup_parameters((n_words, n_w_emb))
        self.CHARS_LOOKUP = self.model.add_lookup_parameters((n_chars, n_c_emb))
        if args.use_dict:
            self.FEAT_LOOKUPs = {k: self.model.add_lookup_parameters((vts[k].size(),\
                                 n_d_emb)) for k,v in vts.items()}
            if args.dict_all:
                self.fwdRNN = dy.VanillaLSTMBuilder(n_layer, n_w_emb+n_w_emb/2+\
                                                    n_d_emb*14,n_hidden, self.model)
                self.bwdRNN = dy.VanillaLSTMBuilder(n_layer, n_w_emb+n_w_emb/2+\
                                                    n_d_emb*14,n_hidden, self.model)
            else:
                self.fwdRNN = dy.VanillaLSTMBuilder(n_layer, n_w_emb+n_w_emb/2+\
                                                    n_d_emb,n_hidden, self.model)
                self.bwdRNN = dy.VanillaLSTMBuilder(n_layer, n_w_emb+n_w_emb/2+\
                                                    n_d_emb,n_hidden, self.model)
        else:
            self.fwdRNN = dy.VanillaLSTMBuilder(n_layer, n_w_emb+n_w_emb/2,
                                                n_hidden, self.model)
            self.bwdRNN = dy.VanillaLSTMBuilder(n_layer, n_w_emb+n_w_emb/2,
                                                n_hidden, self.model)
        self.pH = self.model.add_parameters((n_mlp, n_hidden*2))
        self.pO = self.model.add_parameters((vts[feature].size(), n_mlp))
        self.cFwdRNN = dy.VanillaLSTMBuilder(1, n_c_emb, n_w_emb/4, self.model)
        self.cBwdRNN = dy.VanillaLSTMBuilder(1, n_c_emb, n_w_emb/4, self.model)
        self.vw = vw
        self.vc = vc
        self.vts = vts
        self.feat_dict = feat_dict
        self.feature = feature
        self.best_parameters = None
        self.top_cor = 0

    def graph(self, words):
        dy.renew_cg()

        H = dy.parameter(self.pH)
        O = dy.parameter(self.pO)

        # initialize the RNNs
        f_init = self.fwdRNN.initial_state()
        b_init = self.bwdRNN.initial_state()

        cf_init = self.cFwdRNN.initial_state()
        cb_init = self.cBwdRNN.initial_state()

        wembs = [self.word_rep(w, cf_init, cb_init) for w in words]

        # feed word vectors into biLSTM
        fw_exps = f_init.transduce(wembs)
        bw_exps = b_init.transduce(reversed(wembs))

        # biLSTM states
        bi_exps = [dy.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))]

        # feed each biLSTM state to an MLP
        exps = []
        for x in bi_exps:
            r_t = O*(dy.tanh(H * x))
            exps.append(r_t)
        return exps

    def word_rep(self, w, cf_init, cb_init):
            # word embeddings
            if w in self.vw.w2i:
                w_index = self.vw.w2i[w]
            else:
                w_index = self.vw.w2i["_UNK_"]
            w_emb =  self.WORDS_LOOKUP[w_index]

            # char embeddings
            pad_char = self.vc.w2i["<*>"]
            char_ids = [pad_char] + [self.vc.w2i.get(c,self.vc.w2i["_UNK_"])\
                                     for c in w] + [pad_char]
            char_embs = [self.CHARS_LOOKUP[cid] for cid in char_ids]
            fw_exps = cf_init.transduce(char_embs)
            bw_exps = cb_init.transduce(reversed(char_embs))
            char_emb = dy.concatenate([fw_exps[-1], bw_exps[-1]])
            if args.use_dict:
                # dictionary embeddings
                if w in self.feat_dict:
                    cand_emb_list = []#
                    if args.dict_all:
                        for k,v in sorted(self.vts.items()):
                            candidates = [self.vts[k].w2i[c] for c in self.feat_dict[w][k]]
                            cand_embs = [self.FEAT_LOOKUPs[k][cid] for cid in candidates]
                            cand_emb_list.append(dy.esum(cand_embs))
                    else:
                        candidates = [self.vts[self.feature].w2i[c] for c in self.feat_dict[w][self.feature]]
                        cand_embs = [self.FEAT_LOOKUPs[self.feature][cid] for cid in candidates]
                        cand_emb_list.append(dy.esum(cand_embs))
                    cand_emb = dy.concatenate(cand_emb_list)
                else:
                    cand_emb_list = []
                    if args.dict_all:
                        for k,v in sorted(self.vts.items()):
                            candidates = [self.vts[k].w2i[c] for c in self.vts[k].vocabs()]
                            cand_embs = [self.FEAT_LOOKUPs[k][cid] for cid in candidates]
                            cand_emb_list.append(dy.esum(cand_embs))
                    else:
                        candidates = [self.vts[self.feature].w2i[c] for c in self.vts[self.feature].vocabs()]
                        cand_embs = [self.FEAT_LOOKUPs[self.feature][cid] for cid in candidates]
                        cand_emb_list.append(dy.esum(cand_embs))
                    cand_emb = dy.concatenate(cand_emb_list)

                return dy.concatenate([w_emb, char_emb, cand_emb])
            else:
                return dy.concatenate([w_emb, char_emb])

    def sent_loss_precalc(self, words, tags, vecs):
        errs = []
        for v,t in zip(vecs,tags):
            tid = self.vts[self.feature].w2i[t]
            err = dy.pickneglogsoftmax(v, tid)
            errs.append(err)
        return dy.esum(errs)

    def sent_loss(self, words, tags):
        return self.sent_loss_precalc(words, tags, self.graph(words))

    def tag_sent_precalc(self, words, vecs):
        log_probs = [v.npvalue() for v in vecs]
        tags = []
        for prb in log_probs:
            tag = np.argmax(prb)
            tags.append(self.vts[self.feature].i2w[tag])
        return zip(words, tags)

    def tag_sent(self, words):
        return self.tag_sent_precalc(words, self.graph(words))

    def train(self, train, dev, n_epoch, path):
        if self.optimizer == "Adam":
            trainer = dy.AdamTrainer(self.model)
            L.info(" Optimizer:\tAdam")
        else:
            lr = 0.01
            trainer = dy.SimpleSGDTrainer(self.model, e0=lr)
            L.info("Optimizer:\tSGD with learing rate of {}".format(lr))
        L.info(" =========== Training Logs ===========\n")
        start = time.time()
        i = all_time = dev_time = all_tagged = this_tagged = this_loss = 0
        n_samples_train = len([[w for w,t in s] for s in train])

        for ITER in range(n_epoch):
            random.shuffle(train)
            train_loss = i = 0
            print ("\nEpoch: {}".format(ITER+1))
            for s in train:
                words = [w for w,t in s]
                golds = [t[self.feature] for w,t in s]
                loss_exp = self.sent_loss(words, golds)
                train_loss += loss_exp.scalar_value()
                loss_exp.backward()
                trainer.update()

            sys.stderr.write("\n")

            L.info(" Number of epoch: {}\n".format(ITER+1))
            L.info(" Train Loss: {}".format(train_loss/n_samples_train))

            # dev set accuracy evaluation
            dev_start = time.time()
            this_cor = self.accuracy(dev)
            print("Train Loss: {}".format(train_loss/n_samples_train))
            dev_time += time.time() - dev_start
            train_time = time.time() - start - dev_time
            L.info(" Train Time: {}, Word/Sec: {}\n".format(train_time,\
                   n_samples_train/train_time))
            if this_cor > self.top_cor:
                self.top_cor = this_cor
                self.best_parameters = (ITER, self.get_components())
            trainer.update_epoch(1.0)
        self.save(path)
        L.info(" Training Done.")

    def accuracy(self, corpus):
        correct = n_samples = i = 0
        n_samples_corpus = len([[w for w,t in s] for s in corpus])
        for sent in corpus:
            words = [w for w,t in sent]
            golds = [t[self.feature] for w,t in sent] # list of dict
            tags = [t for w,t in self.tag_sent(words)] # list of dict
            for go, gu in zip(golds, tags):
                if go == gu:
                    correct += 1
            n_samples += len(golds)

        sys.stderr.write("\n")
        print("Accuracy: {:.2f}% ({}/{})".format(\
              100*correct/n_samples, correct, n_samples))

        L.info(" Accuracy of {}:\t{:.2f}\t({}/{})\n".format(self.feature,
               100*correct/n_samples, correct, n_samples))
        return correct

    def save(self, path):
        best_epoch = self.best_parameters[0]
        best_parameters = self.best_parameters[1]
        L.info(" Best Model at: {} epochs".format(best_epoch+1))
        file_name = os.path.join(path, "dev_best.model")
        self.model.save(file_name, best_parameters)

    def load_model(self, model_file):
        self.restore_components(self.model.load(model_file))
        print("Model loaded!")

    def get_components(self):
        if args.use_dict:
            components = (self.WORDS_LOOKUP, self.CHARS_LOOKUP,
                          self.FEAT_LOOKUPs["POS"], self.FEAT_LOOKUPs["PRC3"],
                          self.FEAT_LOOKUPs["PRC2"], self.FEAT_LOOKUPs["PRC1"],
                          self.FEAT_LOOKUPs["PRC0"], self.FEAT_LOOKUPs["PER"],
                          self.FEAT_LOOKUPs["ASP"], self.FEAT_LOOKUPs["VOX"],
                          self.FEAT_LOOKUPs["MOD"], self.FEAT_LOOKUPs["GEN"],
                          self.FEAT_LOOKUPs["NUM"], self.FEAT_LOOKUPs["STT"],
                          self.FEAT_LOOKUPs["CAS"], self.FEAT_LOOKUPs["ENC0"],
                          self.pH, self.pO, self.fwdRNN, self.bwdRNN,\
                          self.cFwdRNN, self.cBwdRNN)
        else:
            components = (self.WORDS_LOOKUP, self.CHARS_LOOKUP, self.pH,\
                          self.pO, self.fwdRNN, self.bwdRNN, self.cFwdRNN,\
                          self.cBwdRNN)
        return components

    def restore_components(self, components):
        if args.use_dict:
            self.WORDS_LOOKUP, self.CHARS_LOOKUP,\
            self.FEAT_LOOKUPs["POS"], self.FEAT_LOOKUPs["PRC3"],\
            self.FEAT_LOOKUPs["PRC2"], self.FEAT_LOOKUPs["PRC1"],\
            self.FEAT_LOOKUPs["PRC0"], self.FEAT_LOOKUPs["PER"],\
            self.FEAT_LOOKUPs["ASP"], self.FEAT_LOOKUPs["VOX"],\
            self.FEAT_LOOKUPs["MOD"], self.FEAT_LOOKUPs["GEN"],\
            self.FEAT_LOOKUPs["NUM"], self.FEAT_LOOKUPs["STT"],\
            self.FEAT_LOOKUPs["CAS"], self.FEAT_LOOKUPs["ENC0"],\
            self.pH, self.pO, self.fwdRNN, self.bwdRNN, self.cFwdRNN,\
            self.cBwdRNN = components
        else:
            self.WORDS_LOOKUP, self.CHARS_LOOKUP, self.pH, self.pO,\
            self.fwdRNN, self.bwdRNN, self.cFwdRNN, self.cBwdRNN = components

    def save_conll(self, corpus, path):
        sents = []
        for sent in corpus:
            words = [w for w,t in sent]
            sent_tags = []
            for w,t in self.tag_sent(words):
                tags = []
                tags.append(w)
                tags.append(t)
                sent_tags.append(tags)
            sents.append(sent_tags)
        output = ""
        for sent in sents:
            n = 1
            for word in sent:
                output += str(n) + "\t" + "\t".join(word) + "\n"
                n += 1
            output += "\n"
        with open(path, "w") as f:
            f.write(output[:-1])
        print("Saved as CoNLL format!\n")

def train_mode():
    feature = args.feature
    now = time.strftime("%Y_%m%d_%H%M%S" ,time.strptime(time.ctime()))
    if args.use_dict:
        if not os.path.exists("log/separate_dict"):
            os.mkdir("log/separate_dict")
        log_dir = os.path.join("log/separate_dict", feature+ "_" +\
                               args.optimizer+now)
    else:
        if not os.path.exists("log/separate"):
            os.mkdir("log/separate")
        log_dir = os.path.join("log/separate", feature+ "_" +\
                               args.optimizer+now)
    os.mkdir(log_dir)

    log_file = os.path.join(log_dir, "experimental_settings.log")
    L.basicConfig(filename=log_file, level=L.INFO)
    L.info(" =========== Training Settings ===========\n")
    if args.use_dict:
        L.info(" Use of Dictionary Embeddings:\tTrue")
    else:
        L.info(" Use of Dictionary Embeddings:\tFalse")
    L.info(" Training Data:\t{}".format(args.train))
    L.info(" Validation Data:\t{}".format(args.dev))
    L.info(" Dimension of Character Embeddings:\t{}".format(args.CEMBED_SIZE))
    L.info(" Dimension of Word Embeddings:\t{}".format(args.WEMBED_SIZE))
    L.info(" Dimension of Dictionary Embeddings:\t{}".format(args.DEMBED_SIZE))
    L.info(" Dimension of LSTM Layer:\t{}".format(args.HIDDEN_SIZE))
    L.info(" Number of LSTM Layers:\t{}".format(args.N_LAYER))
    L.info(" Dimension of MLP Layer:\t{}".format(args.MLP_SIZE))

    if args.dynet_seed:
        L.info(" Random Seed:\t{}".format(args.dynet_seed))
        np.random.seed(args.dynet_seed)
    random.seed(args.dynet_seed)

    train = read(args.train)[:args.limit]
    dev = read(args.dev)
    L.info(" Train Sentences:\t{}".format(len(train)))
    L.info(" Validation Sentences:\t{}".format(len(dev)))

    words, tags = [], []
    chars = set()
    wc = Counter()
    for sent in train:
        for w,p in sent:
            words.append(w)
            tags.append(p)
            chars.update(w)
            wc[w]+=1

    words.append("_UNK_")
    chars.add("<*>")
    chars.add("_UNK_")

    vts = Vocab.morphs()
    vw = Vocab.from_corpus([words])
    vc = Vocab.from_corpus([chars])

    # save voabs
    vocabs = {}
    vocabs["word"], vocabs["tags"], vocabs["chars"] = vw, vts, vc

    with open(os.path.join(log_dir,"vocabs.pickle"), "wb") as f:
        pickle.dump(vocabs, f)

    with open(args.tag_dict, "rb") as f:
        feat_dict = pickle.load(f)

    n_words = vw.size()
    n_chars = vc.size()
    n_tags = vts[feature].size()

    L.info(" Vocabulary Size of Word:\t{}".format(n_words-1))
    L.info(" Vocabulary Size of Character:\t{}".format(n_chars-2))
    L.info(" Vocabulary Size of {}:\t{}".format(feature, n_tags))

    tagger = Tagger(n_words=n_words, n_chars=n_chars,
                    n_w_emb=args.WEMBED_SIZE, n_c_emb=args.CEMBED_SIZE,
                    n_d_emb=args.DEMBED_SIZE, n_hidden=args.HIDDEN_SIZE,
                    n_mlp=args.MLP_SIZE, n_layer=args.N_LAYER, vw=vw, vc=vc,
                    vts=vts, feat_dict=feat_dict, optimizer=args.optimizer,
                    feature=feature)
    tagger.train(train=train, dev=dev, n_epoch=args.epoch, path=log_dir)

def test_mode():
    feature = args.feature
    print("Feature: {}".format(feature))
    model_path = os.path.join(args.model_dir, "dev_best.model")
    if args.use_dict:
        log_file = "./experiment/results_separate_dict.txt"
    else:
        log_file = "./experiment/results_separate.txt"
    L.basicConfig(filename=log_file, level=L.INFO)
    test=read(args.test)
    now = time.strftime(" Date:\t%Y %m%d %H:%M:%S", time.strptime(time.ctime()))
    L.info(now)
    L.info(" Model:\t{}".format(model_path))
    L.info(" Test Sentences:\t{}".format(len(test)))

    with open(os.path.join(args.model_dir, "vocabs.pickle"), "rb") as f:
        vocabs = pickle.load(f)
    with open(args.tag_dict, "rb") as f:
        feat_dict = pickle.load(f)
    vw, vc, vts = vocabs["word"], vocabs["chars"], vocabs["tags"]
    n_words, n_chars, n_tags = vw.size(), vc.size(), vts[feature].size()
    L.info(" Vocabulary Size of {}:\t{}".format(feature, n_tags))

    tagger = Tagger(n_words=n_words, n_chars=n_chars,
                    n_w_emb=args.WEMBED_SIZE, n_c_emb=args.CEMBED_SIZE,
                    n_d_emb=args.DEMBED_SIZE, n_hidden=args.HIDDEN_SIZE,
                    n_mlp=args.MLP_SIZE, n_layer=args.N_LAYER, vw=vw, vc=vc,
                    vts=vts, feat_dict=feat_dict, optimizer=args.optimizer,
                    feature=feature)
    tagger.restore_components(tagger.model.load(model_path))
    tagger.accuracy(test)
    if args.save_conll:
        if args.use_dict:
            tagger.save_conll(test, "./experiment/separate_dict/"+feature+"_dict_pred.conll")
        else:
            tagger.save_conll(test, "./experiment/separate/"+feature+"_pred.conll")

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--train", type=str)
    parser.add_argument("--dev", type=str)
    parser.add_argument("--test", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--feature", default="POS", type=str)
    parser.add_argument('--use_dict', action='store_true')
    parser.add_argument('--dict_all', action='store_true')
    parser.add_argument('--save_conll', action='store_true')
    parser.add_argument("--optimizer", default="SGD", type=str)
    parser.add_argument("--dynet-seed", default=0, type=int)
    parser.add_argument("--dynet-gpus", default=0, type=int)
    parser.add_argument("--dynet-mem", default=512, type=int)
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--tag_dict", default="./data/feat_dict.pickle", type=str)
    parser.add_argument("CEMBED_SIZE", type=int, help="char embedding size")
    parser.add_argument("WEMBED_SIZE", type=int, help="embedding size")
    parser.add_argument("DEMBED_SIZE", type=int, help="dict embedding size")
    parser.add_argument("HIDDEN_SIZE", type=int, help="hidden size")
    parser.add_argument("N_LAYER", type=int, help="number of LSTM layers")
    parser.add_argument("MLP_SIZE", type=int, help="embedding size")
    args = parser.parse_args()

    if args.mode == "train":
        train_mode()
    else:
        test_mode()
