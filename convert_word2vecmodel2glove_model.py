# -*- coding: utf8 -*-
import os
import sys
import gensim
import argparse
import numpy as np

"""
sr: https://radimrehurek.com/gensim/scripts/glove2word2vec.html
-----------------
Word2Vec format:
-----------------
9 4
word1 0.123 0.134 0.532 0.152
word2 0.934 0.412 0.532 0.159
word3 0.334 0.241 0.324 0.188
...
wordn 0.334 0.241 0.324 0.188

-----------------
GloVe format:
-----------------
word1 0.123 0.134 0.532 0.152
word2 0.934 0.412 0.532 0.159
word3 0.334 0.241 0.324 0.188
...
wordn 0.334 0.241 0.324 0.188
"""


def load_and_save_model(args):
    # load model
    word2vec_model = 'full_grams_sg_300_wiki.mdl'
    wiki_ngram_model = gensim.models.Word2Vec.load(os.path.join(args.word2vec_model_path, word2vec_model))

    # get the words and its 300D vector representation and write it to file
    glove_model = "glove_full_grams_sg_300_wiki.txt"
    with open(os.path.join(args.glove_model_path, glove_model), 'w') as gf:
        for token in wiki_ngram_model.wv.vocab.keys():
            vector = wiki_ngram_model.wv[token]
            vector = vector.tolist()
            gf.write(token + " " + " ".join(map(str, vector)) + "\n")
    print("Saved word2vec model to GloVe model in: {}".format(os.path.join(args.glove_model_path, glove_model)))


def main(args):
    load_and_save_model(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert Word2Vec model to GloVe model')
    parser.add_argument('-w2v', "--word2vec_model_path", type=str, default="./models",
                        help="path to pre-trained word2vec model")
    parser.add_argument('-glove', "--glove_model_path", type=str, default="./models",
                        help="path to save GloVe model")
    args = parser.parse_args()
    main(args)
