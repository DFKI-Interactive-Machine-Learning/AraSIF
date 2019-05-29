# -*- coding: utf8 -*-
import os
import re
import sys
from collections import Counter
import argparse
import numpy as np


def compute_wordcount(arguments, input_file, word_freq_counter):
    check_re = re.compile('[a-zA-Z0-9_-]+')
    en_alphanum = '[a-zA-Z0-9]'
    with open(os.path.join(arguments.clean_wikidata_path, input_file), 'r') as f:
        for line in f:
            line = line.strip()
            tokens = line.split()
            for token in tokens:
                # we're only interested in pure Arabic tokens
                re1_res = check_re.match(token)  # a pure Arabic token would return `None`
                re2_res = re.search(en_alphanum, token)  # a pure Arabic token would return `None`
                if re1_res is None and re2_res is None:
                    word_freq_counter[token] += 1
                else:    # it means it contains some English alphabets or Numbers
                    continue
    return word_freq_counter


def main(arguments):
    files = os.listdir(arguments.clean_wikidata_path)
    word_freq_counter = Counter()
    for in_file in files:
        word_freq = compute_wordcount(args, in_file, word_freq_counter)
        word_freq_counter = word_freq + word_freq_counter
    print("total words: {}".format(len(word_freq_counter)))

    with open(os.path.join(arguments.output_dir, "arwiki_vocab_min200.txt"), "w") as af:
        for token, count in word_freq_counter.items():
            if count >= 200:
                af.write(token + " " + str(count) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='clean extracted wiki text')
    parser.add_argument('-input', "--clean_wikidata_path", type=str, default="./cleaned_wikidata", help="'path to the folder: `cleaned wiki data`; this the result of running `./clean_extracted_data.py` on `./extracted_data`")
    parser.add_argument('-output', "--output_dir", type=str, default="./AraSIF_word_counts", help="'path to the word frequency file: this is needed for SIF https://github.com/PrincetonML/SIF_mini_demo/blob/master/examples/sif_embedding.py#L7")
    args = parser.parse_args()

    main(args)
