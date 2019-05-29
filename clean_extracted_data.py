# -*- coding: utf8 -*-
import os
import re
import sys
import argparse
import numpy as np


# Clean/Normalize Arabic Text
def remove_english_content(args, input_file, en_str):
    clean_lines = []
    with open(os.path.join(args.extracted_data_path, input_file), 'r') as f:
        for line in f:
            line = line.strip()
            existence = 0
            for token in en_str:
                if token in line:
                    existence = 1
                    break
            if existence == 1:
                continue
            else:
                clean_lines.append(line)
    return clean_lines


def main(args):
    files = os.listdir(args.extracted_data_path)
    en_str = ("<doc", "</doc>", "https://", "http://")
    for in_file in files:
        clean_lines = remove_english_content(args, in_file, en_str=en_str)
        with open(os.path.join(args.target_data_path, in_file + "_cleaned"), "w") as fout:
            fout.write("\n".join(clean_lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='clean extracted wiki text')
    parser.add_argument('-src', "--extracted_data_path", type=str, default="./extracted_data",
                        help="'path to the folder: `extracted data`; \
                        this the result of running `./wikiextractor/WikiExtractor.py` on `arwiki-latest-pages-articles.xml.bz2`")
    parser.add_argument('-dest', "--target_data_path", type=str, default="./cleaned_wikidata",
                        help="'path to the folder: `cleaned wikidata`; this is where the cleaned data will be stored")
    args = parser.parse_args()
    main(args)
