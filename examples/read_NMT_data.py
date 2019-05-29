import sys
import numpy


def read_data(input_file):
    """
    we will just call this function in `compute_arasif_embedding.py` in '../examples'
    """
    print "reading sentences from input Arabic file ..."
    lines = []
    with open(input_file, 'r') as f:
        for line in f:
            lines.append(line.strip())
        print "number of sentences in file '" + input_file + "' are: " + str(len(lines))
        return lines
