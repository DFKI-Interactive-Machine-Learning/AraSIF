import sys
import numpy
from scipy.spatial.distance import cdist
sys.path.append('../src')
import data_io
import params
import SIF_embedding
import read_NMT_data


# input arabic file
sample_ara = '../NMT_data/sample.ara'  # to compute sif embeddings for all sentences in this file

# Arabic GloVe embedding pre-trained model
wordfile = '../models/glove_full_grams_sg_300_wiki.txt'
weightfile = '../AraSIF_word_counts/arwiki_vocab_min200.txt'  # each line is a word and its frequency

weightpara = 1e-3  # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1  # number of principal components to remove in SIF weighting scheme

# load word vectors
print("Reading embedding matrix. Hang on! this will take a while ...")
(glove_words, We) = data_io.getWordmap(wordfile)
print("shape of Word embedding is: " + str(We.shape))

# load word weights
word2weight = data_io.getWordWeight(weightfile, weightpara)  # word2weight['str'] is the weight for the word 'str'
weight4ind = data_io.getWeight(glove_words, word2weight)  # weight4ind[i] is the weight for the i-th word

# set parameters
params = params.params()
params.rmpc = rmpc

# load sentences
print("reading the input sentences now & converting to indices .. \n")
sample_sents = read_NMT_data.read_data(sample_ara)

# AraSIF embedding for sample sentences
print("computing AraSIF embedding now ...\n")

# x is the array of word indices, m is the binary mask indicating whether there is a word in that location
x, m = data_io.sentences2idx(sample_sents, glove_words)
w = data_io.seq2weight(x, m, weight4ind)  # get word weights
sample_embedding = SIF_embedding.SIF_embedding(We, x, w, params)  # embedding[i,:] is the embedding for sentence i
print("shape of sample sentence embedding is: " + str(sample_embedding.shape))

# serialize for future use
numpy.save('sample_sentence_embedding.npy', sample_embedding)
