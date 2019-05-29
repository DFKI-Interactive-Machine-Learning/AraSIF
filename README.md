# AraSIF

This is a minimal example for computing sentence embedding for Arabic as described [Kalimuthu et. al (2019)](https://sites.google.com/view/wanlp-2019/home#h.p_Bd2Gh1L0z6cE): **"Incremental Domain Adaptation for Neural Machine Translation in Low-Resource Settings"**.

This code adapted from the [original SIF implementation](https://github.com/PrincetonML/SIF_mini_demo) and is written in python. In addition, it requires gensim, numpy, scipy, pickle, and sklearn packages.

#### Install
To install all the dependencies `conda` or `virtualenv` environment is recommended. After activating an isolated environment, install the requirements using:

```
$ pip install -r requirements.txt 
```

---------------------------------------


#### Preparing data and pretrained model for AraSIF
At a high level, there are two main steps:

**1)** First, we need a GloVe pre-trained word embedding model
Since it would be a time-consuming exercise to train a word embedding model from scratch on huge amount of data, we will use existing pretrained models. One such option for Arabic is [AraVec](https://github.com/bakrianoo/aravec) which is a pre-trained Word2Vec model that has been trained on Arabic Wikipedia data. So, we will download that first:

 i)  Go to [aravec#n-grams-models-1](https://github.com/bakrianoo/aravec#n-grams-models-1)
 ii) Download the model with the following characteristics:

   > Model : Wikipedia-SkipGram
   > Docs No.: 1,800,000
   > Vocabularies No.:  662,109
   > Vec-Size: 300

 Or simply click the URL: [aravec_3_wiki/full_grams_sg_300_wiki.zip](https://archive.org/download/aravec_3_wiki/full_grams_sg_300_wiki.zip)

But, this downloaded model is in Word2Vec format. So, we will have to convert this Word2Vec model to a GloVe model since that's what SIF uses to compute sentence embeddings.
To do so, one can use the following utility script: `./convert_word2vecmodel2glove_model.py`

Once executed, this will generate a ~4.1GB file which is in GloVe format and write it to `./models/glove_full_grams_sg_300_wiki.txt`. (***)


**2)** Second, we need a word weight file which is simply a frquency count of tokens in our data

For this, first we have to download & preprocess Arabic Wikipedia dump on which the Word2Vec model was trained on. To do so, one could simply follow the steps below:

 (i)- Download the latest dump of Arabic Wikipedia from the URL: [dumps.wikimedia.org/arwiki/latest/arwiki-latest-pages-articles.xml.bz2](https://dumps.wikimedia.org/arwiki/latest/arwiki-latest-pages-articles.xml.bz2)

 (ii) `cd` into `wikiextractor`

 (iii) Run the following command
  `python WikiExtractor.py -o ../extracted_data -b 50M ../arwiki-latest-pages-articles.xml.bz2`

Now, we need little bit cleaning of the extracted Wikipedia dump. For this, please use the utility script: `clean_extracted_data.py`

Phew! finally, we can extract the much desired word frequencies. For this, we can use the utility script `get_word_counts.py`. This will compute the counts and write to a file in the location `./AraSIF_word_counts/arwiki_vocab_min200.txt` in a desired format.  (***)


#### Compute Sentence Embeddings
To get started, `cd` into the directory `examples/` and run `demo.sh` 

**NOTE**: It is assumed that the Arabic Wikipedia dump is downloaded and preprocessed, pretrained Word2Vec model is downloaded and converted to GloVe format as described above, prior to executing `demo.sh`. We cannot simply distribute these files due to [memory restrictions of GitHub repositories](https://help.github.com/en/articles/what-is-my-disk-quota#file-and-repository-size-limitations).


#### Source code
The original source code for SIF consists of:
* `SIF_embedding.py`: implements the SIF embedding. The SIF weighting scheme is very simple and is implmented in a few lines.
* `data_io.py`: provides the function for loading data.
* utilities: includes `params.py`, and `tree.py`. These provides utility data structure for the above.

Those were written by [@YingyuLiang](https://github.com/YingyuLiang).

The following scripts were written by [@kmario23](https://github.com/kmario23)
`clean_extracted_data.py`, `convert_word2vecmodel2glove_model.py`, `get_word_counts.py`

The `wikiextractor` project is maintained by [@attardi](https://github.com/attardi)


#### References
For more details about how we leverage these sentence embeddings for *incremental domain adaptation*, please see our paper: [Kalimuthu et. al (2019)](https://sites.google.com/view/wanlp-2019/home#h.p_Bd2Gh1L0z6cE): **"Incremental Domain Adaptation for Neural Machine Translation in Low-Resource Settings"**.
```

If you use this code, please consider citing our paper:

@article{Kalimuthu2019WANLP,
	author = {Marimuthu Kalimuthu and Michael Barz and Daniel Sonntag}, 
	title = {Incremental Domain Adaptation for Neural Machine Translation in Low-Resource Settings}, 
	booktitle = {Proceedings of The Fourth Arabic Natural Language Processing Workshop (WANLP)},
        venue = {Florence, Italy},
	year = {2019}
}
```

