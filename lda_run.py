from gensim.test.utils import datapath
from gensim import models, corpora
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from pprint import pprint
import numpy as np
np.random.seed(2018)
import nltk

import json

try:
    nltk.find('corpora')
except:
    nltk.download('wordnet')

#load saved dreams model 
lda_dreams = models.ldamodel.LdaModel.load("models/lda_dreams_model")
dreams_dict = corpora.Dictionary.load("models/lda_dreams_dictionary")

#Preprocessing by lemmatize and stemmin steps
def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

#Applies unseen doc to model and returns associated emotion
def applyModel(model, dictionary, unseen):
    bow_vector = dictionary.doc2bow(preprocess(unseen))
    best_topic = [-float("inf"), None]

    for index, score in sorted(model[bow_vector], key=lambda t: -1*t[1]):
        if score > best_topic[0]:
            best_topic = [score, index]

    emotions_json = None
    with open("models/emotions.json", mode="r") as file:
        contents = file.read()
        emotions_json = json.loads(contents)

    return emotions_json[str(best_topic[1])]

def lda_dreamModel(unseen):
    return applyModel(lda_dreams, dreams_dict, unseen)

