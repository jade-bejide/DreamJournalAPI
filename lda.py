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

#load in dreams data
data = pd.read_csv('dreams.csv', error_bad_lines=False, nrows=14290)
data_text = data[['content']]
#data_text['index'] = data_text.index
documents = data_text


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

processedEntry = documents['content'].map(preprocess)

dictionary = gensim.corpora.Dictionary(processedEntry)

dictionary.save("models/lda_dreams_dictionary")

#dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processedEntry]

#TF-IDF model
tfidf = gensim.models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

#Train using LDA
x = 10


if __name__ == '__main__':
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=x,
                                           id2word=dictionary, passes=2, workers=2)

    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=x,
                                                 id2word=dictionary, passes=2, workers=4)

    
    lda_model.save("models/lda_dreams_model")
    lda_model_tfidf.save("models/lda_dreams_model_tfidf")
    
    unseen_doc = "A swarm of bumble bees chased me through the corridors"
    bow_vector = dictionary.doc2bow(preprocess(unseen_doc))

    for index, score in sorted(lda_model[bow_vector], key=lambda t: -1*t[1]):
        print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 50)))

    emotions = []
    emotionMap = {} #maps emotion to topic-score list

    for entry in range(x):
        emotionMap[str(entry)] = [None, -float("inf")]

    with open("models/emotions.txt", mode="r") as file:
        for line in file:
            if '\n' in line: line = line.strip('\n')
            emotions.append(line)

    seen_emotions = set()
    for emotion in emotions:
        bow_vector = dictionary.doc2bow(preprocess(emotion))

        for index, score in sorted(lda_model[bow_vector], key=lambda t: -1*t[1]):
            if str(index) in emotionMap:
                if score > emotionMap[str(index)][1]:
                    emotionMap[str(index)] = [emotion, score]
                    seen_emotions.add(emotion)
                elif score == emotionMap[str(index)][1] and emotionMap[str(index)][0] in seen_emotions:
                    emotionMap[str(index)][0] += (", " + emotion)
                    seen_emotions.add(emotion)

    for entry in emotionMap:
        emotionMap[entry] = emotionMap[entry][0]

    emotionJson = json.dumps(emotionMap, indent = 4)

    with open("models/emotions.json", "w") as outfile:
        outfile.write(emotionJson)

    
            

