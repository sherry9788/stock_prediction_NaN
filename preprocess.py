# Train the model using Word2vec

import gensim
from gensim.models import Word2Vec
from gensim.corpora import wikicorpus

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def try_hash(model, word):
    try:
        return model[word]
    except:
        return None

# Twitter Text Preprocessing

import preprocessor as p
import re
from nltk.corpus import stopwords
import numpy as np

def preprocess(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split 
    words = letters_only_text.lower().split()

    # remove stopwords
    #stopword_set = set(stopwords.words("english"))
    #meaningful_words = [w for w in words if w not in stopword_set]

    # join the cleaned words in a list
    cleaned_word_list = " ".join(words)

    return cleaned_word_list


filenames = ['11.29.txt','11.30.txt','12.1.txt','12.2.txt','12.3.txt','12.4.txt','12.5.txt','12.6.txt','12.7.txt','12.8.txt','12.9.txt','12.10.txt']


total_vectors=[]
label_counter=0

for fname in filenames:
    total_preprocessed=[]
    with open(fname) as f:
        content = f.readlines()
    tweets = [x.strip() for x in content] 

    for tweet in tweets:
        clean=p.clean(tweet)
        total_preprocessed.append(preprocess(clean))
        


    # Get vector for each tweet with labels 
    vectors=[]

    for i in range(len(total_preprocessed)):
        text=total_preprocessed[i]
        if text is '':
            continue
        tweet_vec=np.zeros(301)
        words = text.split()
        for word in words:
            if word in model.wv.vocab:
                tweet_vec+= numpy.append(model.wv[word],0)
        if label_counter<=877:
            tweet_vec[300]=labels[label_counter]
            label_counter+=1
        vectors.append(tweet_vec) 
    total_vectors.append(vectors)
    
for i in range(len(total_vectors)):
    numpy.savetxt(str(i)+'.csv', total_vectors[i], delimiter=",")



