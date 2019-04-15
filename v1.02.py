
# coding: utf-8

# In[ ]:


import glob
import errno
import codecs
path1 = 'C:/Users/NoT/Desktop/AI/Project/maas_dataset/train/train/pos_2/*.txt'
path2 = 'C:/Users/NoT/Desktop/AI/Project/maas_dataset/train/train/neg_2/*.txt'
labels, texts = [], []
val_x, val_y = [],[]
files = glob.glob(path1)
# texts.append("post")
# labels.append("tags")
for name in files:
    try:
        with codecs.open(name, 'r', encoding='utf-8') as f:
            str = f.read()
            # str = re.sub(' +', ' ', str)
            str = " ".join(str.split())
            labels.append("positive")
            texts.append(str)


    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path2)
for name in files:
    try:
        with codecs.open(name, 'r', encoding='utf-8') as f:
            str = f.read()
            #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            labels.append("negetive")
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise



########################Data cleaning##################

import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
#For removing html tags
i=0
for strr in texts:

    strr = texts[i]
    strr = BeautifulSoup(strr).get_text()


    # Emotional symbols may affect the meaning of the review
    smileys = """:-) :) :o) :] :3 :c) :> =] 8) =) :} :^)
                :D 8-D 8D x-D xD X-D XD =-D =D =-3 =3 B^D :( :/ :-( :'( :D :P""".split()
    smiley_pattern = "|".join(map(re.escape, smileys))

    # [^] matches a single character that is not contained within the brackets
    # re.sub() replaces the pattern by the desired character/string

    # Check to see how we need to perform cleanup
    if removeNumbers and removeSmileys:
        reviewText = re.sub("[^a-zA-Z]", " ", strr)
    elif removeSmileys:
        reviewText = re.sub("[^a-zA-Z0-9]", " ", strr)
    elif removeNumbers:
        reviewText = re.sub("[^a-zA-Z" + smiley_pattern + "]", " ", strr)
    else:
        reviewText = re.sub("[^a-zA-Z0-9" + smiley_pattern + "]", " ", strr)

    texts[i] = strr
    i=i+1



#print(texts[10])
#print(labels[10])
print("File Reading Finished")


# In[ ]:


import pandas as pd
import numpy as np

df = pd.DataFrame({'texts':texts, 'labels': labels})
df.head()


# In[ ]:


df['category_id'] = df['labels'].factorize()[0]

category_id_df = df[['labels', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'labels']].values)

sentiment_counts = df.labels.value_counts()
print(sentiment_counts)




# In[ ]:


import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def normalizer(tweet):

    tokens = nltk.word_tokenize(tweet)
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas


pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
df['normalized_data'] = df.texts.apply(normalizer)
#print(df[['data','normalized_data']].head())


from nltk import ngrams
def ngrams(input_list):
    #onegrams = input_list
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
    return bigrams+trigrams
df['grams'] = df.normalized_data.apply(ngrams)


import collections
def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt


import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(ngram_range=(1,2), max_features=10000)

vectorized_data = count_vectorizer.fit_transform(df.texts)
indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))

#
# #TF_IDF
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn import metrics
# from sklearn.feature_extraction.text import TfidfVectorizer
# tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df = 0, max_features = 100000, stop_words = 'english')
# tfidf_matrix = tf.fit_transform(df.texts)
# print(tfidf_matrix.shape)



def sentiment2target(sentiment):
    return {
        'positive': 1,
        'negetive': 0,

    }[sentiment]



targets = df.labels.apply(sentiment2target)

from sklearn.model_selection import train_test_split
data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.2, random_state=0)
data_train_index = data_train[:,0]
data_train = data_train[:,1:]
data_test_index = data_test[:,0]
data_test = data_test[:,1:]





print("svm started")
#svm

from sklearn import svm
#from sklearn.multiclass import OneVsRestClassifier
#clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
#clf_output = clf.fit(data_train, targets_train)

from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.grid_search import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(data_train, targets_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(data_test)

print(classification_report(targets_test,grid_predictions))


#print(clf.score(data_test, targets_test))
