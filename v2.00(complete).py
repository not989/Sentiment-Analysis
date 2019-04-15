
# coding: utf-8

# In[15]:



# coding: utf-8

# In[104]:



# In[ ]:

import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV

def sentimentToWordlist(rawReview, removeStopwords=False, removeNumbers=False, removeSmileys=False):

    # use BeautifulSoup library to remove the HTML/XML tags (e.g., <br />)
    reviewText = BeautifulSoup(rawReview).get_text()

    # Emotional symbols may affect the meaning of the review
    smileys = """:-) :) :o) :] :3 :c) :> =] 8) =) :} :^):D 8-D 8D x-D xD X-D ( ) XD =-D =D =-3 =3 B^D :( :/ :-( :'( :D :P""".split()
    smiley_pattern = "|".join(map(re.escape, smileys))

    # [^] matches a single character that is not contained within the brackets
    # re.sub() replaces the pattern by the desired character/string

	# Check to see how we need to perform cleanup
    if removeNumbers and removeSmileys:
        reviewText = re.sub("[^a-zA-Z]", " ", reviewText)
    elif removeSmileys:
        reviewText = re.sub("[^a-zA-Z0-9]", " ", reviewText)
    elif removeNumbers:
        reviewText = re.sub("[^a-zA-Z" + smiley_pattern + "]", " ", reviewText)
    else:
        reviewText = re.sub("[^a-zA-Z0-9" + smiley_pattern + "]", " ", reviewText)

    # split in to a list of words
    words = reviewText.lower().split()

    if removeStopwords:
        # create a set of all stop words
        stops = set(stopwords.words("english"))
        # remove stop words from the list
        words = [w for w in words if w not in stops]

    return words







import glob
import errno
import codecs
path1 = 'C:/Users/NoT/Desktop/AI/Project/maas_dataset/train/train/pos_1/*.txt'
path2 = 'C:/Users/NoT/Desktop/AI/Project/maas_dataset/train/train/neg_1/*.txt'
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

print(len(texts))

# i=0
# for strr in texts:

#     strr = texts[i]
#     texts[i] = sentimentToWordlist(strr)
#     i=i+1


clean_data = []
#Loop counter
numRevs = len(texts)

for i in range(0,numRevs):

    #Clean each review> Please look at the definition of the sentimentToWordlist function in the preproc.py script
    clean_data.append(" ".join(sentimentToWordlist(texts[i])))


print(len(clean_data))
print(len(labels))
print("File Reading Finished")


# In[105]:


print("Defining TFIDF Vectorizer")

tfIdfVec = TFIV(
                    min_df=3, # When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold
                    max_features=10000, # If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
                    strip_accents='unicode', # Remove accents during the preprocessing step. 'ascii' is a fast method that only works on characters that have an direct ASCII mapping.
                                             # 'unicode' is a slightly slower method that works on any characters.
                    analyzer='word', # Whether the feature should be made of word or character n-grams.. Can be callable.
                    token_pattern=r'\w{1,}', # Regular expression denoting what constitutes a "token", only used if analyzer == 'word'.
                    ngram_range=(1,5), # The lower and upper boundary of the range of n-values for different n-grams to be extracted.
                    use_idf=1, # Enable inverse-document-frequency reweighting.
                    smooth_idf=1, # Smooth idf weights by adding one to document frequencies.
                    sublinear_tf=1, # Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
                    stop_words = 'english' # 'english' is currently the only supported string value.
                )


# In[106]:


print("Fitting")

tfIdfVec.fit(clean_data) # Learn vocabulary and idf from training set.


# In[107]:


clean_data = tfIdfVec.transform(clean_data) # Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform).


# In[108]:


import pandas as pd
import numpy as np

df = pd.DataFrame({'clean_data':clean_data, 'labels': labels})
df.head()

print("Fitting and transforming done")

print("Label testing")
##label work begins here
def sentiment2target(sentiment):
    return {
        'positive': 1,
        'negetive': 0,

    }[sentiment]

print(df.head())

targets = df.labels.apply(sentiment2target)
print (targets)


# In[16]:


from sklearn.model_selection import train_test_split
data_train, data_test, targets_train, targets_test = train_test_split(clean_data, targets, test_size=0.2, random_state=0)
data_train_index = data_train[:,0]
data_train = data_train[:,1:]
data_test_index = data_test[:,0]
data_test = data_test[:,1:]


# In[110]:


#print(data_train)


# In[111]:



from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.grid_search import GridSearchCV



# In[112]:


param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(data_train, targets_train)


# In[113]:



# In[20]:


grid.best_params_


# In[18]:


grid.best_estimator_


# In[19]:


grid_predictions = grid.predict(data_test)


# In[114]:


print(classification_report(targets_test,grid_predictions))

