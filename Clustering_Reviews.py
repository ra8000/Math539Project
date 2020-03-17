
#Libraries and Definitions
import os
import spacy
import pytextrank
import numpy as np
import string
import matplotlib.pyplot as plt
from string import digits
%matplotlib inline
plt.rcParams['figure.figsize'] = [15, 7]
import pandas as pd
from spacy.lang.en import English
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering, KMeans
from collections import Counter

####Reset to project folder
os.chdir("/home/n/Dropbox (CSU Fullerton)/539 Consulting/ReviewData")

nlp = spacy.load("en_core_web_sm")

# add PyTextRank to the spaCy pipeline
tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)


punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS
parser = English()
digs = string.digits

translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))

def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().translate(translator).strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations]
    mytokens = [''.join(filter(lambda x: x.isalpha(),word)) for word in mytokens]
#     mytokens = [word for word in mytokens filter(lambda x: x.isalpha(), word)]
    return mytokens

class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

def counts(x):
    uni, cnt = np.unique(x, return_counts=True)
    data = np.array([uni, cnt]).T
    return pd.DataFrame(
        data=data[0:, 0:], index=data[0:, 0], columns=["Factors", "Count"]
    )

def normalize(comment, lowercase=True, remove_stopwords=True):
    if lowercase:
        comment = comment.lower()
    comment = ''.join([i for i in comment if i not in digits and i not in punctuations]).strip()
    comment = nlp(comment)
    lemmatized = list()
    for word in comment:
        lemma = word.lemma_.strip()
        if lemma:
            if not remove_stopwords or (remove_stopwords and lemma not in stop_words):
                lemmatized.append(lemma)
    return " ".join(lemmatized)

def vector(sent):
    x = sent.lower()
    x = ''.join([i for i in x if i not in digits and i not in punctuations]).strip()
    doc = nlp(x)
    token_list = []
    filtered_sentence = []
    for token in doc:
        token_list.append(token.text)
    for word in token_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop==False:
            filtered_sentence.append(word)
    v = nlp(" ".join(filtered_sentence)).vector
    return v


#####Read Data File and take Sample
#%%
reviews = pd.read_csv("combined.csv")
samples = reviews.sample(n=1000,random_state=  1)




#Kmeans with BOW Vector or tfidf
#%%
X = samples["Review"].fillna("").to_numpy()
bow_vector = CountVectorizer(tokenizer=spacy_tokenizer,ngram_range=(1,1))
tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)


#%%
X_bow = bow_vector.fit_transform(X)
X_dense = X_bow.todense()

clusterk = KMeans(n_clusters=10)
X_clustk = clusterk.fit(X_dense)


#Counts and Plots
#%%
c = counts(X_clustk.labels_)
plt.bar(c.Factors,c.Count)
print(c)

#Reattach cluster
samples2 = samples.assign(Class = X_clustk.labels_)

#PyTextRank Rank Phrases for each class
#%%
n = range(0,21)
for i in n:
    print(i)
    doc = nlp(normalize(pd.Series.to_string(samples2[samples2.Class==i].Review)))
    for p in doc._.phrases[0:10]:
        print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text))






######  Using Spacy's Vectorizer  On whole review which averages across all words
#%%
X_spvec = (samples.Review
            .apply(vector)
            .apply(pd.Series)
            .fillna(0) )



X_clustk_spvec = clusterk.fit(X_spvec)
c = counts(X_clustk_spvec.labels_)
plt.bar(c.Factors,c.Count)
print(c)

samples3 = samples.assign(Class = X_clustk_spvec.labels_)

#%%
n = range(0,21)
for i in n:
    print(i)
    doc = nlp(normalize(pd.Series.to_string(samples3[samples3.Class==i].Review)))
    for p in doc._.phrases[0:10]:
        print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text))




for i in n:
    print(i)
    doc = nlp(normalize(pd.Series.to_string(samples3[samples3.Class==i].Review)))
    words = [token.text for token in doc]
    top_words = Counter(words).most_common(N)
    for word, frequency in top_words:
        print("%s %d" % (word, frequency))
