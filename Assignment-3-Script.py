# Run in python console
import nltk;

nltk.download('stopwords')
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

spacy.load("en_core_web_sm")
from spacy.lang.en import English

parser = English()

# Set # of topics
num_topics=20

# NLTK Stop words
from nltk.corpus import stopwords


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in stop_words]
    # tokens = [get_lemma(token) for token in tokens]
    return tokens


def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def getDocTopicWeight(lda_model_in, new_doc):
    doc = prepare_text_for_lda(new_doc)
    new_doc_bow = id2word.doc2bow(doc)
    return lda_model_in.get_document_topics(new_doc_bow)


# Define functions for stopwords,
def remove_stopwords(texts_in):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_in]


stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

df = pd.read_csv('230 VP assertions corpus - Group 6.csv')
df.head()

# Convert to list
data = df.content.values.tolist()

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

pprint(data[:1])

# Remove Stop Words
data_nostops = remove_stopwords(data)
pprint(data[:1])

# Create Dictionary
id2word = corpora.Dictionary(data_nostops)
# Create Corpus
texts = data_nostops
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1])
id2word[0]
# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

top_words_per_topic = []
for t in range(lda_model.num_topics):
    top_words_per_topic.extend([(t, ) + x for x in lda_model.show_topic(t, topn = 10)])

pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P']).to_csv("top_words.csv")


# Visualize the topics
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
# git print(vis)

df_doctop = pd.DataFrame(np.zeros((len(data),num_topics),dtype=float), index=np.arange(len(data)), columns=[list(range(num_topics))])
count=0

for a in data:
    doc_topic_weights = getDocTopicWeight(lda_model, a)
    for b in doc_topic_weights:
        # df_doctop = pd.DataFrame(b columns=range(20))
        df_doctop.at[count, b[0]] = b[1]

        # df_doctop.append(b[1]: doc_topic_weights)
    # print(doc_topic_weights[:][1])
    count=count+1

# Heatmap of weights
plt.pcolor(df_doctop)
plt.yticks(np.arange(0, len(df_doctop.index), 1), df_doctop.index)
plt.xticks(np.arange(0, len(df_doctop.columns), 1), df_doctop.columns)
plt.show()

df_doctop['Assertion']=data
df_doctop.to_csv("document_topic_weights.csv")

