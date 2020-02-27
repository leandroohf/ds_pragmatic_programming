# * *************************************************************************
#   Programmer[s]: Leandro Fernandes
#   email: leandroohf@gmail.com
#   Program: text_utils
#   Commentary: My utils for helping with text processing
#   Date: February 6, 2020
#
#   The author believes that share code and knowledge is awesome.
#   Feel free to share and modify this piece of code. But don't be
#   impolite and remember to cite the author and give him his credits.
# * *************************************************************************

import numpy as np
import pandas as pd

from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from typing import List, Optional


def get_max_word_len(sentence: str):

    s1 = word_tokenize(sentence)
    sentence_len = len(s1)

    len_of_largest_word = 0
    largest_word = ''
    # O(n)
    for word in s1:

        if(len(word) > len_of_largest_word):

            len_of_largest_word = len(word)
            largest_word = word

    return largest_word, len_of_largest_word, sentence_len

def create_corpus_table(corpus: list) -> pd.DataFrame:

    corpus_df = pd.DataFrame({"doc": corpus})

    corpus_df['largest_word'] = corpus_df.doc.map(lambda s: get_max_word_len(s)[0])
    corpus_df['len_of_largest_word'] = corpus_df.doc.map(lambda s: get_max_word_len(s)[1])
    corpus_df['doc_len'] = corpus_df.doc.map(lambda s: get_max_word_len(s)[2])

    tokenizer = RegexpTokenizer(r'\w+')
    corpus_df['words'] = corpus_df.doc.map(tokenizer.tokenize)

    return corpus_df

def summarize_corpus(corpus_df: pd.DataFrame ):

    all_words = [word for tokens in corpus_df["words"] for word in tokens]
    n_words = len(all_words)
    vocab_size = len(set(all_words))

    n_docs = corpus_df.shape[0]

    idx = corpus_df["len_of_largest_word"].idxmax()
    largest_word, len_of_largest_word = corpus_df[['largest_word','len_of_largest_word']].iloc[idx]

    min_doc_len = corpus_df["doc_len"].min()
    median_doc_len = corpus_df["doc_len"].median()
    mean_doc_len = corpus_df["doc_len"].mean()
    max_doc_len = corpus_df["doc_len"].max()

    word_summary = pd.DataFrame({'n_words': n_words, 'vocab_size': vocab_size,
                                 'largest_word': largest_word, 'len_of_largest_word': len_of_largest_word }, index=[0])

    doc_summary = pd.DataFrame({'n_docs': n_docs, 'min_doc_len': min_doc_len,'mean_doc_len': mean_doc_len,
                                'median_doc_len': median_doc_len, 'max_doc_len': max_doc_len}, index=[0])

    return word_summary, doc_summary

def get_word_frequency_table(corpus: list, *args, **kargs) -> pd.DataFrame:
    """ Returns DataFrame with the number of times a word is mentioned in all documents.
    Ex: word_freq = 5, word = 'film' means that the word 'film' was mentioned 5 times in all documents
    word_freq is not tf because tf is relative to the document or sentence. This is global  word frequency 
    """

    vectorizer = CountVectorizer(*args, binary=False, **kargs )

    # sparse matrix
    # doc x word count matrix
    # #docs, #words = bag_of_words.shape
    bag_of_words = vectorizer.fit_transform(corpus)

    word_freq = bag_of_words.sum(axis=0)

    word_freq_df = pd.DataFrame({'word_freq': word_freq.tolist()[0]})
    words = vectorizer.get_feature_names()

    word_freq_df['word'] = word_freq_df.apply(lambda r: words[r.name],axis=1)

    # sort by frequency of word
    word_freq_df.sort_values(by='word_freq',ascending=False, inplace=True)

    return word_freq_df

def get_doc_frequency_by_word_table(corpus: list, *args, **kargs) -> pd.DataFrame:
    """ Returns DataFrame with the number of documents that contains the words (df) and the idf  per word
    Ex: doc_freq = 5 (df), word = 'film' means that there are 5 documents that contains the word film in the corpus
    """
    vectorizer = CountVectorizer(*args, binary=True, **kargs )

    # sparse matrix
    # doc x word matrix
    # #docs, #words = X.shape
    doc_word_matrix = vectorizer.fit_transform(corpus)

    doc_freq = doc_word_matrix.sum(axis=0)

    doc_freq_df = pd.DataFrame({'doc_freq': doc_freq.tolist()[0]})
    words = vectorizer.get_feature_names()

    doc_freq_df['idf'] = doc_freq_df.doc_freq.map(lambda x: 1.0/x)
    doc_freq_df['word'] = doc_freq_df.apply(lambda r: words[r.name],axis=1)

    # sort by frequency of word
    doc_freq_df.sort_values(by='doc_freq',ascending=False, inplace=True)

    return doc_freq_df

def get_tfidf_table(corpus: list, *args, **kargs) -> pd.DataFrame:
    """Returns statistics for tfidf in a corpus per word"""

    vectorizer = TfidfVectorizer(*args, **kargs )

    # sparse matrix
    # doc x word matrix
    # #docs, #words = X.shape
    tfidf_of_words = vectorizer.fit_transform(corpus)

    words = vectorizer.get_feature_names()

    tfidf_df = pd.DataFrame({'words': words, 'min': np.min(tfidf_of_words,axis=0), 
                             'mean': np.mean(tfidf_of_words,axis=0),
                             'median': np.median(tfidf_of_words,axis=0),
                             'max': np.max(tfidf_of_words,axis=0)})


    # sort by frequency of word
    tfidf_df.sort_values(by='max',ascending=False, inplace=True)

    return tfidf_df

def clean_text_data(sentences: str, large_words: int , my_stopped_words = Optional[List[str]]) -> str:

    pass
