import csv
import numpy as np
import pandas as pd
import nltk
nltk.download('words')
from nltk.corpus import stopwords, words

# Get stop words
nltk_stopwords = set(stopwords.words())
nltk_words = set(words.words())

def get_dict_from_txt(filepath):
    '''Helper function for opening, reading, and concerting .txt file of vectors.
    Returns a dictionary of the embeddings vectors.'''
    with open(filepath, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ',quoting=csv.QUOTE_NONE)
        embeddings_dict = {line[0]: np.array(list(map(float, line[1:]))) for line in reader}
    return embeddings_dict

def get_words_from_keys(keys):
    '''Helper function for filtering stopwords and non-english words from a list of keys.
    Returns a list of strings.'''
    words = []
    for key in keys:
        if key not in nltk_stopwords and key in nltk_words:
            words.append(key)
    return words

def get_filtered_dict(old_dict, words_to_keep):
    '''Helper function for filtering a dictionary, keeping a specific list of words.
    Returns a dictionary with words as keys and vectors as values.'''
    new_dict = dict()
    for w in words_to_keep:
        val = old_dict.get(w)
        # If dictionary has the word as a key, store its value
        if val is not None:
            new_dict[w] = val
    return new_dict

def get_df_from_txt(filepath):
    '''Helper function for generating dataframe with words as the index and the dimensions as columns.
    Returns that dataframe.'''
    vec_dict = get_dict_from_txt(filepath)
    vec_keys = list(vec_dict.keys())
    vec_words = get_words_from_keys(vec_keys)
    vec_dict_filtered = get_filtered_dict(vec_dict, vec_words)
    return pd.DataFrame.from_dict(vec_dict_filtered, orient="index")
