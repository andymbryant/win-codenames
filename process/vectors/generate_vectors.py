import pandas as pd
import os
from vectors_helpers import *

def main():

    dirname = os.path.dirname(__file__)

    # GloVe
    # I got this model from here: http://nlp.stanford.edu/data/glove.42B.300d.zip
    # Trained on 42 billion words
    print('Importing GloVe...')
    glove_filepath = os.path.join(dirname, 'data/glove_top_200000.txt')
    df_glove = get_df_from_txt(glove_filepath)

    # I got word2vec from the google repo and loaded it using gensim
    # The first header line had to be removed
    # from gensim.models.word2vec import Word2Vec
    # from gensim.models import KeyedVectors
    # model = KeyedVectors.load_word2vec_format('/Users/andybryant/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
    # model.wv.save_word2vec_format('googlenews.txt')

    # Word2vec
    print('Importing Google...')
    google_filepath = os.path.join(dirname, 'data/googlenews_top_200000.txt')
    df_google = get_df_from_txt(google_filepath)

    # fastText
    # I got this model from here: https://fasttext.cc/docs/en/english-vectors.html
    # It's the wiki-news 1m vectors dataset
    print('Importing FT...')
    ft_filepath = os.path.join(dirname, 'data/fasttext_top_200000.txt')
    df_ft = get_df_from_txt(ft_filepath)

    # Make sets of the different words
    glove_words_final = set(df_glove.index.tolist())
    google_words_final = set(df_google.index.tolist())
    ft_words_final = set(df_ft.index.tolist())
    # Get their intersection - the words that appear3 in all of them
    intersection = glove_words_final.intersection(google_words_final, ft_words_final)
    # Get their union - the words that appear in at least one of them
    union = glove_words_final.union(google_words_final, ft_words_final)
    # The ones that do not appear in all three
    diff = union.symmetric_difference(intersection)
    print(f'Num words that the sets share: {len(intersection)}')
    print(f'Num words that the sets do not share: {len(diff)}')

    # Drop any rows with words that are not present in every dataframe
    df_glove.drop(diff, errors='ignore', inplace=True)
    df_google.drop(diff, errors='ignore', inplace=True)
    df_ft.drop(diff, errors='ignore', inplace=True)

    # Make a multindex dataframe with all of the intersection words in descending order of num appearances
    # This was my first approach, but I abandonded it. It's pretty nifty, but ultimately it's cumbersome to work with
    # in other repos!
    # data = {'glove' : df_glove, 'word2vec' : df_google, 'fasttext': df_ft}
    # midx = pd.MultiIndex.from_product([list(df_glove.index), data.keys()])
    # res = pd.concat(data, axis=0, keys=data.keys()).swaplevel(i=0,j=1,axis=0)
    # df_all_vectors = res.sort_index(level=0).reindex(midx)

    print('Generating output...')
    # Instead I exported every vector file individually
    glove_output = os.path.join(dirname, 'output/glove_vectors.pkl')
    df_glove.to_pickle(glove_output, protocol=3)

    google_output = os.path.join(dirname, 'output/google_vectors.pkl')
    df_google.to_pickle(google_output, protocol=3)

    ft_output = os.path.join(dirname, 'output/fasttext_vectors.pkl')
    df_ft.to_pickle(ft_output, protocol=3)

if __name__ == '__main__':
    main()