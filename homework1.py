
# %%
import pandas as pd
from fastai.data.external import URLs, untar_data
import string

# %%
from collections import defaultdict
import re

def remove_contraction(text):
    text = re.sub(r'n\'t', ' not', text)
    text = re.sub(r'\'re', ' are', text)
    text = re.sub(r'\'s', ' is', text)
    text = re.sub(r'\'d', ' would', text)
    text = re.sub(r'\'ll', 'will', text)
    text = re.sub(r'\'t', ' not', text)
    text = re.sub(r'\'ve', ' have', text)
    text = re.sub(r'\'m', ' am', text)

    return text

def word_count(words):
    count = defaultdict(int)

    for word in words:
        count[word] += 1

    return count

def clean_text(df, column):
    # REMOVE PUNCTUATION
    pattern = r'\.*<\s*br\s*\/>|[".,\/#!$%\^&\*;:{}=\-_`~()]'

    tokenized_df = df.replace(to_replace=pattern, value='', regex=True)

    # LOWERCASE
    tokenized_df[column] = tokenized_df[column].apply(str.lower)

    # contractions
    tokenized_df[column] = tokenized_df[column].apply(remove_contraction)

    return tokenized_df

def tokenizer(df, column):
    # words
    sentences = df[column].to_list()
    sentences = ' '.join(sentences)
    words = sentences.split(' ') 
    
    word_counts = word_count(words)
    sorted_word_counts = {k: v for k, v in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)}
    
    ret = pd.DataFrame(columns=['word', 'freq', 'positive', 'negative', 'freq_pos', 'freq_neg'])
    
    ret['word'] = list(sorted_word_counts.keys())
    ret['freq'] = list(sorted_word_counts.values())

    # check if in pos class
    sentences = df.loc[df['label'] == 'positive'][column].to_list()
    sentences = ' '.join(sentences)
    words = sentences.split(' ') 
    pos = []
    freqs =[]
    for word in ret['word'].to_list():
        c = words.count(word)
        freq = c
        if c >0: 
            pos.append(True)
        else:
            pos.append(False)
        freqs.append(freq)
    ret['freq_pos'] = freqs
    
    
    # check if in neg class
    sentences = df.loc[df['label'] == 'negative'][column].to_list()
    sentences = ' '.join(sentences)
    words = sentences.split(' ') 
    
    neg = []
    freqs =[]

    for word in ret['word'].to_list():
        c = words.count(word)
        freq = c
        if c >0: 
            neg.append(True)
        else:
            neg.append(False)
        freqs.append(freq)
    ret['freq_neg'] = freqs

    ret['positive'] = pos
    ret['negative'] = neg
    return ret
    # return list(sorted_word_counts)[:len_vocab]

# %
# %% [markdown]
# ### 2c (5 points). 
# - Implement on your own and train a Naive Bayes sentiment classifier in the _training set_. Requirements: use log scales and add-one smoothing.
# - Report your model performances on the _validation set_, with the 3 vocabs your created in 2b, respectively.

# %%
import numpy as np
from tqdm import tqdm
def build_l_table(df, len_vocab):
    liklihood_table = defaultdict(dict)

    top_n = df.iloc[:len_vocab, :]

    total_pos = top_n['freq_pos'].sum()
    total_neg = top_n['freq_neg'].sum()

    for word in tqdm(top_n['word'], desc='building liklihood table'):
        pos_lik = int(top_n.loc[top_n['word'] == word]['freq_pos'])
        neg_lik = int(top_n.loc[top_n['word'] == word]['freq_neg'])

        liklihood_table[word]['pos'] = np.log((pos_lik + 1) / (total_pos + len_vocab))
        liklihood_table[word]['neg'] = np.log((neg_lik + 1) / (total_neg + len_vocab))
    
    return liklihood_table, (total_pos, total_neg)



def priors(df):
    N = len(df)

    positive_prior = df['label'].value_counts()['positive'] / N
    negative_prior = df['label'].value_counts()['negative'] / N

    return np.log(positive_prior), np.log(negative_prior)

def nb_preprocess(df_train, word_counts, len_vocab):
    p_prior, n_prior = priors(df_train)

    likelihood_table, total_for_classes = build_l_table(word_counts, len_vocab)

    return (p_prior, n_prior), likelihood_table, total_for_classes

def naive_bayes(priors, likelihoods, total_classes, eval_sentences):
    preds = []

    p_prior, n_prior = priors


    for sentence in tqdm(eval_sentences):

        pos_log_lik, neg_log_lik = 0, 0
        for word in sentence.split(' '):
            if word not in list(likelihoods.keys()):
                pos_log_lik += np.log(1 / total_for_classes[0])
                neg_log_lik += np.log(1 / total_for_classes[1])
            else:
                if 'pos' not in list(likelihoods[word].keys()):
                    pos_log_lik += np.log(1 / total_for_classes[0])
                else:
                    pos_log_lik += likelihoods[word]['pos']

                if 'neg' not in list(likelihoods[word].keys()):
                    neg_log_lik += np.log(1 / total_for_classes[1])
                else:
                    neg_log_lik += likelihoods[word]['neg']

        
        pos_pred = p_prior + pos_log_lik
        neg_pred = n_prior + neg_log_lik

        pred = 'positive' if pos_pred > neg_pred else 'negative'

        preds.append(pred)

    return preds



# %%
path = untar_data(URLs.IMDB_SAMPLE)

# %%
df = pd.read_csv(path/'texts.csv')

df_valid = df.loc[df['is_valid']]
df_train = df.loc[df['is_valid'] == True]

df_train, df_valid = clean_text(df_train, 'text'), clean_text(df_valid, 'text')
word_counts = tokenizer(df_train, 'text')

p, l, total_for_classes = nb_preprocess(df_train, word_counts, 1000)
print(naive_bayes(p, l, total_for_classes, df_valid['text'].to_list()))



