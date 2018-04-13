import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import csv
import string
import requests
import io
import random
import collections
import tarfile
import urllib.request
from nltk.corpus import stopwords
from sklearn.manifold import TSNE

def load_movie_data(save_folder_name):
    # save_folder_name = 'MovieReview'
    pos_file = os.path.join(save_folder_name, 'rt-polarity.pos')
    neg_file = os.path.join(save_folder_name, 'rt-polarity.neg')
    # Check if files are already downloaded
    if os.path.exists(save_folder_name):
        pos_data = []
        with open(pos_file, 'r') as temp_pos_file:
            for row in temp_pos_file:
                pos_data.append(row)
        neg_data = []
        with open(neg_file, 'r') as temp_neg_file:
            for row in temp_neg_file:
                neg_data.append(row)
    else:  # If not downloaded, download and save
        movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
        stream_data = urllib.request.urlopen(movie_data_url)
        tmp = io.BytesIO()
        data = stream_data.read()
        tmp.write(data)
        tmp.seek(0)
        tar_file = tarfile.open(fileobj=tmp, mode='r:gz')
        pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
        neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')
        # Save pos/neg reviews
        pos_data = []
        for line in pos:
            pos_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
        neg_data = []
        for line in neg:
            neg_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
        tar_file.close()
        # Write to file
        if not os.path.exists(save_folder_name):
            os.makedirs(save_folder_name)
        # Save files
        with open(pos_file, 'w') as pos_file_handler:
            pos_file_handler.write(''.join(pos_data))
        with open(neg_file, 'w') as neg_file_handler:
            neg_file_handler.write(''.join(neg_data))
    texts = pos_data + neg_data
    target = [1] * len(pos_data) + [0] * len(neg_data)
    return (texts, target)

def normalize_text(texts, stops):
    # Lower case
    texts = [x.lower() for x in texts]
    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation)for x in texts]
    # Remove numbers
    texts = [''.join(c for c in x if c not in '0123456789')for x in texts]
    # Remove stopwords and trim extra whitespace
    texts = [' '.join(word for word in x.split() if word not in stops)for x in texts]
    # Trim extra whitespace
    # texts = [' '.join(x.split()) for x in texts]
    return texts

def build_dictionary(sentences, vocabulary_size):
    # Turn sentences which are list of strings into list of words
    split_sentence = [t.split() for t in sentences]
    split_word = [w.split() for s in split_sentence for w in s]
    split_words = []
    for w in split_word:
        [word] = w
        split_words.append(word)
    # Initial list of [word, word_count] for each word, starting with unknown
    count = [('RARE', -1)]
    # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
    # print('vcab_len',len(collections.Counter(split_words).most_common()))
    count.extend(collections.Counter(split_words).most_common(vocabulary_size-1))
    # Now create the dictionary
    word_dict = {}
    # For each word that we want in the dictionary, add it, then make it the value of the prior dictionary length
    for word, word_count in count:
        word_dict[word] = len(word_dict)
    return word_dict

def text_to_numbers(sentences, word_dict):
    # Initialize the returned data
    data = []
    for sentence in sentences:
        word_data = []
        for word in sentence.split(' '):
            if word in word_dict:
                word_data.append(word_dict[word])
            else:
                word_data.append(0)
        data.append(word_data)
    # print('word:', word)
    return data

def generate_batch_data(sentences, batch_size, window_size, method):
    # Fill up data batch
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # Select random sentence to start
        rand_sentence = np.random.choice(sentences)
        # print('rand_sentence', rand_sentence)
        # Generate consecutive windows to look up
        window_sequences = [rand_sentence[max(ix-window_size,0):(ix+window_size+1)] for ix, x in enumerate(rand_sentence)]
        # Denote which element of each window is the center word of interest
        label_indices = [ix if ix < window_size else window_size for ix, x in enumerate(window_sequences)]
        # Pull out center word of interest for each window and create a tuple for each window
        if method=='skip_gram':
            batch_and_labels = [(x[y],x[:y]+x[y+1:]) for x,y in zip(window_sequences, label_indices)]
        # Make it into a big list of tuples (target word, arounding word)
            tuple_data = [(x,y_) for x, y in batch_and_labels for y_ in y ]
            batch, labels = [list(x) for x in zip(*tuple_data)]
            batch_data.extend(batch[:batch_size])
            label_data.extend(labels[:batch_size])
        elif method == 'cbow':
            batch_and_labels = [(x[:y] + x[y + 1:], x[y]) for x, y in zip(window_sequences, label_indices)]
            batch, labels = [x for x in zip(*batch_and_labels)]
            batch_4m = []
            label_4m = []
            for ix, x in enumerate(batch):
                if len(x) == 4:
                    batch_4m.append(x)
                    label_4m.append(labels[ix])
            batch_data.extend(batch_4m[:batch_size])
            label_data.extend(label_4m[:batch_size])
        elif method == 'doc2vec':
            batch_and_labels = [(x[:y] + x[y + 1:], x[y]) for x, y in zip(window_sequences, label_indices)]
            batch, labels = [x for x in zip(*batch_and_labels)]
            batch_4m = []
            label_4m = []
            for ix, x in enumerate(batch):
                if len(x) == 4:
                    batch_4m.append(x)
                    label_4m.append(labels[ix])
            batch_data.extend(batch_4m[:batch_size])
            label_data.extend(label_4m[:batch_size])
        else:
            raise ValueError('Method {} not in implemented yet.'.format(method))
        # Extract batch and labels

    # Trim batch and label at the end
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]
    # Convert to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))
    return (batch_data, label_data)
