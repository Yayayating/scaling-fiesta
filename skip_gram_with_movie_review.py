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
import pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
sess = tf.Session()
batch_size = 100
embedding_size = 200
vocabulary_size = 10000
generations = 100000
print_loss_every = 2000
num_sampled = int(batch_size/2)
window_size = 2
stops = stopwords.words('english')
stops_append = ['one', 'much', 'even', 'little', 'make', 'enough', 'never', 'makes', 'may', 'us', 'doesnt', 'would', 'theres', 'could', 'really', 'made', 'many', 'thats', 'still', 'isnt', 'every', 'two', 'without', 'though', 'might', 'also', 'another', 'ever', 'dont', 'seems', 'less', 'often', 'almost', 'cant', 'yet', 'quite', 'youre', 'rather', 'de', 'take', 'despite', 'takes', 'seem', 'youll', 'making', 'bit', 'away', 'need', 'always', 'whose', 'actually', 'nearly', 'around', 'hes', 'goes', 'done', 'turns', 'although', 'three', 'wont', 'whats', 'else', 'put', 'youve', 'along', 'whether', 'either', 'neither', 'didnt', 'im']
for i in stops_append:
    stops.append(i)
print_valid_every = 5000
valid_words = ['cliche', 'love', 'hate', 'silly', 'sad']
def load_movie_data():
    save_folder_name = 'MovieReview'
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
    else: # If not downloaded, download and save
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
            pos_data.append(line.decode('ISO-8859-1').encode('ascii', errors = 'ignore').decode())
        neg_data = []
        for line in neg:
            neg_data.append(line.decode('ISO-8859-1').encode('ascii', errors = 'ignore').decode())
        tar_file.close()
        # Write to file
        if not os.path.exists(save_folder_name):
            os.makedirs(save_folder_name)
        # Save files
        with open(pos_file, 'w') as pos_file_handler:
            pos_file_handler.write(''.join(pos_data))
        with open(neg_file, 'w') as neg_file_handler:
            neg_file_handler.write(''.join(neg_data))
    texts = pos_data +neg_data
    target = [1]*len(pos_data) + [0]*len(neg_data)
    return(texts, target)
texts, target = load_movie_data()
# print('texts_len',len(texts))
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
texts = normalize_text(texts, stops)
# print(texts[0:2]) # 输出0，1
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) >2]
texts = [x for x in texts if len(x.split()) >2]
# 构建词汇表
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
word_dict = build_dictionary(texts, vocabulary_size)
word_dict_rev = dict(zip(word_dict.values(), word_dict.keys()))
# print(word_dict_rev)
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
text_number = text_to_numbers(texts, word_dict)
# print('texts:',texts[0:50])
valid_examples = [word_dict[x] for x in valid_words]
# print(text_number)
def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
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
        else:
            raise ValueError('Method {} not in implemented yet.'.format(method))
        # Extract batch and labels
        batch, labels = [list(x) for x in zip(*tuple_data)]
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])
    # Trim batch and label at the end
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]
    # Convert to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))
    return (batch_data, label_data)

# aa,bb = generate_batch_data(text_number, batch_size, window_size, method='skip_gram')
# print(aa,bb)
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0,1.0))
# Create data/target placeholders
x_inputs = tf.placeholder(tf.int32, shape=[batch_size])
y_target = tf.placeholder(tf.int32, shape=[batch_size,1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
# Lookup the word embedding:
embed = tf.nn.embedding_lookup(embeddings, x_inputs)
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,biases=nce_biases,inputs=embed,labels=y_target,num_sampled=num_sampled,num_classes=vocabulary_size))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
# 创建函数查找验证单词周围的单词
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
normalized_embeddings = embeddings/norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
# 声明优化器函数

init = tf.global_variables_initializer()
sess.run(init)
# 迭代训练词嵌套，打印出损失函数和验证单词集单词的最接近的单词
loss_vec = []
loss_x_vec = []
# aa = []
for i in range(generations):
    batch_inputs, batch_labels = generate_batch_data(text_number, batch_size, window_size)
    feed_dict = {x_inputs:batch_inputs, y_target:batch_labels}
    # print(batch_inputs, batch_labels)
    # Run the train step
    sess.run(optimizer, feed_dict=feed_dict)
    # print('embeddings',sess.run(embeddings, feed_dict=feed_dict))
    # Return the loss
    if (i+1)%print_loss_every == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i+1)
        print("Loss at step {} : {} ".format(i+1, loss_val))
    # Validation: Print some random words and top 5 related words
    if(i+1)%print_valid_every == 0:
        sim = sess.run(similarity, feed_dict=feed_dict)
        # print('sim_num',sim[0, :].argsort()[0:50])
        # print('sim',len(sim[0]))
        for j in range(len(valid_words)):
            valid_word = word_dict_rev[valid_examples[j]]
            top_k = 5
            nearest = (-sim[j, :]).argsort()[1:top_k+1]
            log_str = "Nearest to {}:".format(valid_word)
            for k in range(top_k):
                close_word = word_dict_rev[nearest[k]]
                log_str = "%s %s," % (log_str, close_word)
            print(log_str)
# 高频词可视化
embed_vis = sess.run(embeddings, feed_dict={x_inputs:batch_inputs, y_target:batch_labels})
tsne = TSNE(n_components=2)
embed_vis_fitted = tsne.fit_transform(embed_vis)
# print(embed_vis[0:5])
plt.figure(figsize=(35,35))
for i in range(500):
    x = np.transpose(embed_vis_fitted[i])[0]
    y = np.transpose(embed_vis_fitted[i])[1]
    plt.text(x,y,word_dict_rev[i],fontsize=25,rotation=15)
    plt.scatter(x,y,marker='o')
    if (i+1)%100 == 0:
        plt.savefig('Movie_Review_Fig_'+str(i+1)+'.png')
plt.show()
