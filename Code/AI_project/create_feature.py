from __future__ import print_function
from scipy.misc import imread, imresize
import numpy as np
import tensorflow as tf
import vgg16
import gensim
import os
import pickle
import time
import fnmatch
from os.path import basename, splitext
from nltk.tag import _pos_tag, PerceptronTagger
from nltk.corpus import stopwords
from download import download


_perceptronTagger = PerceptronTagger()


def tokenize(raw_content, part_tag='NN'):
    # Subtract stopwords
    tokens = list(gensim.utils.tokenize(raw_content, lowercase=True, deacc=True,
                                        errors='strict', to_lower=True, lower=True))
    standard_stopwords = stopwords.words('english')
    tokens = [word for word in tokens if word.lower() not in standard_stopwords]

    if part_tag is not None:
        tokens = [ww for ww, p in _pos_tag(
            tokens, None, _perceptronTagger) if p == part_tag]
    return tokens


def glob_recursive(pathname, pattern):
    files = []
    for root, dirnames, filenames in os.walk(pathname):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

# Locate glove features


def find_word_vecs(words, data_file):
    vecs = {}
    words = set([w.lower() for w in words])
    with open(data_file, 'r') as f:
        for line in f:
            tokens = line.split()
            if tokens[0] in words:
                vecs[tokens[0]] = np.array([float(num)
                                            for num in tokens[1:]], np.float32)
    return vecs

# Download VGG16 Imagenet weights


def download_vgg16_weights(dirpath):
    url = 'http://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz'
    if not os.path.exists(os.path.join(dirpath, 'vgg16_weights.npz')):
        os.mkdir(dirpath)
        download(url, dirpath)
    else:
        print('Found vgg16 weight, skip')

# Extract VGG16 features


def extract_image_features():
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    batch_size = 100
    filenames = glob_recursive('./data/pascal_sentences/images', '*.jpg')
    num_batches = len(filenames) / batch_size
    img_feats = {}

    with tf.Session() as sess:
        start = time.time()
        vgg = vgg16.vgg16(imgs, './data/vgg16/vgg16_weights.npz', sess)
        print('Loaded vgg16 in %4.4fs' % (time.time() - start))

        for i in range(0, int(num_batches)):
            batch_filenames = filenames[i * batch_size: (i + 1) * batch_size]
            img_data = [imresize(imread(filename, mode='RGB'), (224, 224))
                        for filename in batch_filenames]
            feats = sess.run(vgg.fc2, feed_dict={vgg.imgs: img_data})
            for ii in range(batch_size):
                img_feats[splitext(basename(batch_filenames[ii]))[
                    0]] = feats[ii]
            print('[%d/%d] - finished in %4.4fs' %
                  ((i + 1) * batch_size, len(filenames), time.time() - start))

        batch_filenames = filenames[
            int(num_batches) * batch_size: len(filenames)]
        if len(batch_filenames) > 0:
            img_data = [imresize(imread(filename, mode='RGB'), (224, 224))
                        for filename in batch_filenames]
            feats = sess.run(vgg.fc2, feed_dict={vgg.imgs: img_data})
            for ii in range(len(batch_filenames)):
                img_feats[splitext(basename(batch_filenames[ii]))[
                    0]] = feats[ii]
            print('[%d/%d] - finished in %4.4fs' %
                  (len(filenames), len(filenames), time.time() - start))
    pickle.dump(img_feats, open(
        './data/pascal_sentences/feature/img_feats_vgg16.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    print('Finished')

# Extract glove text features


def extract_text_features():
    print('Extracting pascal sentences text')

    all_txt_files = glob_recursive(
        './data/pascal_sentences/sentences', '*.txt')
    filename_words_map = {}
    all_words = []
    all_labels = []
    filename_label_map = {}

    for it in all_txt_files:
        tokens = it.split('/')
        filename = basename(tokens[-1])
        label = tokens[-2]
        all_labels.append(label)
        filename_label_map[filename] = label
    all_labels = list(set(all_labels))
    with open('./data/pascal_sentences/feature/filename_label_map.pkl', 'wb') as f:
        pickle.dump(filename_label_map, f, pickle.HIGHEST_PROTOCOL)
    with open('./data/pascal_sentences/feature/all_labels.pkl', 'wb') as f:
        pickle.dump(all_labels, f, pickle.HIGHEST_PROTOCOL)
    print('Extracted %d labels' % len(all_labels))

    # Extract unique words and filename to words map
    start = time.time()
    for txt_file in all_txt_files:
        basename_without_ext = splitext(basename(txt_file))[0]
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        tokens = []
        for line in lines:
            tokens += tokenize(line)
        filename_words_map[basename_without_ext] = list(set(tokens))
        all_words += tokens
    unique_words = list(set(all_words))
    with open('./data/pascal_sentences/feature/filename_words_map.pkl', 'wb') as f:
        pickle.dump(filename_words_map, f, pickle.HIGHEST_PROTOCOL)
    with open('./data/pascal_sentences/feature/unique_words.pkl', 'wb') as f:
        pickle.dump(unique_words, f, pickle.HIGHEST_PROTOCOL)
    print('Extracted %d unique words in %4.4fs' %
          (len(unique_words), time.time() - start))

    # Extract word vectors
    print('Extracting word vectors')
    start = time.time()
    word_vecs = find_word_vecs(unique_words, './data/glove.42B.300d.txt')
    pickle.dump(word_vecs, open(
        './data/pascal_sentences/feature/word_vecs.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    print('Extracted %d word vectors in %4.4fs' %
          (len(word_vecs.keys()), time.time() - start))

    # Calc word vecs for images
    print('Calculating vector representations')
    start = time.time()
    text_vecs_map = {}
    for k in filename_words_map.keys():
        words = filename_words_map[k]
        vecs = []
        for w in words:
            if w in word_vecs:
                vecs.append(word_vecs[w])
        text_vecs_map[k] = np.average(vecs, 0)
    pickle.dump(text_vecs_map, open(
        './data/pascal_sentences/feature/filename_vecs_map.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    print('Calculated vector representations in %4.4fs' %
          (time.time() - start))


def main():
    # download_vgg16_weights('./data/vgg16')
    extract_image_features()
    extract_text_features()

if __name__ == '__main__':
    main()
