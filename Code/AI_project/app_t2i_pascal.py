import os
import tensorflow as tf
import numpy as np
from configuration import base_model
import pickle
import forward
import gensim.utils
from nltk.corpus import stopwords
from nltk.tag import _pos_tag, PerceptronTagger

_perceptronTagger = PerceptronTagger()
base_path = os.path.abspath(os.path.dirname(__file__))
MODEL_SAVE_PATH = base_path + './model_pascal/'
visual_feat_dim = 4096
word_vec_dim = 300
base_path = os.path.abspath(os.path.dirname(__file__))

# Subtract Stopwords


def tokenize(raw_content, part_tag='NN'):
    tokens = list(gensim.utils.tokenize(raw_content, lowercase=True, deacc=True,
                                        errors='strict', to_lower=True, lower=True))
    standard_stopwords = stopwords.words('english')
    tokens = [word for word in tokens if word.lower() not in standard_stopwords]

    if part_tag is not None:
        tokens = [ww for ww, p in _pos_tag(
            tokens, None, _perceptronTagger) if p == part_tag]
    return tokens

# Extract Words within Texts:


def extract_words_from_xml(text):
    # f=open(filename)
    # text=f.read()
    # convert to unicode and remove additional line breaks
    text = gensim.utils.to_unicode(text)

    text = gensim.utils.decode_htmlentities(text)
    text = text.replace('\n', ' ')

    tokens = tokenize(text)

    #tokens = [word.encode('utf-8') for word in tokens]

    return tokens

# Extract glove text Features


def extract_text(text):

    # Generate word vec
    with open(base_path + './data/glove_mapfile_pascal/word_vecs.pkl',
              'rb') as f:
        word_vecs = pickle.load(f, encoding='iso-8859-1')

    print('Calculating vector representations')
    text_vecs_map = {}
    words = extract_words_from_xml(text)
    vecs = []
    for w in words:
        if w in word_vecs:
            vecs.append(word_vecs[w])
    text_vecs_map = np.average(vecs, 0)
    text_vecs_map = np.array(text_vecs_map)
    text_vecs_map = text_vecs_map.reshape(1, -1)
    return text_vecs_map

# Define reconstruct model


def restore_model(testPicArr):

    with tf.Graph().as_default() as tg:
        tar_txt = tf.placeholder(tf.float32, [None, word_vec_dim])
        Params = base_model.ModelParams()
        model = forward.Forward(Params)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                retrieval_feats = sess.run(model.logits_w, feed_dict={
                                           model.tar_txt: testPicArr})
                return retrieval_feats
            else:
                print("No checkpoint file found")
                return -1

# Database Result Retrieval


def retrieval(query_feature):
    with open(base_path + "./model_pascal/img_feats_retrieval.pkl", 'rb') as f:
        retrieval_img_feats_trans = pickle.load(f, encoding='iso-8859-1')
    with open(base_path + "./model_pascal/txt_vecs_retrieval.pkl", 'rb') as f:
        retrieval_txt_feats_trans = pickle.load(f, encoding='iso-8859-1')
    with open(base_path + './data/pascal_feature/train_labels.pkl', 'rb') as f:
        train_labels = pickle.load(f, encoding='iso-8859-1')
    with open(base_path + './data/pascal_feature/file_label_map.pkl', 'rb') as f:
        file_label_map = pickle.load(f, encoding='iso-8859-1')

    # Images Retrieval
    wv = query_feature
    diffs = retrieval_img_feats_trans - wv
    dists = np.linalg.norm(diffs, axis=1)
    sorted_idx = np.argsort(dists)
    top_k = sorted_idx[0: 4]
    img_path = []
    print("Image retrieval result")
    for i in top_k:
        filename, label = file_label_map[i].split()
        Path = 'static/images/' + label + '/' + filename + '.jpg'
        print(Path, " ", label)
        img_path.append(Path)

    # img_show = []
    # for i in top_k:
    #     filename,label = file_label_map[i].split()
    #     Str = label + '/' + filename +'.jpg'
    #     img_show.append(Str)
    # img1 = cv2.imread(
    #     './data/images_pascal/' + img_show[0])
    # img2 = cv2.imread(
    #     './data/images_pascal/' + img_show[1])
    # img3 = cv2.imread(
    #     './data/images_pascal/' + img_show[2])
    # img4 = cv2.imread(
    #     './data/images_pascal/' + img_show[3])
    # img1 = img1[:, :, (2, 1, 0)]
    # img2 = img2[:, :, (2, 1, 0)]
    # img3 = img3[:, :, (2, 1, 0)]
    # img4 = img4[:, :, (2, 1, 0)]
    # fig = plt.figure()
    # subplot(221)
    # imshow(img1)
    # # title(img_show[0])
    # axis('off')
    # subplot(222)
    # imshow(img2)
    # # title(img_show[1])
    # axis('off')
    # subplot(223)
    # imshow(img3)
    # # title(img_show[2])
    # axis('off')
    # subplot(224)
    # imshow(img4)
    # # title(img_show[3])
    # axis('off')
    # show()

    # Text Retrieval
    wv = query_feature
    diffs = retrieval_txt_feats_trans - wv
    dists = np.linalg.norm(diffs, axis=1)
    sorted_idx = np.argsort(dists)
    top_k = sorted_idx[0: 4]
    text_content = []
    print("Text retrieval result")
    for i in top_k:
        filename, label = file_label_map[i].split()
        Path = base_path + './data/texts_pascal/' + label + "/" + filename + '.txt'
        f = open(Path)
        text = f.read()
        print("retrieval" + " " + str(i))
        print(Path, " ", label)
        print(text)
        text_content.append(text)
    return img_path, text_content


def application(testTxt):
    #testTxt = "./test/1.txt"
    testPicArr = extract_text(testTxt)
    preValue = restore_model(testPicArr)
    return retrieval(preValue)


def main():
    application("Four bikers are riding on a dirt hill.")

if __name__ == '__main__':
    main()
