import tensorflow as tf

import os
from configuration import base_model
import numpy as np
import pickle
from scipy.misc import imread, imresize
import forward
from tensorflow.keras.applications.vgg16 import vgg16
import matplotlib.pyplot as plt

base_path = os.path.abspath(os.path.dirname(__file__))

MODEL_SAVE_PATH = base_path + './model_pascal/'
visual_feat_dim = 4096
word_vec_dim = 300
base_path = os.path.abspath(os.path.dirname(__file__))

# Define: A reconsctruct model function.


def restore_model(testPicArr):

    # Generate initial tensorflow computing graph.
    with tf.Graph().as_default() as tg:
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        tar_img = tf.placeholder(tf.float32, [None, visual_feat_dim])
        tar_txt = tf.placeholder(tf.float32, [None, word_vec_dim])
        Params = base_model.ModelParams()
        model = forward.Forward(Params)
        saver = tf.train.Saver()

        with tf.Session() as sess:

            # Load "ckpt" model and locate from checkpoint
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            vgg = vgg16.vgg16(imgs, base_path +
            #                  './vgg16/vgg16_weights.npz', sess)
            # Transfer Learning from VGG16.
            feats=sess.run(vgg.fc2, feed_dict={vgg.imgs: testPicArr})
            # Check if "ckpt" existed or not
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                retrieval_feats=sess.run(model.logits_v, feed_dict={
                                           model.tar_img: feats})
                return retrieval_feats
            else:
                print("No checkpoint file found")
                return -1

 # Database images and texts Retrieval


def retrieval(query_feature):
    with open(base_path + "./model_pascal/img_feats_retrieval.pkl", 'rb') as f:
        retrieval_img_feats_trans=pickle.load(f, encoding='iso-8859-1')
    with open(base_path + "./model_pascal/txt_vecs_retrieval.pkl", 'rb') as f:
        retrieval_txt_feats_trans=pickle.load(f, encoding='iso-8859-1')
    with open(base_path + './data/pascal_feature/train_labels.pkl', 'rb') as f:
        train_labels=pickle.load(f, encoding='iso-8859-1')
    with open(base_path + './data/pascal_feature/file_label_map.pkl', 'rb') as f:
        file_label_map=pickle.load(f, encoding='iso-8859-1')

    # Matched images Retrieval
    wv=query_feature
    diffs=retrieval_img_feats_trans - wv
    dists=np.linalg.norm(diffs, axis=1)
    sorted_idx=np.argsort(dists)
    top_k=sorted_idx[0: 4]
    img_path=[]
    print("Image retrieval result")
    for i in top_k:
        filename, label=file_label_map[i].split()
        Path='static/images/' + label + '/' + filename + '.jpg'
        print(Path, " ", label)
        img_path.append(Path)

    # img_show = []
    # for i in top_k:
    #     filename,label = file_label_map[i].split()
    #     Str = label + '/' + filename +'.jpg'
    #     img_show.append(Str)
    # img1 = cv2.imread('./data/images_pascal/' + img_show[0])
    # img2 = cv2.imread('./data/images_pascal/' + img_show[1])
    # img3 = cv2.imread('./data/images_pascal/' + img_show[2])
    # img4 = cv2.imread('./data/images_pascal/' + img_show[3])
    # img1 = img1[:, :, (2, 1, 0)]
    # img2 = img2[:, :, (2, 1, 0)]
    # img3 = img3[:, :, (2, 1, 0)]
    # img4 = img4[:, :, (2, 1, 0)]
    # fig = plt.figure()
    # subplot(221)
    # imshow(img1)
    # #title(img_show[0])
    # axis('off')
    # subplot(222)
    # imshow(img2)
    # #title(img_show[1])
    # axis('off')
    # subplot(223)
    # imshow(img3)
    # #title(img_show[2])
    # axis('off')
    # subplot(224)
    # imshow(img4)
    # #title(img_show[3])
    # axis('off')
    # show()

    # Matched texts Retrieval
    wv=query_feature
    diffs=retrieval_txt_feats_trans - wv
    dists=np.linalg.norm(diffs, axis=1)
    sorted_idx=np.argsort(dists)
    top_k=sorted_idx[0: 4]
    text_content=[]
    print("Text retrieval result")
    for i in top_k:
        filename, label=file_label_map[i].split()
        Path=base_path + './data/texts_pascal/' + label + "/" + filename + '.txt'
        f=open(Path)
        text=f.read()
        print("retrieval" + " " + str(i))
        print(Path, " ", label)
        print(text)
        text_content.append(text)
    return img_path, text_content


def pre_pic(picName):
    img_data=imresize(imread(picName, mode='RGB'), (224, 224))
    img_data=img_data.reshape([1, 224, 224, 3])
    return img_data


def application(testPic):
    # testPic = "./test/34.jpg"
    testPicArr=pre_pic(testPic)
    preValue=restore_model(testPicArr)
    img0=plt.imread(testPic)
    plt.imshow(img0)
    return retrieval(preValue)


def main():
    application("./test/34.jpg")

if __name__ == '__main__':
    main()
