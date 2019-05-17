from __future__ import print_function
import os
import time
import pickle
import tensorflow as tf
import numpy as np
import time
import pickle
import numpy as np
import tensorflow as tf
import sklearn.preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow.contrib.slim as slim
from configuration.base_model import BaseModel, DataIter
from configuration.flip_gradient import flip_gradient
base_path1 = os.path.abspath(os.path.dirname(__file__))
base_path = parent_path = os.path.dirname(base_path1)

# Define Forward Class


class Forward(BaseModel):

    def __init__(self, model_params):
        BaseModel.__init__(self, model_params)
        self.data_iter = DataIter(self.model_params.batch_size)

        self.tar_img = tf.placeholder(
            tf.float32, [None, self.model_params.visual_feat_dim])
        self.tar_txt = tf.placeholder(
            tf.float32, [None, self.model_params.word_vec_dim])
        self.pos_img = tf.placeholder(
            tf.float32, [None, self.model_params.visual_feat_dim])
        self.neg_img = tf.placeholder(
            tf.float32, [None, self.model_params.visual_feat_dim])
        self.pos_txt = tf.placeholder(
            tf.float32, [None, self.model_params.word_vec_dim])
        self.neg_txt = tf.placeholder(
            tf.float32, [None, self.model_params.word_vec_dim])
        self.y = tf.placeholder(tf.int32, [self.model_params.batch_size, 20])
        self.y_single = tf.placeholder(
            tf.int32, [self.model_params.batch_size, 1])
        self.l = tf.placeholder(tf.float32, [])
        self.emb_v = self.visual_feature_embed(self.tar_img)
        self.emb_w = self.label_embed(self.tar_txt)
        self.emb_v_pos = self.visual_feature_embed(self.pos_img, reuse=True)
        self.emb_v_neg = self.visual_feature_embed(self.neg_img, reuse=True)
        self.emb_w_pos = self.label_embed(self.pos_txt, reuse=True)
        self.emb_w_neg = self.label_embed(self.neg_txt, reuse=True)

        # Calculate Triplet Loss
        margin = self.model_params.margin
        alpha = self.model_params.alpha
        self.v_loss_pos = tf.reduce_sum(
            tf.nn.l2_loss(self.emb_v - self.emb_w_pos))
        self.v_loss_neg = tf.reduce_sum(
            tf.nn.l2_loss(self.emb_v - self.emb_w_neg))
        self.w_loss_pos = tf.reduce_sum(
            tf.nn.l2_loss(self.emb_w - self.emb_v_pos))
        self.w_loss_neg = tf.reduce_sum(
            tf.nn.l2_loss(self.emb_w - self.emb_v_neg))
        self.triplet_loss = tf.maximum(0., margin + alpha * self.v_loss_pos - self.v_loss_neg) + \
            tf.maximum(0., margin + alpha * self.w_loss_pos - self.w_loss_neg)

        # Calculate Label Loss
        self.logits_v = self.label_classifier(self.emb_v)
        self.logits_w = self.label_classifier(self.emb_w, reuse=True)
        self.label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits_v) + \
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y, logits=self.logits_w)
        self.label_loss = tf.reduce_mean(self.label_loss)

        # Calculate Emb Loss
        self.emb_loss = 100 * self.label_loss + self.triplet_loss

        # Define a Classifier
        self.emb_v_class = self.domain_classifier(self.emb_v, self.l)
        self.emb_w_class = self.domain_classifier(
            self.emb_w, self.l, reuse=True)

        # Consctruct binary label
        all_emb_v = tf.concat([tf.ones([self.model_params.batch_size, 1]),
                               tf.zeros([self.model_params.batch_size, 1])], 1)
        all_emb_w = tf.concat([tf.zeros([self.model_params.batch_size, 1]),
                               tf.ones([self.model_params.batch_size, 1])], 1)

        # Domain-Adversarial Loss
        self.domain_class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_v_class, labels=all_emb_w) + \
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.emb_w_class, labels=all_emb_v)
        self.domain_class_loss = tf.reduce_mean(self.domain_class_loss)

        # Network Parameters
        self.t_vars = tf.trainable_variables()
        self.vf_vars = [v for v in self.t_vars if 'vf_' in v.name]
        self.le_vars = [v for v in self.t_vars if 'le_' in v.name]
        self.dc_vars = [v for v in self.t_vars if 'dc_' in v.name]
        self.lc_vars = [v for v in self.t_vars if 'lc_' in v.name]

    # Define Image Fully-connected Function
    def visual_feature_embed(self, X, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.tanh(slim.fully_connected(X, 512, scope='vf_fc_0'))
            net = tf.nn.tanh(slim.fully_connected(net, 100, scope='vf_fc_1'))
            net = tf.nn.tanh(slim.fully_connected(
                net, self.model_params.semantic_emb_dim, scope='vf_fc_2'))
        return net

    # Define Textual Fully-connected Function
    def label_embed(self, L, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.tanh(slim.fully_connected(
                L, self.model_params.semantic_emb_dim, scope='le_fc_0'))
            net = tf.nn.tanh(slim.fully_connected(net, 100, scope='le_fc_1'))
            net = tf.nn.tanh(slim.fully_connected(
                net, self.model_params.semantic_emb_dim, scope='le_fc_2'))
        return net

    # Define Label Classifier
    def label_classifier(self, X, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = slim.fully_connected(X, 20, scope='lc_fc_0')
        return net

    # Define Domain Classifier
    def domain_classifier(self, E, l, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            E = flip_gradient(E, l)
            net = slim.fully_connected(
                E, int(self.model_params.semantic_emb_dim / 2), scope='dc_fc_0')
            net = slim.fully_connected(
                net, int(self.model_params.semantic_emb_dim / 4), scope='dc_fc_1')
            net = slim.fully_connected(net, 2, scope='dc_fc_2')
        return net
