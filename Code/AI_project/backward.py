from __future__ import print_function
import time
import pickle
import numpy as np
import tensorflow as tf
import sklearn.preprocessing
from configuration.base_model import ModelParams
from forward import Forward
from sklearn.metrics.pairwise import cosine_similarity
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Main training function:


def train(sess, model):
    # Define Loss Function and optimizers
    total_loss = model.emb_loss + model.domain_class_loss
    total_train_op = tf.train.AdamOptimizer(
        learning_rate=model.model_params.lr_total,
        beta1=0.5).minimize(total_loss)
    emb_train_op = tf.train.AdamOptimizer(
        learning_rate=model.model_params.lr_emb,
        beta1=0.5).minimize(model.emb_loss, var_list=model.le_vars + model.vf_vars)
    domain_train_op = tf.train.AdamOptimizer(
        learning_rate=model.model_params.lr_domain,
        beta1=0.5).minimize(model.domain_class_loss, var_list=model.dc_vars)

    # Initialization
    tf.initialize_all_variables().run()
    model.saver = tf.train.Saver()

    start_time = time.time()
    map_avg_ti = []
    map_avg_it = []
    adv_loss = []
    emb_loss = []
    for epoch in range(model.model_params.epoch):

        p = float(epoch) / model.model_params.epoch
        l = 2. / (1. + np.exp(-10. * p)) - 1
        for batch_feat, batch_vec, batch_labels, idx in model.data_iter.train_data():
            # Generate Labels
            batch_labels_ = batch_labels - np.ones_like(batch_labels)
            label_binarizer = sklearn.preprocessing.LabelBinarizer()
            label_binarizer.fit(range(max(batch_labels_) + 1))
            b = label_binarizer.transform(batch_labels_)
            # Construct Triplets within each batch
            adj_mat = np.dot(b, np.transpose(b))
            mask_mat = np.ones_like(adj_mat) - adj_mat
            img_sim_mat = mask_mat * cosine_similarity(batch_feat, batch_feat)
            txt_sim_mat = mask_mat * cosine_similarity(batch_vec, batch_vec)
            img_neg_txt_idx = np.argmax(img_sim_mat, axis=1).astype(int)
            txt_neg_img_idx = np.argmax(txt_sim_mat, axis=1).astype(int)
            # print('{0}'.format(img_neg_txt_idx.shape)
            batch_vec_ = np.array(batch_vec)
            batch_feat_ = np.array(batch_feat)
            img_neg_txt = batch_vec_[img_neg_txt_idx, :]
            txt_neg_img = batch_feat_[txt_neg_img_idx, :]
            # _, label_loss_val, dissimilar_loss_val, similar_loss_val =
            # sess.run([total_train_op, model.label_loss,
            # model.dissimilar_loss, model.similar_loss],
            # feed_dict={model.tar_img: batch_feat, model.tar_txt: batch_vec,
            # model.y: b, model.y_single: np.transpose([batch_labels]),model.l:
            # l})
            sess.run([emb_train_op, domain_train_op],
                     feed_dict={model.tar_img: batch_feat,
                                model.tar_txt: batch_vec,
                                model.pos_txt: batch_vec,
                                model.neg_txt: img_neg_txt,
                                model.pos_img: batch_feat,
                                model.neg_img: txt_neg_img,
                                model.y: b,
                                model.y_single: np.transpose([batch_labels]),
                                model.l: l})
            label_loss_val, triplet_loss_val, emb_loss_val, domain_loss_val, v_loss_pos, v_loss_neg, w_loss_pos, w_loss_neg = sess.run(
                [model.label_loss, model.triplet_loss, model.emb_loss, model.domain_class_loss, model.v_loss_pos,
                 model.v_loss_neg, model.w_loss_pos, model.w_loss_neg],
                feed_dict={model.tar_img: batch_feat,
                           model.tar_txt: batch_vec,
                           model.pos_txt: batch_vec,
                           model.neg_txt: img_neg_txt,
                           model.pos_img: batch_feat,
                           model.neg_img: txt_neg_img,
                           model.y: b,
                           model.y_single: np.transpose([batch_labels]),
                           model.l: l})
            print(
                'Epoch: [%2d][%4d/%4d] time: %4.4f, emb_loss: %.8f, domain_loss: %.8f, label_loss: %.8f, triplet_loss: %.8f , %.8f ,%.8f , %.8f ,%.8f' % (
                    epoch, idx, model.data_iter.num_train_batch, time.time(
                    ) - start_time, emb_loss_val, domain_loss_val,
                    label_loss_val, triplet_loss_val, v_loss_pos, v_loss_neg, w_loss_pos, w_loss_neg
                ))
        if epoch % 10 == 0:
            start = time.time()

            test_img_feats_trans = []
            test_txt_vecs_trans = []
            test_labels = []
            for feats, vecs, labels, i in model.data_iter.test_data():
                feats_trans = sess.run(model.logits_v, feed_dict={
                                       model.tar_img: feats})
                vecs_trans = sess.run(model.logits_w, feed_dict={
                                      model.tar_txt: vecs})
                test_labels += labels
                for ii in range(len(feats)):
                    test_img_feats_trans.append(feats_trans[ii])
                    test_txt_vecs_trans.append(vecs_trans[ii])
            test_img_feats_trans = np.asarray(test_img_feats_trans)
            test_txt_vecs_trans = np.asarray(test_txt_vecs_trans)

            top_k = [50]
            avg_precs = []
            all_precs = []
            for k in top_k:
                for i in range(len(test_txt_vecs_trans)):
                    query_label = test_labels[i]

                    # Sorted distances
                    wv = test_txt_vecs_trans[i]
                    diffs = test_img_feats_trans - wv
                    dists = np.linalg.norm(diffs, axis=1)
                    sorted_idx = np.argsort(dists)
                    precs = []
                    for topk in range(1, k + 1):
                        hits = 0
                        top_k = sorted_idx[0: topk]
                        if np.sum(query_label) != test_labels[top_k[-1]]:
                            continue
                        for ii in top_k:
                            retrieved_label = test_labels[ii]
                            if np.sum(retrieved_label) == query_label:
                                hits += 1
                        precs.append(float(hits) / float(topk))
                    if len(precs) == 0:
                        precs.append(0)
                    avg_precs.append(np.average(precs))
                mean_avg_prec = np.mean(avg_precs)
                all_precs.append(mean_avg_prec)
            print('[Eval - txt2img] mAP: %f in %4.4fs' %
                  (all_precs[0], (time.time() - start)))
            t2i = all_precs[0]

            avg_precs = []
            all_precs = []
            top_k = [50]

            for k in top_k:
                for i in range(len(test_img_feats_trans)):
                    query_img_feat = test_img_feats_trans[i]
                    ground_truth_label = test_labels[i]

                    # Sorted Distance
                    diffs = test_txt_vecs_trans - query_img_feat
                    dists = np.linalg.norm(diffs, axis=1)
                    sorted_idx = np.argsort(dists)
                    precs = []
                    for topk in range(1, k + 1):
                        hits = 0
                        top_k = sorted_idx[0: topk]
                        if np.sum(ground_truth_label) != test_labels[top_k[-1]]:
                            continue
                        for ii in top_k:
                            retrieved_label = test_labels[ii]
                            if np.sum(ground_truth_label) == retrieved_label:
                                hits += 1
                        precs.append(float(hits) / float(topk))
                    if len(precs) == 0:
                        precs.append(0)
                    avg_precs.append(np.average(precs))
                mean_avg_prec = np.mean(avg_precs)
                all_precs.append(mean_avg_prec)
            print('[Eval - img2txt] mAP: %f in %4.4fs' %
                  (all_precs[0], (time.time() - start)))
            if epoch == 200:
                model.saver.save(
                    sess, './model_pascal/model.ckpt', global_step=epoch)


def main(_):
    graph = tf.Graph()
    model_params = ModelParams()
    model_params.update()
    with graph.as_default():
        model = Forward(model_params)
    with tf.Session(graph=graph, config=config) as sess:
        train(sess, model)

if __name__ == '__main__':
    tf.app.run()
