import os
import json
import tensorflow as tf
import os, time, pickle
base_path1 =os.path.abspath(os.path.dirname(__file__))
base_path = parent_path = os.path.dirname(base_path1)
class BaseDataIter(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def train_data(self):
        raise NotImplemented

    def test_data(self):
        raise NotImplemented
    

class BaseModelParams(object):
    """
    Base class for model parameters
    Any model that takes parameters should derive this class to provide parameters
    """
    def __init__(self):
        """
        Common parameters
        Derived classes should override these parameters
        """
        # Checkpoint root directory; it may contain multiple directories for
        # different models
        self.checkpoint_dir = None

        # Sample directory
        self.sample_dir = None

        # Log directory
        self.log_dir = None

        # Dataset directory; this is the root directory of all datasets.
        # E.g., if dataset coco is located at /mnt/data/coco, then this
        # value should be /mnt/data
        self.dataset_dir = None

        # Name of the dataset; it should be the same as the directory
        # name containing this dataset.
        # E.g., if dataset coco is located at /mnt/data/coco, then this
        # value should be coco
        self.dataset_name = None

        # Name of this model; it is used as the base name for checkpoint files
        self.model_name = None

        # Name of the directory containing the checkpoint files.
        # This can be the same as the model name; however, it can also be encoded
        # to contain certain details of a particular model.
        # This directory will be a subdirectory under checkpoint directory.
        self.model_dir = None

        # Checkpoint file to load
        self.ckpt_file = None

    def load(self, f):
        """
        Load parameters from specified json file.
        The loaded parameters override those with the same name defined in this subclasses
        :param f:
        :return:
        """
        self.__dict__ = json.load(f)

    def loads(self, s):
        """
        Load parameters from json string
        The loaded parameters override those with the same name defined in this subclasses
        :param s:
        :return:
        """
        self.__dict__ = json.loads(s)

    def update(self):
        """
        Update the params
        :return:
        """
        raise Exception('Not implemented')


class BaseModel(object):
    """
    Base class for models
    """
    def __init__(self, model_params=None):
        """

        """
        self.model_params = model_params
        self.saver = None

    def get_checkpoint_dir(self):
        """
        Get the dir for all checkpoints.
        Implemented by the derived classes.
        :return:
        """
        if self.model_params is not None and self.model_params.checkpoint_dir is not None:
            return self.model_params.checkpoint_dir
        else:
            raise Exception('get_checkpoint_dir must be implemented by derived classes')

    def get_model_dir(self):
        """
        Get the model dir for the checkpoint
        :return:
        """
        if self.model_params is not None and self.model_params.model_dir is not None:
            return self.model_params.model_dir
        else:
            raise Exception('get_model_dir must be implemented by derived classes')

    def get_model_name(self):
        """
        Get the base model name.
        Implemented by the derived classes.
        :return:
        """
        if self.model_params is not None and self.model_params.model_name is not None:
            return self.model_params.model_name
        else:
            raise Exception('get_model_name must be implemented by derived classes')

    def get_sample_dir(self):
        """
        Get the dir for samples.
        Implemented by the derived classes.
        :return:
        """
        if self.model_params is not None and self.model_params.sample_dir is not None:
            return self.model_params.sample_dir
        else:
            raise Exception('get_sample_dir must be implemented by derived classes')

    def get_dataset_dir(self):
        """
        Get the dataset dir.
        Implemented by the derived classes.
        :return:
        """
        if self.model_params is not None and self.model_params.dataset_dir is not None:
            return self.model_params.dataset_dir
        else:
            raise Exception('get_dataset_dir must be implemented by derived classes')

    def check_dirs(self):
        if not os.path.exists(self.get_sample_dir()):
            os.mkdir(self.get_sample_dir())

        # sanity check for dataset
        if not os.path.exists(self.get_dataset_dir()):
            raise Exception('Dataset dir %s does not exist' % self.get_dataset_dir())

    def save(self, step, sess):
        checkpoint_dir = os.path.join(self.get_checkpoint_dir(), self.get_model_dir())

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(sess,
                        os.path.join(checkpoint_dir, self.get_model_name()),
                        global_step=step)

    def load(self, sess):
        """
        Load from a specified directory.
        This is for resuming training from a previous snapshot and is called from train(),
        therefore, a saver is created in train()

        Args:
            sess: tf session
        """
        print(' [*] Reading checkpoints...')

        checkpoint_dir = os.path.join(self.get_checkpoint_dir(), self.get_model_dir())

        ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt_path is not None:
            self.saver.restore(sess, ckpt_path)
            return True
        else:
            return False

    def load_for_testing(self, ckpt_path, sess):
        """
        Load from specified checkpoint file.
        This is for testing the model, a saver will be created here to restore the variables

        Args:
            ckpt_path: path to the checkpoint file
            sess: tf session
        """
        print(' [*] Reading checkpoints...')

        if not os.path.exists(ckpt_path):
            return False

        self.saver = tf.train.Saver()
        self.saver.restore(sess, ckpt_path)
        return True


class DataIter(BaseDataIter):
    def __init__(self, batch_size):
        BaseDataIter.__init__(self, batch_size)
        self.num_train_batch = 0
        self.num_test_batch = 0

        with open(base_path + './data/pascal_feature/train_img_feats.pkl', 'rb') as f:
            self.train_img_feats = pickle.load(f, encoding='iso-8859-1')
        with open(base_path + './data/pascal_feature/train_txt_vecs.pkl', 'rb') as f:
            self.train_txt_vecs = pickle.load(f, encoding='iso-8859-1')
        with open(base_path + './data/pascal_feature/train_labels.pkl', 'rb') as f:
            self.train_labels = pickle.load(f, encoding='iso-8859-1')
        with open(base_path + './data/pascal_feature/train_img_feats.pkl', 'rb') as f:
            self.test_img_feats = pickle.load(f, encoding='iso-8859-1')
        with open(base_path + './data/pascal_feature/train_txt_vecs.pkl', 'rb') as f:
            self.test_txt_vecs = pickle.load(f, encoding='iso-8859-1')
        with open(base_path + './data/pascal_feature/train_labels.pkl', 'rb') as f:
            self.test_labels = pickle.load(f, encoding='iso-8859-1')
        '''perm=np.arange(1000)
        np.random.shuffle(perm)
        self.train_img_feats=np.array(self.train_img_feats)[perm]
        self.train_txt_vecs=np.array(self.train_txt_vecs)[perm]
        self.train_labels=np.array(self.train_labels)[perm]'''

        self.num_train_batch = int((len(self.train_img_feats)) / self.batch_size)
        self.num_test_batch = int(len(self.test_img_feats) / self.batch_size)

    def train_data(self):
        for i in range(self.num_train_batch):
            batch_img_feats = self.train_img_feats[i * self.batch_size: (i + 1) * self.batch_size]
            batch_txt_vecs = self.train_txt_vecs[i * self.batch_size: (i + 1) * self.batch_size]
            batch_labels = self.train_labels[i * self.batch_size: (i + 1) * self.batch_size]
            yield batch_img_feats, batch_txt_vecs, batch_labels, i

    def test_data(self):
        for i in range(self.num_test_batch):
            batch_img_feats = self.test_img_feats[i * self.batch_size: (i + 1) * self.batch_size]
            batch_txt_vecs = self.test_txt_vecs[i * self.batch_size: (i + 1) * self.batch_size]
            batch_labels = self.test_labels[i * self.batch_size: (i + 1) * self.batch_size]
            yield batch_img_feats, batch_txt_vecs, batch_labels, i


class ModelParams(BaseModelParams):
    def __init__(self):
        BaseModelParams.__init__(self)

        self.epoch = 1600
        self.margin = .1
        self.alpha = 5
        self.batch_size = 64
        self.visual_feat_dim = 4096
        self.word_vec_dim = 300
        # self.word_vec_dim = 5000
        self.lr_total = 0.0001
        self.lr_emb = 0.0001
        self.lr_domain = 0.0001
        self.top_k = 50
        self.semantic_emb_dim = 40
        self.dataset_name = 'wikipedia_dataset'
        self.model_name = 'adv_semantic_zsl'
        self.model_dir = 'adv_semantic_zsl_%d_%d_%d' % (self.visual_feat_dim, self.word_vec_dim, self.semantic_emb_dim)

        self.checkpoint_dir = 'checkpoint'
        self.sample_dir = 'samples'
        self.dataset_dir = './data'
        self.log_dir = 'logs'

    def update(self):
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        self.log_dir = os.path.join(self.log_dir, self.model_dir)
        self.dataset_dir = os.path.join(self.dataset_dir, self.dataset_name)