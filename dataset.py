import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.coco.coco import COCO
from utils.vocabulary import Vocabulary

class DataSet(object):
    def __init__(self,
                 image_ids,
                 image_files,
                 batch_size,
                 word_idxs=None,
                 masks=None,
                 is_train=False,
                 shuffle=False):
        self.image_ids = np.array(image_ids)
        self.image_files = np.array(image_files)
        self.word_idxs = np.array(word_idxs)
        self.masks = np.array(masks)
        self.batch_size = batch_size
        self.is_train = is_train
        self.shuffle = shuffle
        self.setup()

    def setup(self):
        """ Setup the dataset. """
        self.count = len(self.image_ids)
        self.num_batches = int(np.ceil(self.count * 1.0 / self.batch_size))
        self.fake_count = self.num_batches * self.batch_size - self.count
        self.idxs = list(range(self.count))
        self.reset()

    def reset(self):
        """ Reset the dataset. """
        self.current_idx = 0

    def next_batch(self):
        """ Fetch the next batch. """
        assert self.has_next_batch()

        if self.has_full_next_batch():
            start, end = self.current_idx, \
                         self.current_idx + self.batch_size
            current_idxs = self.idxs[start:end]
        else:
            start, end = self.current_idx, self.count
            current_idxs = self.idxs[start:end] + \
                           self.idx[0:self.batch_size - (end-start)]

        image_files = self.image_files[current_idxs]
        if self.is_train:
            word_idxs = self.word_idxs[current_idxs]
            masks = self.masks[current_idxs]
            self.current_idx = (self.current_idx + self.batch_size) % self.count
            return image_files, word_idxs, masks
        else:
            self.current_idx += self.batch_size
            return image_files

    def has_next_batch(self):
        """ Determine whether there is a batch left. """
        return self.current_idx < self.count

    def has_full_next_batch(self):
        """ Determine whether there is a full batch left. """
        return self.current_idx + self.batch_size <= self.count

def prepare_train_data(config):
    """ Prepare the data for training the model. """

    image_ids = np.load('/media/omnisky/683cd494-d120-4dbd-81a4-eb3a90330106/story/mytrain_data/train_image_ids.npy')
    image_files = np.load('/media/omnisky/683cd494-d120-4dbd-81a4-eb3a90330106/story/mytrain_data/train_image_files.npy')
    masks = np.load('/media/omnisky/683cd494-d120-4dbd-81a4-eb3a90330106/story/mytrain_data/train_masks.npy')
    word_idxs = np.load('/media/omnisky/683cd494-d120-4dbd-81a4-eb3a90330106/story/mytrain_data/train_word_idxs_new.npy')
    print("Building the dataset...")
    dataset = DataSet(image_ids,
                      image_files,
                      config.batch_size,
                      word_idxs,
                      masks,
                      True,
                      False)
    print("Dataset built.")
    return dataset
'''
def prepare_eval_data(config):
    """ Prepare the data for evaluating the model. """
    coco = COCO(config.eval_caption_file)
    image_ids = list(coco.imgs.keys())
    image_files = [os.path.join(config.eval_image_dir,
                                coco.imgs[image_id]['file_name'])
                                for image_id in image_ids]

    print("Building the vocabulary...")
    if os.path.exists(config.vocabulary_file):
        vocabulary = Vocabulary(config.vocabulary_size,
                                config.vocabulary_file)
    else:
        vocabulary = build_vocabulary(config)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))

    print("Building the dataset...")
    dataset = DataSet(image_ids, image_files, config.batch_size)
    print("Dataset built.")
    return coco, dataset, vocabulary

def prepare_test_data(config):
    """ Prepare the data for testing the model. """
    files = os.listdir(config.test_image_dir)
    image_files = [os.path.join(config.test_image_dir, f) for f in files
        if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]
    image_ids = list(range(len(image_files)))

    print("Building the vocabulary...")
    if os.path.exists(config.vocabulary_file):
        vocabulary = Vocabulary(config.vocabulary_size,
                                config.vocabulary_file)
    else:
        vocabulary = build_vocabulary(config)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))

    print("Building the dataset...")
    dataset = DataSet(image_ids, image_files, config.batch_size)
    print("Dataset built.")
    return dataset, vocabulary

def build_vocabulary(config):
    """ Build the vocabulary from the training data and save it to a file. """
    coco = COCO(config.train_caption_file)
    coco.filter_by_cap_len(config.max_caption_length)

    vocabulary = Vocabulary(config.vocabulary_size)
    vocabulary.build(coco.all_captions())
    vocabulary.save(config.vocabulary_file)
    return vocabulary
'''