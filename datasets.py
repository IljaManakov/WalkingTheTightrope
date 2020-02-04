import os
from os.path import join
from copy import copy
import random

import torch as pt
from torch.utils.data.dataset import Dataset, ConcatDataset, Subset
from torchvision.datasets import STL10
from skimage.io import imread
import pandas as pd
import numpy as np
from cv2 import resize, INTER_CUBIC


class CelebA(Dataset):

    def __init__(self, folder, fraction=0.8, scale=1):

        self.folder = folder

        self.image_files = [f for f in os.listdir(folder) if '.jpg' in f]
        self.image_files.sort()
        self.landmarks = pd.read_csv(join(folder, 'list_landmarks_align_celeba.csv'))
        self.attributes = pd.read_csv(join(folder, 'list_attr_celeba.csv'))
        last_index = int(len(self.image_files)*abs(fraction))
        self.image_files = self.image_files[:last_index] if fraction > 0 else self.image_files[-last_index:]
        self.attributes = self.attributes.iloc[:last_index] if fraction > 0 else self.attributes.iloc[-last_index:]
        self.landmarks = self.landmarks.iloc[:last_index] if fraction > 0 else self.landmarks.iloc[-last_index:]
        self.scale = scale
        self.images = []

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, item):

        image = self.preprocess(self.image_files[item])
        attributes = pt.from_numpy(self.attributes.iloc[item].to_numpy()[1:].astype(int).clip(0, 1))
        landmarks = pt.from_numpy(self.landmarks.iloc[item].to_numpy()[1:].astype(int)).float()
        landmarks[::2] += 7
        landmarks[1::2] -= 13

        return image, attributes, landmarks*self.scale

    def preprocess(self, file):
        image = imread(join(self.folder, file))
        image = np.pad(image, ((0, 0), (7, 7), (0, 0)), mode='reflect')[13:-13, ...]
        if self.scale != 1:
            new_size = int(self.scale * image.shape[0])
            image = resize(image, (new_size, new_size), interpolation=INTER_CUBIC)
        image = pt.from_numpy(image).permute(2, 0, 1).float() / 255
        return image


class STL10(STL10):

    def __init__(self, folder, fraction=0.8, split='test', part=None):

        super().__init__(folder, split)

        self.folder = folder

        last_index = int(len(self)*abs(fraction))
        part = part or slice(last_index) if fraction > 0 else slice(-last_index, -1)
        self.data = self.data[part]
        self.labels = self.labels[part]
        self.data = pt.from_numpy(self.data).float()/255

    def __len__(self):

        return self.data.shape[0]

    def __getitem__(self, item):

        image = self.data[item]
        label = self.labels[item]

        return image, label


class Pokemon(Dataset):

    def __init__(self, folder, fraction=0.8, scale=1):

        self.folder = folder
        self.image_files = [f for f in os.listdir(folder) if '.png' in f]
        self.image_files.sort(key=lambda x: (int(x[:3]), x[3:]))
        last_index = int(len(self.image_files)*abs(fraction))
        self.image_files = self.image_files[:last_index] if fraction > 0 else self.image_files[-last_index:]
        self.images = []
        for file in self.image_files:
            image = imread(join(folder, file))
            if scale != 1:
                new_size = int(scale*image.shape[0])
                image = resize(image, (new_size, new_size))
            image = pt.from_numpy(image).permute(2, 0, 1)[:3, ...].float() / 255
            self.images.append(image)

        self.data = pd.read_csv(join(folder, 'pokemon.csv'))
        self.types = {t: i for i, t in enumerate(self.data['Type 1'].unique())}

    def __len__(self):

        return len(self.images)

    def __getitem__(self, item):

        image = self.images[item]
        target = pt.zeros(len(self.types))
        type1 = self.types[self.data.iloc[item]['Type 1']]
        target[type1] = 1
        type2 = self.data.iloc[item]['Type 2']
        if isinstance(type2, str):
            type2 = self.types[type2]
            target[type2] = 1

        return image, target


class Representations(Dataset):

    def __init__(self, file, fraction=0.8, transformation=lambda x: x, shuffle=1):

        self.samples = pt.load(file)
        last_index = int(len(self.samples)*abs(fraction))
        self.samples = self.samples[:last_index] if fraction > 0 else self.samples[-last_index:]
        self.transformation = transformation
        if shuffle:
            random.Random(shuffle).shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.transformation(self.samples[item])


class Folds(object):
    """
    utility class that splits a given data set into folds and is iterable.
    When iterating, the class returns a tuple containing a training and test set.
    The test set is one of the folds, while the training set is the concatenation of the remaining folds.
    """

    def __init__(self, *, n_folds: int, dataset: Dataset):
        """
        initializes the folds
        :param n_folds: number of folds
        :param dataset: pytorch dataset object
        """
        super(Folds, self).__init__()
        self.dataset = dataset
        self.folds = [Subset(dataset, inds) for inds in np.array_split(range(len(dataset)), n_folds)]
        self.current_fold = 0

    def __len__(self):
        return len(self.folds)

    def __iter__(self):
        """
        resets the current fold before iteration
        :return: self
        """
        self.current_fold = 0
        return self

    def __next__(self):
        """
        provides the next pair of training and test set
        :return: (training set, test set)
        """
        if self.current_fold == len(self):
            raise StopIteration

        folds = copy(self.folds)
        test_set = folds.pop(self.current_fold)
        training_set = ConcatDataset(folds)

        self.current_fold += 1
        return training_set, test_set