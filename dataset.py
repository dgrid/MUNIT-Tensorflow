import os
import random
from typing import Tuple
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import batch_and_drop_remainder
from tensorflow.contrib.data import prefetch_to_device
from sklearn.model_selection import train_test_split

from utils import load_pickle, dump_pickle, get_files, ImageData

class DatasetBuilder:

    def __init__(self, batch_size, data_folder='/home/user/share/data', image_data_processor=None):
        self.batch_size = batch_size
        self.data_folder = data_folder
        self.image_data_processor = image_data_processor

        # assume 'data' directory exists
        self.dataset_path_trainA = os.path.join(self.data_folder, 'data', 'trainA.pkl')
        self.dataset_path_trainB = os.path.join(self.data_folder, 'data', 'trainB.pkl')
        self.dataset_path_testA = os.path.join(self.data_folder, 'data', 'testA.pkl')
        self.dataset_path_testB = os.path.join(self.data_folder, 'data', 'testB.pkl')

        # construct self._trainA and self._trainB
        self._build_dataset()

    def _build_dataset(self):
        """
        return dataset, if precomputed pkl is not found, dump it.
        """
        print("_"*20)
        print()
        print("start to process data")
        print('##### test info #####')
        if os.path.exists(self.dataset_path_trainA) and os.path.exists(self.dataset_path_trainB) and os.path.exists(self.dataset_path_testA) and os.path.exists(self.dataset_path_testB):
            trainA = load_pickle(self.dataset_path_trainA)
            trainB = load_pickle(self.dataset_path_trainB)
        else:
            # select domain_A and domain_B
            folder_name = ['cloudy', 'rainy', 'sunny', 'night']
            weather_list = []
            for weather in os.listdir(self.data_folder):
                if weather in folder_name:
                    weather_list.append(weather)
            if len(weather_list) != 2:
                raise ValueError(f"prepare two domains in {self.data_folder}: {weather_list}")
            weather_A, weather_B = weather_list

            # load images from weather_a and weather_B
            images_A = dict()
            get_files(os.path.join(self.data_folder, weather_A), images_A)
            images_B = dict()
            get_files(os.path.join(self.data_folder, weather_B), images_B)

            # remove no-instance images
            trainA = []
            for key, value in images_A.items():
                if len(value['instance']) > 0:
                    trainA.append(value)
            trainB = []
            for key, value in images_B.items():
                if len(value['instance']) > 0:
                    trainB.append(value)

            # shuffle datum
            random.shuffle(trainA)
            random.shuffle(trainB)
            print(f"domain A({weather_A}): ", len(trainA))
            print(f"domain B({weather_B}): ", len(trainB))

            print('##### data test end ######')

            # split data
            trainA, trainB, testA, testB = train_test_split(trainA, trainB, test_size=0.2, random_state=0)
            os.makedirs(os.path.dirname(self.dataset_path_trainA), exist_ok=True)
            dump_pickle(trainA, self.dataset_path_trainA)
            dump_pickle(trainB, self.dataset_path_trainB)
            dump_pickle(testA, self.dataset_path_testA)
            dump_pickle(testB, self.dataset_path_testB)

        print("data ready")
        print()

        self._trainA = trainA
        self._trainB = trainB
        self._dataset_num = max(len(trainA), len(trainB))

    def build_dataset(self, gpu_device):
        generator_A = partial(self.generator, dataset=self._trainA)
        generator_B = partial(self.generator, dataset=self._trainB)

        # Dataset for Dataset Iterator
        output_types = {
            'global': tf.float32,
            'instance': tf.float32,
            # 'background': tf.float32,
        }
        # output shapes without batch
        output_shapes = {
            'global': self.image_data_processor.get_image_shape(),
            # 'background': self.image_data_processor.get_image_shape(),
            'instance': self.image_data_processor.get_object_shape()
        }
        dataset_A = tf.data.Dataset.from_generator(generator_A, output_types, output_shapes)
        dataset_B = tf.data.Dataset.from_generator(generator_B, output_types, output_shapes)

        dataset_A = dataset_A.prefetch(self.batch_size) \
                    .shuffle(self.dataset_num) \
                    .apply(batch_and_drop_remainder(self.batch_size)) \
                    .repeat() \
                    .apply(prefetch_to_device(gpu_device, None))
        dataset_B = dataset_B.prefetch(self.batch_size) \
                    .shuffle(self.dataset_num) \
                    .apply(batch_and_drop_remainder(self.batch_size)) \
                    .repeat() \
                    .apply(prefetch_to_device(gpu_device, None))

        # Iterator
        trainA_iterator = dataset_A.make_one_shot_iterator()
        trainB_iterator = dataset_B.make_one_shot_iterator()

        return trainA_iterator, trainB_iterator

    @property
    def dataset_num(self):
        return self._dataset_num

    def generator(self, dataset):
        for data in dataset:
            # TODO: do parallel
            processed = self.image_data_processor.processing(data)

            gens = {
                'global': processed['global'],
                # 'background': processed['background'],
                'instance': processed['instance']
            }
            yield gens


class DebugDatasetBuilder:

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_dataset(self, gpu_device) -> Tuple[tf.data.Iterator, tf.data.Iterator]:
        d_A = np.random.normal(loc=0.0, scale=1.0, size=[360, 360, 3])
        d_B = np.random.normal(loc=0.0, scale=1.0, size=[360, 360, 3])

        d_a = np.random.normal(loc=0.0, scale=1.0, size=[120, 120, 3])
        d_b = np.random.normal(loc=0.0, scale=1.0, size=[120, 120, 3])

        d_abg = np.random.normal(loc=0.0, scale=1.0, size=[360, 360, 3])
        d_bbg = np.random.normal(loc=0.0, scale=1.0, size=[360, 360, 3])

        # generators for Dataest
        domain_A_all = {
            'global': d_A,
            'instance': d_a,
            # 'background': tf.cast(d_abg, tf.float32),
        }
        domain_B_all = {
            'global': d_B,
            'instance': d_b,
            # 'background': tf.cast(d_bbg, tf.float32),
        }
        generator_A = partial(self.generator, domain=domain_A_all)
        generator_B = partial(self.generator, domain=domain_B_all)

        # Dataset for Dataset Iterator
        output_types = {
            'global': tf.float32,
            'instance': tf.float32,
            # 'background': tf.float32,
        }
        output_shapes = {
            'global': tf.TensorShape((360, 360, 3)),
            # 'background': tf.TensorShape((self.batch_size, 120, 120, 3)),
            'instance': tf.TensorShape((120, 120, 3)),
        }
        dataset_A = tf.data.Dataset.from_generator(generator_A, output_types, output_shapes)
        dataset_B = tf.data.Dataset.from_generator(generator_B, output_types, output_shapes)

        dataset_A = dataset_A.prefetch(self.batch_size) \
                    .apply(batch_and_drop_remainder(self.batch_size)) \
                    .repeat() \
                    .apply(prefetch_to_device(gpu_device, None))
        dataset_B = dataset_B.prefetch(self.batch_size) \
                    .apply(batch_and_drop_remainder(self.batch_size)) \
                    .repeat() \
                    .apply(prefetch_to_device(gpu_device, None))

        # Iterator
        trainA_iterator = dataset_A.make_one_shot_iterator()
        trainB_iterator = dataset_B.make_one_shot_iterator()

        return trainA_iterator, trainB_iterator

    @property
    def dataset_num(self) -> int:
        return self.batch_size

    def generator(self, domain):
        yield domain
