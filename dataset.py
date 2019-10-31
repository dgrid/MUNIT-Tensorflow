import os
import random

import tensorflow as tf

from utils import load_pickle, dump_pickle, get_files, ImageData

class DatasetBuilder:

    def __init__(self, data_folder):
        self.data_folder = data_folder

        # assume 'data' directory exists
        self.dataset_before_split = os.path.join(self.data_folder, 'data', 'all_data.pkl')
        self.dataset_path_trainA = os.path.join(self.data_folder, 'data', 'trainA.pkl')
        self.dataset_path_trainB = os.path.join(self.data_folder, 'data', 'trainB.pkl')
        self.dataset_path_testA = os.path.join(self.data_folder, 'data', 'testA.pkl')
        self.dataset_path_testB = os.path.join(self.data_folder, 'data', 'testB.pkl')

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
            if os.path.exists(self.dataset_before_split):
                all_images = load_pickle(self.dataset_before_split)
            else:
                folder_name = ['cloudy', 'rainy', 'sunny', 'night']
                all_images = dict()
                weather_list = os.listdir(self.data_folder)
                for weather in weather_list:
                    if weather not in folder_name:
                        continue

                    weather_dir = os.path.join(self.data_folder, weather)
                    if os.path.isdir(weather_dir):
                        get_files(weather_dir, all_images)

                dump_pickle(all_images, self.dataset_before_split)

            print('dividing data into part A & part B')
            print('all images : ', type(all_images), len(all_images))
            new_data = []
            for key, value in all_images.items():
                if len(value['instance']) > 0:
                    # temp = value['instance']
                    # value['instance'] = temp
                    new_data.append(value)
            data_num = len(new_data) // 2
            random.shuffle(new_data)
            print('new data', len(new_data))
            trainA = []
            trainB = []
            count = 0
            for i in range(data_num * 2):
                if i < data_num:
                    trainA.append(new_data[i])
                else:
                    trainB.append(new_data[i])

            print('##### data test end ######')
            # split data
            trainA, trainB, testA, testB = train_test_split(trainA, trainB, test_size=0.2, random_state=0)
            dump_pickle(trainA, self.dataset_path_trainA)
            dump_pickle(trainB, self.dataset_path_trainB)
            dump_pickle(testA, self.dataset_path_testA)
            dump_pickle(testB, self.dataset_path_testB)

        print("data ready")
        print()

        self._trainA = trainA
        self._trainB = trainB
        self._dataset_num = max(len(trainA), len(trainB))

    def build_dataset(self):
        return self._trainA, self._trainB

    @property
    def dataset_num(self):
        return self._dataset_num

    def generator(self, dataset, image_data_processor: ImageData):
        for data in dataset:
            # TODO: do parallel
            processed = image_data_processor.processing(data)
            gens = {
                'global': processed['global'],
                'background': processed['background'],
                'instances': tf.stack(processed['instance'])
            }
            yield gens