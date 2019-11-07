import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc
import os, random
import numpy as np
from PIL import Image
import pickle

# https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
# https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/

class ImageData:

    def __init__(self, img_h, img_w, channels, augment_flag=False, obj_h=120, obj_w=120):
        print('utils.imagedata.init')
        self.img_h = img_h
        self.img_w = img_w
        self.channels = channels
        self.augment_flag = augment_flag

        # objects
        self.obj_h = obj_h
        self.obj_w = obj_w
        print('utils.imagedata.init.end')

    def get_image_shape(self):
        # print('utils.get_image_shape')
        # channels-last
        return [self.img_h, self.img_w, self.channels]

    def get_object_shape(self):
        # channels-last
        return [self.obj_h, self.obj_w, self.channels]

    def image_processing(self, filename):
        print('utils.imagedata.iamge_processing')
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.img_h, self.img_w])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag :
            augment_size_h = self.img_h + (30 if self.img_h == 256 else 15)
            augment_size_w = self.img_w + (30 if self.img_w == 256 else 15)
            p = random.random()
            if p > 0.5:
                img = augmentation(img, augment_size_h, augment_size_w)

        print('utils.imagedata.iamge_processing.end')
        return img

    def check_size(self, filename):
        # if the instance is larger than 60 * 60 pixels
        print('utils.iamgedata.check size')
        image = Image.open(filename)
        print(image.size)
        print(image)
        if image.height >= 60 and image.width >= 60:
            print('utils.iamgedata.check size.end')
            return True
        else:
            print('utils.iamgedata.check size.end')
            return False     

    def object_resize(self, filename, height=120, width=120):
        print('utils.imagedatset.object_resize')
        # object will be resized to 120*120 pixels                                                                                                                      
        x = tf.read_file(filename)
        x_decode = tf.image.decode_png(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [height, width])
        img = tf.cast(img, tf.float32) / 127.5 - 1
        print('utils.imagedatset.object_resize.end')    
        return img



    def image_resize(self, filename, resize=True, height=360, width=360):
        print('utils.imagedatset.image_resize')
        # the short side of image will be resized to 360 pixels
        # due to the limitation of GPU memory
        x = tf.read_file(filename)
        img = tf.image.decode_png(x, channels=self.channels)
        if resize:
            img = tf.image.resize_images(img, [height, width])
        img = tf.cast(img, tf.float32) / 127.5 - 1
        print('utils.imagedatset.image_resize.end')
        return img

    def processing(self, file):
        print('utils.imagedataset.processing')
        global_image = self.image_resize(file['global'])
        instance = []
        for file_path in file['instance']:
            if self.check_size(file_path):
                instance.append(self.object_resize(file_path))
        
        # # background = self.image_resize(file['background'])
        print()
        print('*' * 50)
        print('global shape   : ', global_image.shape.as_list())
        print('instance len   : ', len(instance))
        # print(type(random_instance))
        # print(random_instance.shape)
        # print('instance shape : ', random_instance.shape.as_list())
        if len(instance) > 0:
            random_instance = random.sample(instance, 1)[0]
            print('utils.imagedataset.processing.end')
            return {'global': global_image,
                # 'background': background,
                'instance': random_instance}
        print('utils.imagedataset.processing.end')


def load_test_data(image_path, size_h=256, size_w=256):
    print('utils.imagedataset.load_test_data')
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size_h, size_w])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)
    print('utils.imagedataset.load_test_data.end')

    return img

def preprocessing(x):
    print('utils.imagedataset.processing')
    x = x/127.5 - 1 # -1 ~ 1
    print('utils.imagedataset.processing.end')
    return x

def augmentation(image, aug_img_h, aug_img_w):
    print('utils.imagedataset.augmentation')
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [aug_img_h, aug_img_w])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    return image

def save_images(images, size, image_path):
    print('utils.imagedataset.save_images')
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    print('utils.imagedataset.inverse_transform')
    return (images+1.) / 2

def imsave(images, size, path):
    print('utils.imagedataset.imsave')
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    print('utils.imagedataset.merge')
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def pytorch_xavier_weight_factor(gain=0.02, uniform=False) :
    print('utils.imagedataset.ipytorch_xavier_weight_factor')

    if uniform :
        factor = gain * gain
        mode = 'FAN_AVG'
    else :
        factor = (gain * gain) / 1.3
        mode = 'FAN_AVG'

    return factor, mode, uniform

def pytorch_kaiming_weight_factor(a=0.0, activation_function='relu', uniform=False) :
    print('utils.imagedataset.pytorch_kaiming_weight_factor')

    if activation_function == 'relu' :
        gain = np.sqrt(2.0)
    elif activation_function == 'leaky_relu' :
        gain = np.sqrt(2.0 / (1 + a ** 2))
    elif activation_function =='tanh' :
        gain = 5.0 / 3
    else :
        gain = 1.0

    if uniform :
        factor = gain * gain
        mode = 'FAN_IN'
    else :
        factor = (gain * gain) / 1.3
        mode = 'FAN_IN'

    return factor, mode, uniform

def get_files(dir, files):
    print('utils.imagedataset.get_files')
    pytorch_kaiming_weight_factor
    flist = os.listdir(dir)
    for file in flist:
        new_path = os.path.join(dir, file)
        if os.path.isfile(new_path):
            num = file.count('_')
            index = file.rstrip('.png').rsplit('_', num - 1)
            image_index = index[0]
            if image_index not in files:
                image = {'global': '',
                         'instance': [],
                         'background': ''}
                files[image_index] = image

            if len(index) == 1:
                files[image_index]['global'] = new_path
            elif index[-1] == 'background':
                files[image_index]['background'] = new_path
            else:
                files[image_index]['instance'].append(new_path)

        if os.path.isdir(new_path):
            get_files(new_path, files)


def load_pickle(filename, mode='rb'):
    with open(filename, mode) as f:
        loaded = pickle.load(f)
    return loaded


def dump_pickle(obj, filename, mode='wb'):
    with open(filename, mode) as f:
        pickle.dump(obj, f)

def get_bacth_size(batch_size, gup_num):
    if  gup_num > batch_size:
        return 1
    else:
        return batch_size // gup_num