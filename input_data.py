# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import glob
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
#%%
def get_driver_data():
    driver_dict = dict()
    path = os.path.join('data', 'driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        driver_dict[arr[2]] = arr[0]
    f.close()
    return driver_dict
#%%
def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    shuffle_driver_id = np.arange(len(driver_id))
    np.random.shuffle(shuffle_driver_id)
    
    for i in shuffle_driver_id:
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    #data = np.array(data, dtype=np.float32)
    #target = np.array(target, dtype=np.float32)
    #index = np.array(index, dtype=np.uint32)
    return data, target, index
#%%
def load_train():
    '''
    Returns:
        list of train data dir, labels, driver ids and unique dirvers
    '''
    X_train = []
    y_train = []
    driver_id = []
    driver_data = get_driver_data()

    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('data', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            X_train.append(fl)
            y_train.append(j)
            driver_id.append(driver_data[flbase])
    print('There are %d training data' % (len(X_train)))
    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, driver_id, unique_drivers
#%%
def load_test(img_rows, img_cols, color_type=1):
    print('Read test images')
    # 读取测试数据
    path = os.path.join('data', 'test', '*.jpg')
    # 找到path下的所有jpg文件
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0

    # 所有jpg文件总数除以10向下取整
    thr = math.floor(len(files)/10)
    for fl in files:
        # 获取图片名，xxxx.jpg
        flbase = os.path.basename(fl)
        # 将图片路径加入测试集
        X_test.append(fl)
        # 将图片对应的名称加入X_test_id
        X_test_id.append(flbase)
        # 累计数加1
        total += 1
        # 如果读到批数，输出从xxxx个文件中读取xxxx张图
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))
    # 返回测试集与测试集图片对应的名称
    return X_test, X_test_id
#%%
# 使用cv2將圖片resize
def get_im_cv2(paths, img_rows, img_cols, color_type=1):
    imgs = []
    for path in paths:
        if color_type == 1:
            img = cv2.imread(path, 0)
        elif color_type == 3:
            img = cv2.imread(path)
        resized = cv2.resize(img, (img_cols, img_rows))
        imgs.append(resized)
    return np.array(imgs).reshape(len(paths), img_rows, img_cols, color_type)


#%%
#def img_augmentation(X_train, y_train):
#    num = 4 * X_train.shape[0]
#    shuffle_id = np.arange(num)
#    np.random.shuffle(shuffle_id)
#    imgs = []
#    labels = []
#    shuffle_imgs = []
#    shuffle_labels = []
#    for i in range(X_train.shape[0]):
#        imgs.append(X_train[i])
#        labels.append(y_train[i])
#        
#        imgs.append(tf.keras.preprocessing.image.random_shift(X_train[i], 0.8, 0.8))
#        labels.append(y_train[i])
#        
#        imgs.append(tf.keras.preprocessing.image.flip_axis(X_train[i], 0))
#        labels.append(y_train[i])
#        
#        imgs.append(tf.keras.preprocessing.image.flip_axis(X_train[i], 1))
#        labels.append(y_train[i])
##    
#    for i in shuffle_id:
#        shuffle_imgs.append(imgs[i])
#        shuffle_labels.append(labels[i])
#    return np.array(shuffle_imgs), np.array(shuffle_labels)

#%%
import random
def img_augmentation(X_train, y_train):
    for i in range(X_train.shape[0]):
        rand = random.randint(1, 4)
        if rand == 2:
            X_train[i] = tf.keras.preprocessing.image.random_shift(X_train[i], 0.4, 0.2)
        elif rand == 3:
            X_train[i] = tf.keras.preprocessing.image.flip_axis(X_train[i], random.randint(0,1))
        else:
            X_train[i] = tf.keras.preprocessing.image.random_rotation(X_train[i], 90)
    return X_train, y_train
#%%
def get_train_batch(X_train, y_train, batch_size, img_w, img_h, color_type, is_augmentation=False):
    while 1:
        for i in range(0, len(X_train), batch_size):
            x = get_im_cv2(X_train[i:i+batch_size], img_w, img_h, color_type)
            y = y_train[i:i+batch_size]
            if is_augmentation:
                x, y = img_augmentation(x, y)
            yield({'input': x}, {'output': y})
    
#%%
def get_test_batch(X_test, batch_size, img_w, img_h, color_type):
    while 1:
        for i in range(0, len(X_test), batch_size):
            yield({'input': get_im_cv2(X_test[i:i+batch_size], img_w, img_h, color_type)})
            
#%% USE TF
def get_batch(X_train, y_train, img_w, img_h, color_type, batch_size, capacity):
    '''
    Args:
        X_train: train img names
        y_train: train labels
        img_w: image width
        img_h: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        X_train_batch: 4D tensor [batch_size, width, height, chanel],\
                        dtype=tf.float32
        y_train_batch: 1D tensor [batch_size], dtype=int32
    '''
    X_train = tf.cast(X_train, tf.string)

    y_train = tf.cast(y_train, tf.int32)
    # y_train = tf.keras.utils.to_categorical(y_train, 10)
    
    # make an input queue
    input_queue = tf.train.slice_input_producer([X_train, y_train])
    

    y_train = input_queue[1]
    X_train_contents = tf.read_file(input_queue[0])
    X_train = tf.image.decode_jpeg(X_train_contents, channels=color_type)

    #####################################
    # data argumentation should go to here
    #####################################
    X_train = tf.image.resize_images(X_train, [img_h, img_w], 
                                     tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # mean_pixel = [123.68, 116.779, 103.939]
    # for c in range(3):
    #     X_train[:, :, c] = X_train[:, :, c] - mean_pixel[c]

    X_train_batch, y_train_batch = tf.train.batch([X_train, y_train],
                                                  batch_size=batch_size,
                                                  num_threads=64,
                                                  capacity=capacity)
    y_train_batch = tf.one_hot(y_train_batch, 10)
#    X_train_batch, y_train_batch=img_augmentation(X_train_batch, y_train_batch)
    return X_train_batch, y_train_batch

    
    
