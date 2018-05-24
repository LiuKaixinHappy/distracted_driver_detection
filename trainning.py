# -*- coding: utf-8 -*-
import input_data as inp
import model as md
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#%%
img_w = 224
img_h = 224
color_type = 3
train_batch_size = 16
valid_batch_size = 16
capacity = 10
max_step=10000
#%%
unique_list_train = ['p002', 'p012', 'p014', 'p015', 
                     'p016', 'p021', 'p022', 'p024',
                     'p026', 'p035', 'p039', 'p041', 
                     'p042', 'p045', 'p047', 'p049',
                     'p050', 'p051', 'p052', 'p056',
                     'p061', 'p064', 'p066', 'p072',
                     'p075']
unique_list_valid = ['p081']

#%%
X_train_all, y_train_all, driver_id, unique_drivers = inp.load_train()
X_train, y_train, train_index = inp.copy_selected_drivers(X_train_all, 
                                                      y_train_all,
                                                      driver_id, 
                                                      unique_list_train)
X_valid, y_valid, test_index = inp.copy_selected_drivers(X_train_all, 
                                                     y_train_all, 
                                                     driver_id,
                                                     unique_list_valid)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)

#%%
model = md.my_model(img_w, img_h, color_type)
#model = tf.keras.models.load_model('log/weights-0.8149.hdf5')
tf.keras.utils.plot_model(model,'models/vgg_std16_model.png')
#%%
w = model.layers[-1].get_weights()
#%%
from tensorflow.python.keras._impl.keras import backend as K
import cv2
def visualize_class_activation_map(model, img_path, target_class):
    origin_img = inp.get_im_cv2([img_path], 224, 224, 3)
    class_weights = model.layers[-1].get_weights()[0]

    final_conv_layer = model.layers[17]
    get_output = K.function([model.layers[0].input],[final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([origin_img])

    conv_outputs = conv_outputs[0, :, :, :]
#    print(conv_outputs.shape)
    cam = np.zeros(dtype=np.float32, shape=(conv_outputs.shape[0], conv_outputs.shape[1]))

    for i, w in enumerate(class_weights[:, target_class]):
        cam += conv_outputs[:, :, i] * w

    cam = cv2.resize(cam, (224, 224))
    cam = 100 * cam
    plt.imshow(origin_img[0])
    plt.imshow(cam, alpha=0.8, interpolation='nearest')
    plt.show()

 #%%
impaths = ['data/train/c0/img_34.jpg','data/train/c1/img_6.jpg', 'data/train/c2/img_94.jpg', 'data/train/c3/img_5.jpg',
           'data/train/c4/img_14.jpg', 'data/train/c5/img_56.jpg', 'data/train/c6/img_0.jpg', 'data/train/c7/img_39404.jpg',
           'data/train/c8/img_26.jpg', 'data/train/c9/img_19.jpg']

#%%
cam = visualize_class_activation_map(model, impaths[7], 7)
#%%

from PIL import Image
def comb_imgs(o_imgs, col, row, each_width, each_height, new_type):
    new_img = Image.new(new_type, (each_width, each_height)) 
    new_img.paste(o_imgs[0], (0,0)) 
#    new_img.paste(o_imgs[1], (224,0, 224*2,224)) 
#    new_img.paste(o_imgs[2], (224*2,0,224*3,224))
#    new_img.paste(o_imgs[3], (224*3,0))
#    new_img.paste(o_imgs[4], (224*4,0))
#    new_img.paste(o_imgs[5], (0,224))
#    new_img.paste(o_imgs[6], (224,224))
#    new_img.paste(o_imgs[7], (224*2,224))
#    new_img.paste(o_imgs[8], (224*3,224))
#    new_img.paste(o_imgs[9], (224*4,224))
    return new_img

#%%
pred = model.predict(inp.get_im_cv2(impaths, 224, 224, 3))
#%%
for i in pred:
    i = i.tolist()
    print(i.index(max(i)))
#%%
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-5), loss='categorical_crossentropy', 
                  metrics=[tf.keras.metrics.categorical_accuracy, 
                           tf.keras.metrics.top_k_categorical_accuracy])
#tf.keras.utils.plot_model(model,'models/vgg_std16_model.png')
#%%
ckpt_path = 'log/weights-{val_loss:.4f}.hdf5'
ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path, 
                                          monitor='val_loss', 
                                          verbose=1, 
                                          save_best_only=True, 
                                          mode='min')
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='min', min_lr=1e-9)
#%%
result = model.fit_generator(generator=inp.get_train_batch(X_train, y_train, train_batch_size, img_w, img_h, color_type, True), 
          steps_per_epoch=1351, 
          epochs=50, verbose=1,
          validation_data=inp.get_train_batch(X_valid, y_valid, valid_batch_size,img_w, img_h, color_type, False),
          validation_steps=52,
          callbacks=[lr, ckpt, early_stop],
          max_queue_size=capacity,
          workers=1)
import pygame
pygame.mixer.init()
track = pygame.mixer.music.load("/home/roman/Music/time_over.wav")
pygame.mixer.music.play()
#%%
model.evaluate_generator(inp.get_train_batch(X_valid, y_valid, valid_batch_size,img_w, img_h, color_type, True),
                         steps=51)

 #%%
plt.figure
plt.plot(result.epoch, result.history['acc'], label='acc')
plt.plot(result.epoch, result.history['val_acc'], label='val_acc')
plt.scatter(result.epoch, result.history['acc'], marker='*')
plt.scatter(result.epoch, result.history['val_acc'], marker='*')
plt.legend(loc='under right')
plt.show()
#%%
plt.figure
plt.plot(result.epoch, result.history['loss'], label='loss')
plt.plot(result.epoch, result.history['val_loss'], label='val_loss')
plt.scatter(result.epoch, result.history['loss'], marker='*')
plt.scatter(result.epoch, result.history['val_loss'], marker='*')
plt.legend(loc='upper right')
plt.show()
#%%
results_acc = []
results_acc.append(result.history['acc'])
results_val_acc = []
results_val_acc.append(result.history['val_acc'])

results_loss = []
results_loss.append(result.history['loss'])
results_val_loss = []
results_val_loss.append(result.history['val_loss'])
#%% USE KERAS
##model2 = md.my_model(img_w, img_h, color_type)
##model2.load_weights('log/weights-2.1935.hdf5')
#
#model = tf.keras.models.load_model('log/weights-2.1935.hdf5')
#result = model.fit_generator(generator=inp.get_train_batch(X_train, y_train, train_batch_size, img_w, img_h, color_type), 
#          steps_per_epoch=1350, 
#          epochs=5, verbose=1,
#          validation_data=inp.get_train_batch(X_valid, y_valid, valid_batch_size,img_w, img_h, color_type),
#          validation_steps=206,
#          callbacks=[ckpt],
#          max_queue_size=capacity,
#          workers=1)
#%% USE TF
#X_train_batch, y_train_batch = inp.get_batch(X_train, y_train, 
#                                             img_w, img_h, color_type, 
#                                             train_batch_size, capacity)
#X_valid_batch, y_valid_batch = inp.get_batch(X_valid, y_valid, 
#                                             img_w, img_h, color_type, 
#                                             valid_batch_size, capacity)

#with tf.Session() as sess:
# 
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    try:
#        for step in np.arange(max_step):
#            if coord.should_stop() :
#                break
#            X_train, y_train = sess.run([X_train_batch, 
#                                             y_train_batch])
#            X_valid, y_valid = sess.run([X_valid_batch,
#                                             y_valid_batch])
#            #learnrate_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1,factor=0.2, min_lr=0.00000001)        
#            ckpt_path = 'log/weights-{val_loss:.4f}.hdf5'
#            ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path, 
#                                                      monitor='val_loss', 
#                                                      verbose=1, 
#                                                      save_best_only=True, 
#                                                      mode='min')
#            model.fit(X_train, y_train, batch_size=64, 
#                          epochs=50, verbose=1,
#                          validation_data=(X_valid, y_valid),
#                          callbacks=[ckpt])
#            
#            del X_train, y_train, X_valid, y_valid
#
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)
#    sess.close()
