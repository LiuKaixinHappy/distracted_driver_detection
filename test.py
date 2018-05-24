# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import input_data as inp
import model as md
import tensorflow as tf
import matplotlib.pyplot as plt
#%%
model = tf.keras.models.load_model('log/weights-2.1935.hdf5')
tf.keras.utils.plot_model(model,'models/model_2.1935.png')
#%%
#import numpy as np
#import input_data as inp
#import tensorflow as tf
#import numpy as np
#import predict as prd
##%%TEST
#
#import matplotlib.pyplot as plt
#BATCH_SIZE = 8
#CAPACITY = 512
#IMG_W = 224
#IMG_H = 224
#CHANEL = 3
##%%
#unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
#                 'p026', 'p035', 'p039',
#                 'p041', 'p042', 'p045', 'p047', 'p049',
#                 'p050', 'p051', 'p052', 'p056']
#unique_list_valid = ['p061', 'p064', 'p066', 'p072',
#                     'p075', 'p081']
#X_train_all, y_train_all, driver_id, unique_drivers = inp.get_files()
#X_train, y_train, train_index = inp.copy_selected_drivers(X_train_all, 
#                                                      y_train_all,
#                                                      driver_id, 
#                                                      unique_list_train)
#X_valid, y_valid, test_index = inp.copy_selected_drivers(X_train_all, 
#                                                     y_train_all, 
#                                                     driver_id,
#                                                     unique_list_valid)
#X_train_batch, y_train_batch = inp.get_batch(X_train, y_train, 
#                                         IMG_W, IMG_H, CHANEL, BATCH_SIZE, CAPACITY)
##%%
#test_data, test_id = prd.load_test(IMG_W, IMG_H, CHANEL)
#
#X_test = prd.get_batch(test_data, 
#                       IMG_W, IMG_H, CHANEL, 
#                       BATCH_SIZE, CAPACITY)
#model = tf.keras.models.load_model('log/weights-2.1837.hdf5')
#prediction = []
#with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    try:
#        while not coord.should_stop() and i < 2:
##            X_train, y_train = sess.run([X_train_batch, 
##                                                     y_train_batch])
#            X_test = sess.run(X_test)
##            prediction.append(X_test)
##            test_prediction = model.predict(X_test, batch_size=64, verbose=1)
##            prd.create_submission(test_prediction, test_id, 'predict')
##            del test_prediction
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
##                print('label:%d' % y_train[j])
#                plt.imshow(X_test[j, :, :, :])
#                plt.show()
#            i+=1
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)


