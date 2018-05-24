# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import datetime
import pandas as pd
import input_data as inp
#import numpy as np
#%%
img_w = 224
img_h = 224
color_type = 3
batch_size = 2
capacity = 2000

#%%
def create_submission(predictions, test_id, info):
    print('create submission...')
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False, mode='a+')
    print('done!')

#%%
test_data, test_id = inp.load_test(img_w, img_h, color_type)

model = tf.keras.models.load_model('log/weights-1.9138.hdf5')

predictions = model.predict_generator(inp.get_test_batch(test_data, batch_size, img_w, img_h, color_type),
                                      steps=39863, 
                                      max_queue_size=capacity, 
                                      workers=1, verbose=1)
create_submission(predictions, test_id[0:len(predictions)], 'predict')
















