# -*- coding: utf-8 -*-
import tensorflow as tf
#%%

def my_model(img_rows, img_cols, color_type=1):
    inputs = tf.keras.Input(shape=(img_rows, img_cols, color_type), name='input')
   
    x = tf.keras.layers.Convolution2D(64, 3, 1, padding='same', activation='relu', name='cov1_64-3-1-same-relu')(inputs)
#    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same', strides=(2, 2), name='max_pool1-2-2')(x)
    x = tf.keras.layers.Convolution2D(64, 3, 1, padding='same', activation='relu', name='cov2_64-3-1-same-relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same', strides=(2, 2), name='max_pool1-2-2')(x)
    x = tf.keras.layers.Convolution2D(128, 3, 1, padding='same', activation='relu', name='cov3_128-3-1-same-relu')(x)
    x = tf.keras.layers.Convolution2D(128, 3, 1, padding='same', activation='relu', name='cov4_128-3-1-same-relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same', strides=(2, 2), name='max_pool2-2-2')(x)
    x = tf.keras.layers.Convolution2D(256, 3, 1, padding='same', activation='relu', name='cov5_256-3-1-same-relu')(x)
    x = tf.keras.layers.Convolution2D(256, 3, 1, padding='same', activation='relu', name='cov6_256-3-1-same-relu')(x)
    x = tf.keras.layers.Convolution2D(256, 3, 1, padding='same', activation='relu', name='cov7_256-3-1-same-relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same', strides=(2, 2), name='max_pool3-2-2')(x)
    x = tf.keras.layers.Convolution2D(512, 3, 1, padding='same', activation='relu', name='cov8_512-3-1-same-relu')(x)
    x = tf.keras.layers.Convolution2D(512, 3, 1, padding='same', activation='relu', name='cov9_512-3-1-same-relu')(x)
    x = tf.keras.layers.Convolution2D(512, 3, 1, padding='same', activation='relu', name='cov10_512-3-1-same-relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same', strides=(2, 2), name='max_pool4-2-2')(x)
    x = tf.keras.layers.Convolution2D(512, 3, 1, padding='same', activation='relu', name='cov11_512-3-1-same-relu')(x)
    x = tf.keras.layers.Convolution2D(512, 3, 1, padding='same', activation='relu', name='cov12_512-3-1-same-relu')(x)
    x = tf.keras.layers.Convolution2D(512, 3, 1, padding='same', activation='relu', name='cov13_512-3-1-same-relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same', strides=(2, 2), name='max_pool5-2-2')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
#    x = tf.keras.layers.Flatten(name='flatten')(x)
#    x = tf.keras.layers.Dense(512, activation='relu', name='dense1-512-relu')(x)
#    x = tf.keras.layers.Dropout(0.7)(x)
#    x = tf.keras.layers.Dense(1024, activation='relu', name='dense2-1024-relu')(x)
#    x = tf.keras.layers.Dropout(0.7)(x) 
#    x = tf.keras.layers.Dense(2048, activation='relu', name='dense3-2048-relu')(x)
#    x = tf.keras.layers.Dropout(0.5)(x)
#    x = tf.keras.layers.Dense(2048, activation='relu', name='dense3-2048-relu')(x)
#    x = tf.keras.layers.Dropout(0.5)(x)
#    x = tf.keras.layers.Dense(4096, activation='relu', name='dense4-4096-relu')(x)
#    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(10, activation='softmax', name='output')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
#    base_model = tf.keras.models.load_model('log/weights-2.0739.hdf5')
#    model.layers[0].set_weights(base_model.layers[0].get_weights())
#    model.layers[1].set_weights(base_model.layers[1].get_weights())
#    model.layers[2].set_weights(base_model.layers[2].get_weights())
#    model.layers[3].set_weights(base_model.layers[3].get_weights())
#    model.layers[4].set_weights(base_model.layers[4].get_weights())
#    model.layers[5].set_weights(base_model.layers[5].get_weights())
#    model.layers[6].set_weights(base_model.layers[6].get_weights())
#    model.layers[7].set_weights(base_model.layers[7].get_weights())
#    model.layers[8].set_weights(base_model.layers[8].get_weights())
#    model.layers[9].set_weights(base_model.layers[9].get_weights())
#    model.layers[10].set_weights(base_model.layers[10].get_weights())
#    model.layers[11].set_weights(base_model.layers[11].get_weights())
#    model.layers[12].set_weights(base_model.layers[12].get_weights())
#    model.layers[13].set_weights(base_model.layers[13].get_weights())
#    model.layers[14].set_weights(base_model.layers[14].get_weights())
#    sgd = tf.keras.optimizers.SGD(lr=1e-4, decay=1e-5, momentum=0.9, nesterov=True)
#    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    return model
#%%
def vgg_std16_model(img_rows, img_cols, color_type=1):
#    inputs = tf.keras.Input(shape=(img_rows, img_cols, color_type), name='input')
    model = tf.keras.applications.VGG16(include_top=False, pooling='max')

    x = model.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(10, activation='softmax', name='output')(x)
#    
    model2 = tf.keras.Model(inputs=model.input, outputs=output)
#    
    sgd = tf.keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model2.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

    return model2