import tensorflow as tf
import keras
import tensorflow.python.client.device_lib

assert tf.__version__ >= '2.0'
from keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Lambda, Input

l_data_xml = []

import matplotlib as plot
import cv2
import os
import glob

import numpy as np
from tensorflow.python.keras import backend
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.datasets import make_classification
from keras.wrappers.scikit_learn import KerasClassifier
import xml.etree.ElementTree as ET

print(tensorflow.python.client.device_lib.list_local_devices())
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
sess = tf.compat.v1.Session(config=config)
backend.set_session(sess)

path_1 = "/home/sisi/Documents/face_maskDS"
path_2 = "/home/sisi/Documents/face_maskDS/face_maskDS/annotations"
path_3 = "/home/sisi/Documents/face_maskDS/face_maskDS/images"

xml_df = []
img_df = []

print(path_1)

maxDX = -1
maxDY = -1
list_ann = sorted(glob.glob(os.path.join(path_2, '*.xml')))
wmask_all, wout_mask_all, incmask_all = [], [], []
wmask_train, woutmask_train = [], []
wmask_test, woutmask_test = [], []
face_cuts = []  # np.zeros((len(list_ann), 128, 128))
list_imgs = []
all_mask = []
labels = []
objects = {}
train_args = {'input_len': 340}
print('Reading and grayscaling images plus localising faces.. ')
for i in list_ann:
    tree = ET.parse(i)
    root = tree.getroot()
    filename = root.find('filename').text
    objects[filename] = {}
    imgpath = i.replace('annotations', 'images').replace('xml', 'png')
    objects[filename]['imgpath'] = imgpath
    objs = root.findall('object')
    for item, c in zip(objs, range(len(objs))):
        # iterate child elements of item
        for child in item:
            if child.tag == 'name':
                c = str(c)
                objects[filename][c] = {}
                if child.text == 'without_mask':
                    objects[filename][c]['cat'] = 0
                elif child.text == 'with_mask':
                    objects[filename][c]['cat'] = 1
                elif child.text == 'mask_weared_incorrect':
                    objects[filename][c]['cat'] = 2
            # special checking for namespace object content:media
            elif child.tag == 'bndbox':
                objects[filename][c]['bndbox'] = []
                for bnd in child:
                    objects[filename][c]['bndbox'].append(int(bnd.text))
                tmp = objects[filename][c]['bndbox']
                img = cv2.imread(imgpath)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                face_cuts.append(img[tmp[1]:tmp[3], tmp[0]:tmp[2]])
                face_cuts.append(img[tmp[1]:tmp[3], tmp[0]:tmp[2]])
                img_cut = face_cuts[len(face_cuts) - 1]
                img = np.concatenate((img_cut, np.zeros((340 - img_cut.shape[0], img_cut.shape[1]))), axis=0)
                img = np.concatenate((img, np.zeros((img.shape[0], 340 - img_cut.shape[1]))), axis=1)
                face_cuts[len(face_cuts) - 1] = img
                all_mask.append(face_cuts[len(face_cuts) - 1])
                # if objects[filename][c]['cat'] == 0:
                #     wmask_all.append(face_cuts[len(face_cuts)-1])
                # if objects[filename][c]['cat'] == 1:
                #     wout_mask_all.append(face_cuts[len(face_cuts) - 1])
                # if objects[filename][c]['cat'] == 2:
                #     incmask_all.append(face_cuts[len(face_cuts) - 1])
                labels.append(objects[filename][c]['cat'])

                if abs(tmp[1] - tmp[3]) > maxDY:
                    maxDY = abs(tmp[1] - tmp[3])
                if abs(tmp[0] - tmp[2]) > maxDX:
                    maxDX = abs(tmp[0] - tmp[2])

xml_df = []
img_df = []

print("DataSource unzipped and preprocessing of faces cuts is finished!")
print('Max X = ', maxDX, 'Max Y = ', maxDY)
print("Dividing train/val sets as 70%/30%")
fold_var = 1

# all_mask = np.stack(all_mask)
all_mask_train, all_mask_test = all_mask[0:int(len(all_mask) * .7) - 1], all_mask[int(len(all_mask) * .7) - 1: int(
    len(all_mask)) - 1]
wmask_train, woutmask_train = wmask_all[0:int(len(wmask_all) * .7) - 1], wout_mask_all[
                                                                         0:int(len(wout_mask_all) * .7) - 1]
wmask_test, woutmask_test = wmask_all[int(len(wmask_all) * .7): len(wmask_all) - 1], wout_mask_all[
                                                                                     int(len(wout_mask_all) * .7): len(
                                                                                         wout_mask_all) - 1]
all_mask_train, all_mask_test = np.stack(all_mask_train), np.stack(all_mask_test)
labels_train = np.stack(labels[0:len(all_mask_train)])

tf.keras.backend.set_floatx('float64')
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# tf.compat.v1.disable_eager_execution()
backend.set_session(sess)
# model = Sequential()
print("Setting adaptively shaped CNN..\n")

# def creatModel():
class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        # generic_l = keras.layers.Layer(name='generic')
        self.l_input = Input(shape=(train_args['input_len'], train_args['input_len']), name="imag_input")
        self.l_face0 = Conv1D(256, padding="same", kernel_size=2, activation='relu',
                              input_shape=(train_args['input_len'], train_args['input_len']))#(self.l_input)
        # l_face1 = Conv1D(256, kernel_size=2, activation='relu')(l_face0)
        self.l_face1 = Conv1D(128, kernel_size=2, activation='relu')#(self.l_face0)
        self.l_face2 = Conv1D(64, kernel_size=2, activation='relu')#(self.l_face1)
        self.l_face3 = Conv1D(32, kernel_size=2, activation='relu')#(self.l_face2)
        self.l_face4 = Conv1D(16, kernel_size=2, activation='relu')#(self.l_face3)
        self.l_face5 = Conv1D(8, kernel_size=2, activation='relu')#(self.l_face4)
        self.l_face6 = Conv1D(4, kernel_size=2, activation='relu')#(self.l_face5)
        # l_face8 = Conv2D(2, kernel_size=2, activation='relu')(l_face7)
        # l_matify = l_model.add_layer
        # l_persc = l_model.add_layer
        self.l_flat = Flatten(data_format='channels_last')#(self.l_face6)

        self.l_fc1 = Dense(4, activation='softmax')#(self.l_flat)  # wmask
        self.l_output = Dense(1, activation='softmax')#(self.l_fc1)  # woutmask
        self.cce = tf.keras.losses.CategoricalCrossentropy()
        # self.l_output = Lambda(self.l2_norm, name="lambda_layer")([np.array(self.l_fc1, self.l_fc2)])

    def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              **kwargs):

        tf.keras.Model.compile(self, optimizer, loss, metrics)

    def call(self, inputs, training=True, mask=None):
        # print(inputs)
        # m = self.l_input(inputs)
        # print('inputs shape: ', inputs.shape)
        m = self.l_face6(self.l_face5(self.l_face4(self.l_face3(self.l_face2(self.l_face1(self.l_face0(inputs)))))))

        # if training:
        m = self.l_flat(m)
        # print('Tensor shape after flattening: ' , m.shape)
        m = self.l_fc1(m)  # wmask
        # print('Tensor shape after Dense layer: ', m.shape)
        # m2 = self.l_fc2(m1)  # woutmask
        # m = self.l_output(m1)

        out = self.l_output(m)
        # print('Output: ', out)

        return out

    def loss(self, x, y):
        # x, y = args[0], args[1]
        # compile
        return self.cce(x, y).numpy()
        # return (x - y) ** 2

#  = Output((1,), name="acc_output")(lmbda)y_pred
model = CustomModel()
# k_model = tf.keras.Model(inputs=model.l_input, outputs=model.l_output, name='CoV_Mask_Detctr')
model.compile(optimizer='Adam', loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=tf.keras.metrics.Accuracy())
# model.ru
# n_eagerly = False
# l_dist = model.add(output_layer)#l2-dist between two dense vectors

checkpoint = tf.keras.callbacks.ModelCheckpoint('/home/sisi/Documents/face_maskDS/saved_models/checkpoint_makssksksss',
                                                monitor='val_accuracy', verbose=3,
                                                save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print("Shhh..! Training Begins Now!")

all_acc = []
# k_model = KerasClassifier(build_fn=creatModef call(self, inputs, training=True, mask=None):

model_kfolds = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
# for x, y in model_kfolds.split(all_mask_train, labels_train):m1
# optimizer = tf.optimizers.Adam()

np_all_mask = np.stack(all_mask_train)
np_labels = np.stack(labels_train)

# with tf.GradientTape() as t:
for x, y in model_kfolds.split(np_all_mask, np_labels):
    results = model.fit(x=np_all_mask[x], y=np_labels[x], validation_data=np_all_mask[y], validation_split=0.25,
                        steps_per_epoch=len(all_mask_train) // 16,
                        epochs=4, batch_size=16, verbose=3,
                        callbacks=checkpoint)
    all_acc.append(results)
# current_loss = model.loss(results, labels_train)
# grads = t.gradient(current_loss, [])
# optimizer.apply_gradients(zip(grads, []))
# print('\nCatergorical loss: ', results)

# cv_result = model.fit(x=all_mask_train[y], y=all_mask_test[y],
#                       epochs=100, batch_size=16, verbose=1)

# for i in range(0, int(len(all_mask_train)/16), 16):
#     results = train(model, all_mask_train[i: i+16], labels_train[i:i+16])
#     print('cv_result = ', results)
#     all_acc.append(results)

all_acc = np.stack(all_acc)

backend.clear_session()
print("Baseline: %.2f%% (%.2f%%)" % (all_acc.mean() * 100, all_acc.std() * 100))

# estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
# model.load_weights(path_1+"/saved_models/model_" + str(fold_var) + ".h5")
#
# for x, y in model_kfolds.split(all_mask_train, labels[0:int(len(labels)*.7)-1]):
# predicted_labels = k_model.fit(x, y)n K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)
# fit_params = {'epochs': 50, 'lr': 5e-3}
# results = cross_val_score(model, all_mask_train, labels[0:len(all_mask_train)],
#                           cv=model_kfolds, verbose=1, scoring='accuracy', fit_params={'epochs': 50, 'lr': 5e-3})# all_acc.append(results)
# print('Accuracy = ', results)
# results = dict(zip(model.metrics_names, results))
# all_acc = np.stack(all_acc)
# VALIDATION_ACCURACY.append(results['accuracy'])
# VALIDATION_LOSS.append(results['loss'])
