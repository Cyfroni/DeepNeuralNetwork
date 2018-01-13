import tflearn
import numpy as np

def preprocessA(data):
    for i in range(len(data)):
      data[i][0] = 1. if data[i][0] == 'F' else 0.
    return np.array(data, dtype=np.float32)


def preprocessR(data):
    w, h = 2, len(data)
    Matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(len(data)):
        if data[i] == 'No':
            Matrix[i][0] = 1
            Matrix[i][1] = 0
        else:
            Matrix[i][0] = 0
            Matrix[i][1] = 1
    return Matrix

def preprocess(data, columns_to_ignore):
    # Sort by descending id and delete columns
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
      # Converting 'sex' field to float (id is 1 after removing labels column)
      data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data, dtype=np.float32)


'''
A, R = tflearn.data_utils.load_csv("KaggleV2-May-2016.csv",
                                   target_column=-1, columns_to_ignore=[1, 2, 4, 5, 7], has_header=True)
A = preprocessA(A)
Re = preprocessR(R)
inputl = 8
test = 70000
end = 110527
activ = 'relu'
out = 'softmax'
'''
A, R = tflearn.data_utils.load_csv("excel.csv",
                                   target_column=-1, columns_to_ignore=[1, 2, 4, 5, 7], has_header=True)
A = preprocessA(A)
Re = preprocessR(R)
inputl = 8
test = 70095-(110527 - 70000)
end = 70095
activ = 'relu'
out = 'softmax'


'''
from tflearn.data_utils import load_csv
A, Re = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)
to_ignore=[1, 6]
data = preprocess(A,to_ignore)
inputl = 6
test = 900
end = 1309
activ = 'relu'
out = 'softmax'
'''


# Building deep neural network
input_layer = tflearn.input_data(shape=[None, inputl])
dense1 = tflearn.fully_connected(input_layer, 100, activation=activ,
                                 regularizer='L2', weight_decay=0.001)
dense2 = tflearn.fully_connected(dense1, 100, activation=activ,
                                 regularizer='L2', weight_decay=0.001)
dense3 = tflearn.fully_connected(dense2, 100, activation=activ,
                                 regularizer='L2', weight_decay=0.001)
act = tflearn.fully_connected(dense3, 2, activation=out)

opt = tflearn.Adam(learning_rate=0.001)
net = tflearn.regression(act, optimizer=opt)

#net = skflow.models.logistic_regression(net, 2, class_weight=c_w);

#net = tflearn.regression(softmax)


model = tflearn.DNN(net)
model.fit(A[0:test], Re[0:test], n_epoch=100, show_metric=True, validation_set=(A[test:end], Re[test:end]))









# Regression using SGD with learning rate decay and Top-3 accuracy
'''adam = tflearn.Adam(learning_rate=0.001)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=adam, metric=top_k,
                         loss='categorical_crossentropy')'''


# Training
'''model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(A, R, n_epoch=20,
          show_metric=True, run_id="dense_model")'''

