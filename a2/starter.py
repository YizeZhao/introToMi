import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):

    return max(np.zeros(x.shape), x)


def softmax(x):

    x = x - max(x)
    exponent = np.exp(x)
    return exponent/np.sum(exponent, axis=1, keepdims=True)


def computeLayer(X, W, b):

    return np.matmul(np.transpose(X), W) + b


def CE(target, prediction):

    return (-1) * np.mean(np.matmul(target, np.log(prediction)))


def gradCE(target, prediction):
    '''

    :param target: label y, one-hot encoded
    :param prediction: output from fully connected layers (ie.Z), Z = Wx+b
    :return: returns the gradient of cross-entropy loss with respect to z
    '''
    return softmax(prediction) - target

def get_xavier(units_in, units_out, units_num):
    variance = 2 / (units_in + units_out)
    weights = np.random.normal(0, np.sqrt(variance), (units_in, units_num))
    return weights


def output_layer_gradients(o, y, h):
    '''

    :param o: ouput probability of softmax function, p = softmax(o), o = w_oh + b_o
    :param y: label, in one-hot
    :param h: h = relu(w_h + b_h), output from fidden layer
    :return: dl_dwo, dl_dbo, delta_o
    '''

    do_dbo = np.ones(1, (len(y)))

    delta_o = gradCE(y, o)
    dl_dwo = np.matmul(np.transpose(h), delta_o)
    dl_dbo = np.matmul(do_dbo, delta_o)

    return dl_dwo, dl_dbo, delta_o


def hidden_layer_gradients(z, x, delta_o, y, w_o):
    '''

    :param z: z = w_h x + b_h
    :param x: input data
    :param delta_o: sensitivity of output layer
    :param y:
    :param w_o:
    :return:
    '''

    #dh_dz = np.zeros(len(z))
    dh_dz = [1*(z_n > 0) for z_n in z]
    print(dh_dz)

def main():
    hidden_layer_gradients([1,-1,2,-3,4, 0, -1], 0, 0, 0, 0)



if __name__ == '__main__':
    main()