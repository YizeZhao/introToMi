#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
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
        # trainData, trainTarget = Data, Target
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        #validData, validTarget = Data, Target
        testData, testTarget = Data[16000:], Target[16000:]
        #testData, testTarget = Data, Target
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
    x_copy = np.copy(x)
    x_copy[x < 0] = 0

    return x_copy

def grad_relu(x):
    dx = np.zeros_like(x)
    dx[x>0] = 1
    return dx


def softmax(x):

    o = np.copy(x)
    o = o - np.max(o)
    return np.exp(o) / np.sum(np.exp(o), axis=1, keepdims=True)


# def computeLayer(X, W, b):
#
#     return np.matmul(np.transpose(X), W) + b


def CE(target, prediction):
    return (-1) * np.mean(np.multiply(target, np.log(prediction)))


def gradCE(target, prediction):

    p = np.copy(prediction)
    y = np.copy(target)
    return p - y

def get_xavier(units_in, units_num, units_out):
    variance = 2.0 / (units_in + units_out)
    weights = np.random.normal(0, np.sqrt(variance), (units_in, units_num))
    return weights


def forward_propogation(x, w_h, w_o, b_h, b_o):
    z = np.add(np.matmul(x, w_h), b_h)
    h = relu(z)
    o = np.add(np.matmul(h, w_o), b_o)
    p = softmax(o)

    return z, h, o, p

def get_sensitivity(p, y, w_o):
    N = y.shape[0]
    delta_o = (p - y)/N
    delta_h = np.matmul(delta_o, np.transpose(w_o))

    return delta_o, delta_h


def accuracy(y, p):
    '''

    :param p: predicted probability, (N x 10)
    :param y: one-hot label, (N x 10)
    :return: correctly classified/N
    '''
    predicted_class = np.argmax(p, axis=1)
    truth = np.argmax(y, axis=1)
    acc = np.sum(predicted_class == truth)/y.shape[0]
    return acc


def get_acc_loss(w_h, w_o, b_h, b_o):
    z, h, o, train_p = forward_propogation(train_x, w_h, w_o, b_h, b_o)
    train_loss = CE(train_y, train_p)
    train_acc = accuracy(train_y, train_p)

    z, h, o, valid_p = forward_propogation(valid_x, w_h, w_o, b_h, b_o)
    valid_loss = CE(valid_y, valid_p)
    valid_acc = accuracy(valid_y, valid_p)

    z, h, o, test_p = forward_propogation(test_x, w_h, w_o, b_h, b_o)
    test_loss = CE(test_y, test_p)
    test_acc = accuracy(test_y, test_p)

    print('accuracy: ', train_acc, valid_acc, test_acc)

    return train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc


def training(args):

    gamma = args.gamma
    alpha = args.alpha

    feature_num = train_x.shape[1]
    class_num = train_y.shape[1]

    w_h = get_xavier(feature_num, args.hidden_num, class_num)
    w_o = get_xavier(args.hidden_num, class_num, 1)

    b_h = np.zeros((1, args.hidden_num))
    b_o = np.zeros((1, class_num))

    v_wh = 1e-5 * np.ones_like(w_h)
    v_wo = 1e-5 * np.ones_like(w_o)

    v_bh = 1e-5 * np.ones_like(b_h)
    v_bo = 1e-5 * np.ones_like(b_o)

    for epoch in range(args.epochs):
        #train_x, train_y = shuffle(train_x, train_y)

        train_z, train_h, train_o, train_p = forward_propogation(train_x, w_h, w_o, b_h, b_o)
        delta_o, delta_h = get_sensitivity(train_p, train_y, w_o)

        dl_dwo = np.matmul(np.transpose(train_h), delta_o)
        dl_dbo = np.sum(delta_o, axis=0)

        dl_dwh = np.matmul(np.transpose(train_x), delta_h)
        dl_dbh = np.sum(delta_h, axis=0)

        # update weights
        v_wh = gamma * v_wh + alpha * dl_dwh
        v_wo = gamma * v_wo + alpha * dl_dwo
        v_bh = gamma * v_bh + alpha * dl_dbh
        v_bo = gamma * v_bo + alpha * dl_dbo

        w_h = w_h - v_wh
        w_o = w_o - v_wo
        b_h = b_h - v_bh
        b_o = b_o - v_bo


        if (epoch+1)%5 == 0:
            train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc = get_acc_loss(w_h, w_o, b_h, b_o)
            print("epoch:", epoch, " | train_loss:", train_loss, " train_acc:", train_acc, " valid_loss:", valid_loss,
                  " valid_acc:", valid_acc, " test_loss:", test_loss, " test_acc:", test_acc)

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

train_x = trainData.reshape(trainData.shape[0], -1)
valid_x = validData.reshape(validData.shape[0], -1)
test_x = testData.reshape(testData.shape[0], -1)

train_y, valid_y, test_y = convertOneHot(trainTarget, validTarget, testTarget)
def main(args):
    # Get training data
    training(args)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--hidden_num', type=float, default=100)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.005)



    args = parser.parse_args()

    main(args)