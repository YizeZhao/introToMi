import tensorflow as tf
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


def softmax(x):

    o = np.copy(x)
    o = o - np.max(o)
    exponent = np.exp(o)
    return exponent/np.sum(exponent, axis=1, keepdims=True)


def computeLayer(X, W, b):

    return np.matmul(np.transpose(X), W) + b


def CE(target, prediction):
    return (-1) * np.mean(np.multiply(target, np.log(prediction)))


def gradCE(target, prediction):
    '''

    :param target: label y, one-hot encoded
    :param prediction: output from fully connected layers (ie.o), o = Wx+b
    :return: returns the gradient of cross-entropy loss with respect to z
    '''
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

def output_layer_gradients(p, y, h):
    '''

    :param o: ouput probability of softmax function, p = softmax(o), o = w_oh + b_o
    :param y: label, in one-hot
    :param h: h = relu(w_h + b_h), output from fidden layer
    :return: dl_dwo, dl_dbo, delta_o
    '''

    do_dbo = np.ones((1, (len(y))))

    # delta_o = gradCE(y, o)
    delta_o = (p - y)/y.shape[0]
    dl_dwo = np.matmul(np.transpose(h), delta_o)
    dl_dbo = np.matmul(do_dbo, delta_o)

    return dl_dwo, dl_dbo, delta_o


def hidden_layer_gradients(z, x, delta_o, w_o):
    '''

    :param z: z = w_h x + b_h
    :param x: input data
    :param delta_o: sensitivity of output layer
    :param y:
    :param w_o:
    :return:
    '''

    dh_dz = np.copy(z)
    dh_dbh = np.ones((1, (len(z))))
    dh_dz[dh_dz > 0] = 1
    dh_dz[dh_dz <= 0] = 0

    #print(dh_dz)
    delta_h = np.multiply(np.matmul(delta_o, np.transpose(w_o)), dh_dz)
    dl_dwh = np.matmul(np.transpose(x), delta_h)
    dl_dbh = np.matmul(dh_dbh, delta_h)

    return dl_dwh, dl_dbh


def accuracy(y, p):
    '''

    :param p: predicted probability, (N x 10)
    :param y: one-hot label, (N x 10)
    :return: correctly classified/N
    '''
    predicted_class = np.argmax(p, axis=1)
    # idx = np.arange(y.shape[0])  # for every sample
    # acc = np.mean(y[idx, predicted_class])  # +1 if correctly classified, divide by n
    truth = np.argmax(y, axis=1)
    print(predicted_class[:20])
    print(truth[:20])
    acc = np.sum(predicted_class == truth)/y.shape[0]
    return acc


def get_acc_loss(train_x, valid_x, test_x, train_y, valid_y, test_y, w_h, w_o, b_h, b_o):
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


def training(args, train_x, valid_x, test_x, train_y, valid_y, test_y):

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
        dl_dwo, dl_dbo, delta_o = output_layer_gradients(train_p, train_y, train_h)
        dl_dwh, dl_dbh = hidden_layer_gradients(train_z, train_x, delta_o, w_o)

        # update weights
        v_wh = gamma * v_wh + alpha * dl_dwh
        v_wo = gamma * v_wo + alpha * dl_dwo
        v_bh = gamma * v_bh + alpha * dl_dbh
        v_bo = gamma * v_bo + alpha * dl_dbo

        w_h = w_h - v_wh
        w_o = w_o - v_wo
        b_h = b_h - v_bh
        b_o = b_o - v_bo

        # w_h = w_h - alpha * dl_dwh
        # w_o = w_o - alpha * dl_dwo
        # b_h = b_h - alpha * dl_dbh
        # b_o = b_o - alpha * dl_dbo

        #train_loss = CE(train_y, train_p)
        # print(epoch, train_loss)

        if (epoch+1)%5 == 0:
            train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc = get_acc_loss(train_x, valid_x, test_x, train_y, valid_y, test_y, w_h, w_o, b_h, b_o)
            print("epoch:", epoch, " | train_loss:", train_loss, " train_acc:", train_acc, " valid_loss:", valid_loss,
                  " valid_acc:", valid_acc, " test_loss:", test_loss, " test_acc:", test_acc)

def relu(S):
    X = np.copy(S)
    X[S<0] = 0
    return X

def derivative_relu(S):
    dS = np.zeros_like(S)
    dS[S>0] = 1
    return dS

def softmax(S):
    X = np.exp(S) / np.sum(np.exp(S), axis=1, keepdims=True)
    return X

def computeLayer(X, W, b):
    S = X @ W + b
    return S

def avgCE(target, prediction):
    N = prediction.shape[0]
    L = - (1/N) * np.sum(target * np.log(prediction))
    return L

def gradCE(target, prediction):
    N = prediction.shape[0]
    dE_dX = - (1/N) * target / prediction
    return dE_dX

def accuracy(Y, Y_pred):
    j = np.argmax(Y_pred, axis=1)
    i = np.arange(Y.shape[0])
    return np.mean(Y[i, j])

def xavier_init(neurons_in, n_units, neurons_out):
    shape = (neurons_in, n_units)
    var = 2./(neurons_in + neurons_out)
    W = np.random.normal(0, np.sqrt(var), shape)
    return W


def parseData(data):
    num_data = data.shape[0]
    X = data.reshape(num_data, -1)
    return X

def init_weights(n_input, n_hidden, n_output):
    W = []
    W.append(None)
    W.append(xavier_init(n_input, n_hidden, n_output))
    W.append(xavier_init(n_hidden, n_output, 1))
    return W

def init_biases(n_input, n_hidden, n_output):
    b = []
    b.append(None)
    b.append(np.zeros((1, n_hidden)))
    b.append(np.zeros((1, n_output)))
    return b

def forward_propagation(X_input, W, b):
    X, S = [None]*3, [None]*3
    X[0] = X_input

    # UPDATE HIDDEN LAYER
    S[1] = X[0] @ W[1] + b[1]
    X[1] = relu(S[1])

    # UPDATE OUTPUT LAYER
    S[2] = X[1] @ W[2] + b[2]
    X[2] = softmax(S[2])

    return X, S

def backpropagation(X, S, W, Y):
    SENS = [None]*3

    # SEED SENSITIVITY
    N = Y.shape[0]
    SENS[2] = (1/N) * (X[2] - Y)

    # BACKPROPAGATION
    SENS[1] = (SENS[2] @ (W[2]).T) * derivative_relu(S[1])

    return SENS

def compute_gradients(X_input, Y, W, b):
    gradW = [0]*3
    gradb = [0]*3
    N = X_input.shape[0]

    # RUN PROPAGATIONS
    X, S = forward_propagation(X_input, W, b)
    SENS = backpropagation(X, S, W, Y)

    # GRADIENT of OUTPUT LAYER WEIGHTS + BIASES
    gradW[2] = (X[1]).T @ SENS[2]
    gradb[2] = np.sum(SENS[2], axis=0)

    # GRADIENT of HIDDEN LAYER WEIGHTS + BIASES
    gradW[1] = (X[0]).T @ SENS[1]
    gradb[1] = np.sum(SENS[1], axis=0)

    return gradW, gradb

def measure_performance(W, b):
    Y_pred, S = forward_propagation(X_train, W, b)
    train_loss = avgCE(Y_train, Y_pred[2])
    train_acc = accuracy(Y_train, Y_pred[2])

    Y_pred, S = forward_propagation(X_valid, W, b)
    valid_loss = avgCE(Y_valid, Y_pred[2])
    valid_acc = accuracy(Y_valid, Y_pred[2])

    Y_pred, S = forward_propagation(X_test, W, b)
    test_loss = avgCE(Y_test, Y_pred[2])
    test_acc = accuracy(Y_test, Y_pred[2])

    return train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc

def gradient_descent(X, Y, n_epochs, alpha, gamma, n_hidden_units):
    n_input_neurons = X.shape[1]
    n_output_neurons = Y.shape[1]

    # INITIALIZE WEIGHTS + BIASES
    W = init_weights(n_input_neurons, n_hidden_units, n_output_neurons)
    b = init_biases(n_input_neurons, n_hidden_units, n_output_neurons)
    VW_o = np.ones_like(W[2]) * 1e-5
    VW_h = np.ones_like(W[1]) * 1e-5
    Vb_o = np.ones_like(b[2]) * 1e-5
    Vb_h = np.ones_like(b[1]) * 1e-5

    # Create Loss/Accuracy dictionaries
    loss = {'train': [], 'valid': [], 'test': []}
    accuracy = {'train': [], 'valid': [], 'test': []}

    for t in range(n_epochs):
        gradW, gradb = compute_gradients(X, Y, W, b)
        print("EPOCH {}".format(t))
        # UPDATE OUTPUT LAYER
        VW_o = gamma * VW_o + alpha * gradW[2]
        W[2] = W[2] - VW_o
        Vb_o = gamma * Vb_o + alpha * gradb[2]
        b[2] = b[2] - Vb_o

        # UPDATE HIDDEN LAYER
        VW_h = gamma * VW_h + alpha * gradW[1]
        W[1] = W[1] - VW_h
        Vb_h = gamma * Vb_h + alpha * gradb[1]
        b[1] = b[1] - Vb_h

        # MEASURE PERFORMANCE
        train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc = measure_performance(W, b)
        loss['train'].append(train_loss)
        accuracy['train'].append(train_acc)
        loss['valid'].append(valid_loss)
        accuracy['valid'].append(valid_acc)
        loss['test'].append(test_loss)
        accuracy['test'].append(test_acc)

    return W, b, loss, accuracy


# PARSE THE DATA --------------------------------------------------------------------
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
X_train, X_valid, X_test = parseData(trainData), parseData(validData), parseData(testData)
Y_train, Y_valid, Y_test = convertOneHot(trainTarget, validTarget, testTarget)

def main():
    start_time = time.time()
    W, b, loss, accuracy = gradient_descent(X_train, Y_train,
                                    n_epochs=200,
                                    alpha=0.005,
                                    gamma=0.9,
                                    n_hidden_units=100)
    end_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))

    print("TRAINING ----------------------------")
    print("Loss:    ", loss['train'][-1])
    print("Accuracy:", accuracy['train'][-1])

    print("VALIDATION ----------------------------")
    print("Loss:    ", loss['valid'][-1])
    print("Accuracy:", accuracy['valid'][-1])

    print("TESTING ----------------------------")
    print("Loss:    ", loss['test'][-1])
    print("Accuracy:", accuracy['test'][-1])


    plt.plot(loss['train'], color='blue', label='training data')
    plt.plot(loss['valid'], color='red', label='validation data')
    plt.plot(loss['test'], color='green', label='test data')
    plt.legend()
    plt.title('Loss Curves')
    plt.ylabel('Average CE Loss')
    plt.xlabel('Epoch')
    plt.show()

    plt.plot(accuracy['train'], color='blue', label='training data')
    plt.plot(accuracy['valid'], color='red', label='validation data')
    plt.plot(accuracy['test'], color='green', label='test data')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

if __name__ == '__main__':
    main()