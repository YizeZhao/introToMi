import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import time


def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


def MSE(W, b, x, y, reg):
    # compute loss
    sample_len = len(y)
    # z = np.zeros(sample_len)
    # for sample in range(sample_len):
    #     z[sample] = np.dot(x[sample], W) + b
    l_d_1 = np.matmul(x, W) + b
    l_d_2 = l_d_1 ** 2
    l_w_2 = W ** 2
    l = np.mean((y - l_d_1)**2) + reg/2 * sum(l_w_2)

    return l


def gradMSE(W, b, x, y, reg):
    # compute gradients
    # x: 3500*784
    # W: 784*1
    sample_len = len(y)
    l_d_1 = np.matmul(x, W) + b - y
    l_d_1 = np.transpose(l_d_1)
    dl_dw = (2 / sample_len) * np.matmul(l_d_1,x) + np.transpose(reg * W)
    dl_db = (2 / sample_len) * np.sum(l_d_1,axis=1)
    return dl_dw, dl_db


def crossEntropyLoss(W, b, x, y, reg):
    z = np.matmul(x, W) + b
    y_hat = 1 / (1 + np.exp((-1)*z))

    loss = np.mean((-1) * y * np.log(y_hat) - (1-y) * np.log(1-y_hat)) + reg/2 * sum(W ** 2)
    return loss


def gradCE_old(W, b, x, y, reg):
    z = np.matmul(x, W) + b
    y_hat = 1 / (1 + np.exp((-1)*z))
    for i in range(len(y_hat)):
        if abs(y_hat[i]-1) < 0.1:
            y_hat[i] -= 0.001
        elif y_hat[i] < 0.1:
            y_hat[i] += 0.001

    inter_mat_dw = np.transpose((-1)*y/y_hat + (1-y)/(1-y_hat) * (np.exp((-1)*z)/(1 + np.exp(-1*z))**2))
    dl_dw = np.mean(np.matmul(inter_mat_dw, x)) + np.transpose(reg * W)
    dl_db = np.mean((-1)*y/y_hat - (1-y)/(1-y_hat) * (np.exp((-1)*z)/(1 + np.exp(-1*z))**2))
    # print('dl_dw: ', np.shape(dl_dw))
    # print('dl_db: ', np.shape(dl_db))
    return dl_dw, dl_db


def gradCE(W, b, x, y, reg):
    n = np.shape(y)[0]
    z = np.matmul(x, W) + b
    y_hat = 1 / (1 + np.exp((-1)*z))

    dl_dw = (1/n) * np.matmul(np.transpose(y_hat - y), x) + np.transpose(reg * W)
    dl_db = (1/n) * np.sum((y_hat - y))
    # print('dl_dw: ', np.shape(dl_dw))
    # print('dl_db: ', np.shape(dl_db))
    return dl_dw, dl_db


def grad_descent(W, b, train_x, train_y, alpha, epochs, reg, error_tol, valid_x, valid_y, test_x, test_y):
    # trainning loop
    train_loss_rec = []
    train_acc_rec = []
    valid_loss_rec = []
    valid_acc_rec = []
    test_loss_rec = []
    test_acc_rec = []
    last_w = 0
    cur_w = 0

    for epoch in range(epochs):
    # epoch = 0
    # while True:
    #     epoch += 1
        if args.lossType == 'MSE':
            dl_dw, dl_db = gradMSE(W, b, train_x, train_y, reg)
            loss = MSE
        elif args.lossType == 'CE':
            dl_dw, dl_db = gradCE(W, b, train_x, train_y, reg)
            loss = crossEntropyLoss
        W = W - np.transpose(alpha * dl_dw)
        b = b - alpha * dl_db

        # cur_w = math.sqrt(sum(W**2))
        # if abs(cur_w - last_w) < error_tol:
        #     break
        # last_w = math.sqrt(sum(W ** 2))

        if (epoch+1) % 10 == 0:
            train_loss = loss(W, b, train_x, train_y, args.reg)
            train_acc = get_acc(W, b, train_x, train_y)
            valid_acc = get_acc(W, b, valid_x, valid_y)
            valid_loss = loss(W, b, valid_x,  valid_y, reg)
            test_acc = get_acc(W, b, test_x, test_y)
            test_loss = loss(W, b, test_x, test_y, reg)

            train_loss_rec.append(train_loss)
            train_acc_rec.append(train_acc)
            valid_loss_rec.append(valid_loss)
            valid_acc_rec.append(valid_acc)
            test_loss_rec.append(test_loss)
            test_acc_rec.append(test_acc)

            print("epoch:", epoch, " | train_loss:", train_loss, " train_acc:", train_acc, " valid_loss:", valid_loss, " valid_acc:", valid_acc, " test_loss:", test_loss, " test_acc:", test_acc)

    plot_loss_acc(train_loss_rec, train_acc_rec, valid_loss_rec, valid_acc_rec, test_loss_rec, test_acc_rec)
    # train_loss = loss(W, b, train_x, train_y, args.reg)
    # train_acc = get_acc(W, b, train_x, train_y)
    # valid_acc = get_acc(W, b, valid_x, valid_y)
    # valid_loss = loss(W, b, valid_x, valid_y, reg)
    # test_acc = get_acc(W, b, test_x, test_y)
    # test_loss = loss(W, b, test_x, test_y, reg)
    # print("epoch:", epoch, " | train_loss:", train_loss, " train_acc:", train_acc, " valid_loss:", valid_loss, " valid_acc:", valid_acc, " test_loss:", test_loss, " test_acc:", test_acc)
    return W, b


def get_acc(W, b, x, y):
    y_pred = np.dot(x, W) + b
    corr = 0
    for sample in range(len(y)):
        if (y_pred[sample] > 0.5 and y[sample] == 1) or (y_pred[sample] < 0.5 and y[sample] == 0):
            corr = corr + 1
    return corr/len(y)


def plot_loss_acc(train_loss_rec, train_acc_rec, valid_loss_rec, valid_acc_rec, test_loss_rec, test_acc_rec):
    x = np.arange(len(train_loss_rec)) + 1
    x = x * 10
    plt.subplot(211)
    plt.plot(x, train_loss_rec, label = "train_loss")
    plt.plot(x, valid_loss_rec, label="validation_loss")
    plt.plot(x, test_loss_rec, label="test_loss")
    plt.xlabel('number of epochs')
    plt.ylabel('loss')
    plt.title('Loss VS epochs')
    plt.legend(loc = 'upper right')

    plt.subplot(212)
    plt.plot(x, train_acc_rec, label = "train_accuracy")
    plt.plot(x, valid_acc_rec, label = "validation_accuracy")
    plt.plot(x, test_acc_rec, label="test_accuracy")
    plt.xlabel('number of epochs')
    plt.ylabel('accuracy')
    plt.title('Accuracy VS epochs')
    plt.legend(loc = 'lower right')
    plt.show()


def get_analytical(train_x, train_y, valid_x, valid_y, test_x, test_y, args):
    n = np.shape(train_x)[0]
    x_0 = np.ones((n, 1))
    X = np.hstack((x_0, train_x))

    psydo_inv = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)),  np.transpose(X))
    w_combined = np.matmul(psydo_inv, train_y)
    W = w_combined[1:]
    b = w_combined[0]

    print("analytical results")
    train_loss = MSE(W, b, train_x, train_y, args.reg)
    train_acc = get_acc(W, b, train_x, train_y)
    valid_acc = get_acc(W, b, valid_x, valid_y)
    valid_loss = MSE(W, b, valid_x, valid_y, args.reg)
    test_acc = get_acc(W, b, test_x, test_y)
    test_loss = MSE(W, b, test_x, test_y, args.reg)

    print("analytical results", " | train_loss:", train_loss, " train_acc:", train_acc, " valid_loss:", valid_loss,
      " valid_acc:", valid_acc, " test_loss:", test_loss, " test_acc:", test_acc)
    return W, b


def main(args):
    train_x, valid_x, test_x, train_y, valid_y, test_y = loadData()
    train_x = np.resize(train_x, (len(train_y), 28*28))
    valid_x = np.resize(valid_x, (len(valid_y), 28*28))
    test_x = np.resize(test_x, (len(test_y), 28*28))

    print('train data length: ', len(train_x))
    print('one train sample', train_y[0])
    print('train target length: ', len(train_y))
    print('one train target', train_y[0][0])

    W = np.random.rand(28*28, 1)
    b = np.random.rand(1)

    grad_start = time.time()
    grad_descent(W, b, train_x, train_y, args.lr, args.epochs, args.reg, args.error_tol, valid_x, valid_y, test_x, test_y)
    grad_end = time.time()
    get_analytical(train_x, train_y, valid_x, valid_y, test_x, test_y, args)
    anal_end = time.time()
    grad_time = grad_end - grad_start
    anal_time = anal_end - grad_end
    print('gradient descent computation time: ', grad_time)
    print('analytical solution computation time: ', anal_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--reg', type=float, default=0.1)
    parser.add_argument('--error_tol', type=float, default=10**(-7))
    parser.add_argument('--lossType', choices=['MSE', 'CE'], default='CE')

    args = parser.parse_args()

    main(args)