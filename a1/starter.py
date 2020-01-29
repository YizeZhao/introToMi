import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

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
    l_d_1 = np.dot(x, W) + b
    l_d_2 = l_d_1 ** 2
    l_w_2 = W ** 2
    l = np.mean((y - l_d_1)**2) + reg/2 * sum(l_w_2)

    return l



def gradMSE(W, b, x, y, reg):
    # compute gradients
    # x: 3500*784
    # W: 784*1
    sample_len = len(y)
    l_d_1 = np.dot(x, W) + b - y
    l_d_1 = np.transpose(l_d_1)
    dl_dw = (2 / sample_len) * np.dot(l_d_1,x) + np.transpose(reg * W)
    dl_db = (2 / sample_len) * np.sum(l_d_1,axis=1)
    return dl_dw, dl_db




def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, valid_x, valid_y, test_x, test_y):
    # trainning loop
    train_loss_rec = []
    train_acc_rec = []
    valid_loss_rec = []
    valid_acc_rec = []
    test_loss_rec = []
    test_acc_rec = []

    for epoch in range(epochs):

        dl_dw, dl_db = gradMSE(W, b, x, y, reg)
        W = W - np.transpose(alpha * dl_dw)
        b = b - alpha * dl_db

        if (epoch+1) % 10 == 0:
            train_loss = MSE(W, b, x, y, reg)
            train_acc = get_acc(W, b, x, y)
            valid_acc = get_acc(W, b, valid_x, valid_y)
            valid_loss = MSE(W, b, valid_x,  valid_y, reg)
            test_acc = get_acc(W, b, test_x, test_y)
            test_loss = MSE(W, b, test_x, test_y, reg)

            train_loss_rec.append(train_loss)
            train_acc_rec.append(train_acc)
            valid_loss_rec.append(valid_loss)
            valid_acc_rec.append(valid_acc)
            test_loss_rec.append(test_loss)
            test_acc_rec.append(test_acc)


            print("epoch:", epoch, " | train_loss:", train_loss, " train_acc:", train_acc, " valid_loss:", valid_loss, " valid_acc:", valid_acc, " test_loss:", test_loss, " test_acc:", test_acc)

    plot_loss_acc(train_loss_rec, train_acc_rec, valid_loss_rec, valid_acc_rec, test_loss_rec, test_acc_rec)

def get_acc(W, b, x, y):
    y_pred = np.dot(x, W) + b
    corr = 0
    for sample in range(len(y)):
        if (y_pred[sample] > 0.5 and y[sample] == 1) or (y_pred[sample] < 0.5 and y[sample] == 0):
            corr = corr + 1
    return corr/len(y)

# def crossEntropyLoss(W, b, x, y, reg):
#
#
# # Your implementation here
#
# def gradCE(W, b, x, y, reg):
#
#
# # Your implementation here
#


# def buildGraph(loss="MSE"):
#     # Initialize weight and bias tensors
#     tf.set_random_seed(421)
#
#     if loss == "MSE":
#     # Your implementation
#
#     elif loss == "CE":
# # Your implementation here


def plot_loss_acc(train_loss_rec, train_acc_rec, valid_loss_rec, valid_acc_rec, test_loss_rec, test_acc_rec):

    x = np.arange(len(train_loss_rec)) + 1
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


def main(args):

    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = np.resize(trainData, (len(trainTarget), 28*28))
    validData = np.resize(validData, (len(validTarget), 28*28))
    testData = np.resize(testData, (len(testTarget), 28*28))

    print('train data length: ', len(trainData))
    print('one train sample', trainData[0])
    print('train target length: ', len(trainTarget))
    print('one train target', trainTarget[0][0])

    W = np.random.rand(28*28, 1)
    b = np.random.rand(1)


    grad_descent(W, b, trainData, trainTarget, args.lr, args.epochs, args.reg, args.error_tol, validData, validTarget, testData, testTarget)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--reg', type=int, default=0.5)
    parser.add_argument('--error_tol', type=int, default=0.2)


    args = parser.parse_args()

    main(args)