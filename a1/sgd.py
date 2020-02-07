import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from starter import *
import argparse

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def get_acc_two(pred, labels):

    if np.shape(pred)[0] != np.shape(labels)[0]:
        print("size mismatch")
        return False
    batch_size = np.shape(pred)[0]
    correct = (np.sum((pred>=0.5)==labels))
    return correct/batch_size


def buildGraph(args, valid_size, test_size, dim):
    # Initialize weight and bias tensors
    tf.set_random_seed(421)

    # specify datasizes
    X = tf.placeholder(shape=(args.batch_size, dim), dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=(args.batch_size, 1), dtype=tf.float32, name='Y')

    valid_x = tf.placeholder(shape=(valid_size, dim), dtype=tf.float32)
    valid_y = tf.placeholder(shape=(valid_size, 1), dtype=tf.float32)

    test_x = tf.placeholder(shape=(test_size, dim), dtype=tf.float32)
    test_y = tf.placeholder(shape=(test_size, 1), dtype=tf.float32)

    #reg = tf.placeholder(shape=(1), dtype=tf.float32)

    W = tf.Variable(
        tf.truncated_normal(shape=(dim, 1), mean=0.0, stddev=0.5, dtype=tf.float32))
    b = tf.Variable(tf.random_uniform(shape=(), minval=0, maxval=1, dtype=tf.float32))

    train_pred = tf.matmul(X,W) + b
    valid_pred = tf.matmul(valid_x,W) + b
    test_pred = tf.matmul(test_x,W) + b
        


    if args.lossType == "MSE":
        train_loss = tf.losses.mean_squared_error(Y, train_pred) + args.reg * tf.nn.l2_loss(W)  # l2_loss Computes half the L2 norm of a tensor
        valid_loss = tf.losses.mean_squared_error(valid_y, valid_pred) + args.reg * tf.nn.l2_loss(W)
        test_loss = tf.losses.mean_squared_error(test_y, test_pred) + args.reg * tf.nn.l2_loss(W)
    elif args.lossType == "CE":
        train_loss = tf.losses.sigmoid_cross_entropy(Y, train_pred) + args.reg * tf.nn.l2_loss(W)
        valid_loss = tf.losses.sigmoid_cross_entropy(valid_y, valid_pred) + args.reg * tf.nn.l2_loss(W)
        test_loss = tf.losses.sigmoid_cross_entropy(test_y, test_pred) + args.reg * tf.nn.l2_loss(W)
    else:
        print("undefined loss type")

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999).minimize(train_loss)

    return W, b, X, Y, valid_x, valid_y, test_x, test_y, train_pred, valid_pred, test_pred, train_loss, valid_loss, test_loss, optimizer



def get_rand_permutation(train_size):
    train_index = np.arange(train_size)
    train_shuffled = np.random.shuffle(train_index)

    #valid_index = np.arange(valid_size)
    #valid_shuffled = np.random.shuffle(valid_index)

    #test_index = np.arange(test_size)
    #test_shuffled = np.random.shuffle(test_index)

    return train_index


def sgd(args):

    break_out = 0
    break_in = 0

    train_x, valid_x, test_x, train_y, valid_y, test_y = loadData()

    train_x = np.resize(train_x, (len(train_y), 28*28))
    valid_x = np.resize(valid_x, (len(valid_y), 28*28))
    test_x = np.resize(test_x, (len(test_y), 28*28))

    train_size = np.shape(train_y)[0]
    valid_size = np.shape(valid_y)[0]
    test_size = np.shape(test_y)[0]

    dim = (np.shape(train_x)[1])
    W_, b_, X_, Y_, valid_x_, valid_y_, test_x_, test_y_, train_pred_, valid_pred_, test_pred_, train_loss_, \
    valid_loss_, test_loss_, optimizer_ \
        = buildGraph(args, valid_size, test_size, dim)

    n_batches = math.floor((int(train_y.shape[0])/args.batch_size))

    train_loss_rec = []
    train_acc_rec = []
    valid_loss_rec = []
    valid_acc_rec = []
    test_loss_rec = []
    test_acc_rec = []

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(args.epochs):

            train_shuffled = get_rand_permutation(train_size)
            train_x = train_x[train_shuffled]
            train_y = train_y[train_shuffled]


            for j in range(n_batches):

                batch_x = train_x[j * args.batch_size : (j + 1) * args.batch_size]
                batch_y = train_y[j * args.batch_size : (j + 1) * args.batch_size]



                _W, _b, _Y, _valid_y, _test_y, _train_pred, _valid_pred, _test_pred, _train_loss, _valid_loss, _test_loss, _optimizer = sess.run([ W_, b_, Y_, valid_x_, test_y_, train_pred_, valid_pred_, test_pred_, train_loss_, valid_loss_, test_loss_, optimizer_], feed_dict = {
                    X_: batch_x,
                    Y_: batch_y,
                    valid_x_: valid_x,
                    valid_y_: valid_y,
                    test_x_: test_x,
                    test_y_: test_y})




                train_acc = get_acc_two(_train_pred, _Y)
                valid_acc = get_acc_two(_valid_pred, valid_y)
                test_acc = get_acc_two(_test_pred, test_y)


                if (i+1)%10 and (j+1)%10:

                    print("epoch:", i, "batch:", j, " | train_loss:", _train_loss, " | train_acc:", train_acc, " | valid_loss:", _valid_loss, " | train_acc:", valid_acc, " | test_loss:", _test_loss, " | test_acc:", test_acc)
                    train_loss_rec.append(_train_loss)
                    train_acc_rec.append(train_acc)
                    valid_loss_rec.append(_valid_loss)
                    valid_acc_rec.append(valid_acc)
                    test_loss_rec.append(_test_loss)
                    test_acc_rec.append(test_acc)

    plot_loss_acc(train_loss_rec, train_acc_rec, valid_loss_rec, valid_acc_rec, test_loss_rec, test_acc_rec)



def main(args):

    sgd(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--reg', type=int, default=0.5)
    parser.add_argument('--error_tol', type=int, default=0.2)
    parser.add_argument('--lossType', choices=['MSE', 'CE'], default='CE')
    parser.add_argument('--beta-1', choices=['MSE', 'CE'], default='CE')
    parser.add_argument('--beta-2', choices=['MSE', 'CE'], default='CE')


    args = parser.parse_args()

    main(args)















