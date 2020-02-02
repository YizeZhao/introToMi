import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from starter import *


def get_acc_two(pred, labels):

    if np.shape(pred)[0] != np.shape(labels)[0]:
        print("size mismatch")
        return False
    batch_size = np.shape(pred)[0]
    correct = (np.sum((pred>=0.5)==labels))
    return correct/batch_size


def buildGraph(args, train_size, dim):
    # Initialize weight and bias tensors
    seed = tf.set_random_seed(421)
    sgd_graph = tf.graph()
    with sgd_graph.as_defalt():

        # specify datasizes
        train_x = tf.placeholder(shape=(train_size, dim), dtype=tf.float32)
        train_y = tf.placeholder(shape=(train_size, 1), dtype=tf.float32)

        #valid_x = tf.placeholder(shape=(valid_size, dim), dtype=tf.float32)
        #valid_y = tf.placeholder(shape=(valid_size, 1), dtype=tf.float32)

        #test_x = tf.placeholder(shape=(test_size, dim), dtype=tf.float32)
        # test_y = tf.placeholder(shape=(test_size, 1), dtype=tf.float32)

        #reg = tf.placeholder(shape=(1), dtype=tf.float32)

        W = tf.Variable(
            tf.truncated_normal(shape=(dim, 1), mean=0.0, stddev=0.5, dtype=tf.float32))
        b = tf.Variable(tf.random_uniform(shape=(1), minval=0, maxval=1, dtype=tf.float32))

        train_pred = tf.matmul(train_x,W) + b
        # valid_pred = tf.matmul(valid_x,W) + b
        # test_pred = tf.matmul(test_x,W) + b
        


        if args.losstype == "MSE":
            train_loss = tf.losses.mean_squared_error(train_y, train_pred) + args.reg * tf.nn.l2_loss(W)  # l2_loss Computes half the L2 norm of a tensor
            #valid_loss = tf.losses.mean_squared_error(valid_y, valid_pred) + args.reg * tf.nn.l2_loss(W)
            #test_loss = tf.losses.mean_squared_error(test_y, test_pred) + args.reg * tf.nn.l2_loss(W)
        elif args.losstype == "CE":
            train_loss = tf.losses.sigmoid_cross_entropy(train_y, train_pred) + args.reg * tf.nn.l2_loss(W)
            #valid_loss = tf.losses.sigmoid_cross_entropy(valid_y, valid_pred) + args.reg * tf.nn.l2_loss(W)
            #loss = tf.losses.sigmoid_cross_entropy(test_y, test_pred) + args.reg * tf.nn.l2_loss(W)
        else:
            print("undefined loss type")

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(train_loss)

        return W, b, train_pred, train_y, train_loss, optimizer


def get_rand_permutation(train_size, valid_size, test_size):
    train_index = np.arange(train_size)
    train_shuffled = np.random.shuffle(train_index)

    valid_index = np.arange(valid_size)
    valid_shuffled = np.random.shuffle(valid_index)

    test_index = np.arange(test_size)
    test_shuffled = np.random.shuffle(test_index)

    return train_shuffled, valid_shuffled, test_shuffled


def sgd(args):
    train_x, valid_x, test_x, train_y, valid_y, test_y = loadData()

    train_x = np.resize(train_x, (len(train_y), 28*28))
    valid_x = np.resize(valid_x, (len(valid_y), 28*28))
    test_x = np.resize(test_x, (len(test_y), 28*28))

    train_size = np.shape(train_y)[0]
    valid_size = np.shape(valid_y)[0]
    test_size = np.shape(test_y)[0]

    dim = (np.shape(train_x)[0][0])**2
    W, b, train_pred, train_y, train_loss, optimizer = buildGraph(args, train_size, dim)

    n_batches = math.floor((train_y.shape[0]/args.batch_size))
    with tf.Sessions() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(args.epochs):

            train_shuffled, valid_shuffled, test_shuffled = get_rand_permutation(train_size, valid_size, test_size)
            train_x = train_x[train_shuffled]
            train_y = train_y[train_shuffled]

            valid_x = valid_x[valid_shuffled]
            valid_y = valid_y[valid_shuffled]

            test_x = test_x[test_shuffled]
            test_y = test_y[test_shuffled]

            for j in range(n_batches):

                batch_x = train_x[i * args.batch_size : (i + 1) * args.batch_size]
                batch_y = train_x[i * args.batch_size : (i + 1) * args.batch_size]


                _W, _b, _train_pred, _train_y, _train_loss, _optimizer = sess.run([ W, b, train_pred, train_y, train_loss, optimizer], feed_dict = {
                    train_x: batch_x,
                    train_y: batch_y})

                


















