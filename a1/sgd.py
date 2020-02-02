import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math


def get_acc_two(pred, labels):

    if np.shape(predictions)[0] != np.shape(labels)[0]:
        print("size mismatch")
        return False
    batch_size = np.shape(predictions)[0]
    correct = np.sum((predictions>=0.5)==labels
    return correct/batch_size


def buildGraph(args, train_size, valid_size, test_size, dim):
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

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        return W, b, train_pred, train_y, loss, optimizer


def sgd(W, b, train_x, train_y, args.lr, args.epochs, args.reg, args.error_tol, valid_x, valid_y, test_x, test_y):
    sample_n = np.shape(train_y)[0]

    with tf.Sessions(graph = sgd_graph):
        batch_number = math.floor(sample_n/args.batch_size)
        tf.global_variables_initializer().run()














