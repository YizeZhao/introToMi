# ===================================== IMPORT ============================================#
import argparse
import time
import math
import numpy as np


import torch

from torch.utils.data import DataLoader



from model import  NotMinstClassifier
from dataset import NMDataset

from starter import *

import matplotlib.pyplot as plt


def evaluate_ce(model, test_loader, criterion):
    correct = 0
    total = 0
    eval_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            else:
                images, labels = images, labels
            outputs = model(images.float())
            loss = criterion(outputs.float(), torch.max(labels, axis=1)[1])
            eval_loss += loss
            predicted_class = torch.argmax(outputs, axis=1)
            truth = torch.argmax(labels, axis=1)
            total += labels.size(0)
            correct += (predicted_class == truth).sum().item()

    return (100 * correct / total), (eval_loss/total)


def main(args):

    # =============================== Initialize arguments ================================
    if torch.cuda.is_available():
        print("Using cuda")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_x, valid_x, test_x, trainTarget, validTarget, testTarget = loadData()
    train_y, valid_y, test_y = convertOneHot(trainTarget, validTarget, testTarget)


    train_dataset = NMDataset(train_x, train_y)
    validate_dataset = NMDataset(valid_x, valid_y)
    test_dataset = NMDataset(test_x,test_y)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(validate_dataset, batch_size=6000, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2724, shuffle=False)


    train_acc_record = []
    valid_acc_record = []
    test_acc_record = []
    train_loss_record = []
    valid_loss_record = []
    test_loss_record = []

    evaluate = evaluate_ce
    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        notMinstCNN = NotMinstClassifier(args.hidden_size, args.num_kernel, args.r_drops).cuda()
    else:
        notMinstCNN = NotMinstClassifier(args.hidden_size, args.num_kernel, args.r_drops)

    optimizer = torch.optim.Adam(notMinstCNN.parameters(), lr=args.lr)

    for epoch in range(args.epochs):  # loop over the dataset multiple times

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = notMinstCNN(inputs.float())
            loss = criterion(outputs.float(), torch.max(labels, 1)[1])

            # UNCOMMENT FOR REGULARIZATION
            # l2 = 0
            # for name, p in notMinstCNN.fc1.named_parameters():
            #     if 'weight' in name:
            #         l2 = l2 + (p**2).sum()
            # for name, p in notMinstCNN.fc2.named_parameters():
            #     if 'weight' in name:
            #         l2 = l2 + (p**2).sum()
            #
            # loss = loss + args.r_lambda * l2/labels.shape[0]
            loss.backward()
            optimizer.step()

            if (i)%args.eval_every == 0:
                valid_acc, valid_loss = evaluate(notMinstCNN, valid_loader, criterion)
                train_acc, train_loss = evaluate(notMinstCNN, train_loader, criterion)
                test_acc, test_loss = evaluate(notMinstCNN, test_loader, criterion)
                print("epoch:", epoch, "batch: ", i, " | train_loss:", train_loss.item(), " train_acc:", \
                      train_acc, " valid_loss:", valid_loss.item(),
                      " valid_acc:", valid_acc, " test_loss:", test_loss.item(), " test_acc:", test_acc)

                train_acc_record.append(train_acc)
                valid_acc_record.append(valid_acc)
                train_loss_record.append(train_loss)
                valid_loss_record.append(valid_loss)
                test_loss_record.append(test_loss)
                test_acc_record.append(test_acc)


    print('Finished Training')

    x = (np.arange(len(train_loss_record)) + 1)*args.eval_every
    plt.figure(figsize=(12,7))
    plt.plot(x, train_loss_record, label = "train_loss")
    plt.plot(x, valid_loss_record, label="validation_loss")
    plt.plot(x, test_loss_record, label="test_loss")
    plt.xlabel('number of steps')
    plt.ylabel('loss')
    plt.title('Loss VS SGD steps')
    plt.legend(loc = 'upper right')
    plt.savefig('/content/introToMi/a2/loss_curve.pdf')
    plt.savefig('/content/drive/My Drive/ece421_plots/loss_curve_lambda1.pdf')

    plt.show()

    plt.figure(figsize=(12,7))
    plt.plot(x, train_acc_record, label = "train_accuracy")
    plt.plot(x, valid_acc_record, label = "validation_accuracy")
    plt.plot(x, test_acc_record, label="test_accuracy")
    plt.xlabel('number of steps')
    plt.ylabel('accuracy')
    plt.title('Accuracy VS SGD steps')
    plt.legend(loc = 'lower right')
    plt.savefig('/content/introToMi/a2/acc_curve.pdf')
    plt.savefig('/content/drive/My Drive/ece421_plots/acc_curve_lambda1.pdf')

    
    plt.show()




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--lr', type=float, default = 0.0001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--loss_type', choices=['mse', 'ce'], default='ce')
    parser.add_argument('--hidden_size', type=int, default=784)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_kernel', type=int, default=27)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--r_lambda', type=float, default=0.01)
    parser.add_argument('--r_drops', type=float, default=0.9)


    args = parser.parse_args()

    main(args)
