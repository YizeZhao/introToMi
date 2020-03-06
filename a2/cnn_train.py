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
            images, labels = images.cuda(), labels.cuda()
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




    evaluate = evaluate_ce
    criterion = torch.nn.CrossEntropyLoss()

    notMinstCNN = NotMinstClassifier(args.hidden_size, args.num_kernel).cuda()

    optimizer = torch.optim.Adam(notMinstCNN.parameters(), lr=args.lr)


    train_acc_record = []
    valid_acc_record = []
    test_acc_record = []
    train_loss_record = []
    valid_loss_record = []
    test_loss_record = []

   # since = time()

    for epoch in range(args.epochs):  # loop over the dataset multiple times

        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            #print("inputs, labels: ", inputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = notMinstCNN(inputs.float())
            if args.loss_type == 'ce' :
                loss = criterion(outputs.float(), torch.max(labels, 1)[1])
            elif args.loss_type == 'mse' :
                loss = criterion(outputs.float(), labels.float())
            loss.backward()
            optimizer.step()

            if (i+1)%args.eval_every == 0:
                valid_acc, valid_loss = evaluate(notMinstCNN, valid_loader, criterion)
                train_acc, train_loss = evaluate(notMinstCNN, train_loader, criterion)
                test_acc, test_loss = evaluate(notMinstCNN, test_loader, criterion)
                print("epoch:", epoch, "batch: ", i, " | train_loss:", train_loss.item(), " train_acc:", train_acc, " valid_loss:", valid_loss.item(),
                      " valid_acc:", valid_acc, " test_loss:", test_loss.item(), " test_acc:", test_acc)

                train_acc_record.append(train_acc)
                valid_acc_record.append(valid_acc)
                train_loss_record.append(train_loss)
                valid_loss_record.append(valid_loss)
                test_loss_record.append(test_loss)
                test_acc_record.append(test_acc)

    print('Finished Training')

    x = np.arange(args.epochs) + 1
    plt.subplot(211)
    plt.plot(x, train_loss_record, label = "train_loss")
    plt.plot(x, valid_loss_record, label="validation_loss")
    plt.plot(x, test_loss_record, label="test_loss")
    plt.xlabel('number of epochs')
    plt.ylabel('loss')
    plt.title('Loss VS epochs')
    plt.legend(loc = 'upper right')

    plt.subplot(212)
    plt.plot(x, train_acc_record, label = "train_accuracy")
    plt.plot(x, valid_acc_record, label = "validation_accuracy")
    plt.plot(x, test_acc_record, label="test_accuracy")
    plt.xlabel('number of epochs')
    plt.ylabel('accuracy')
    plt.title('Accuracy VS epochs')
    plt.legend(loc = 'lower right')
    plt.show()




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--lr', type=float, default = 0.0001)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--loss_type', choices=['mse', 'ce'], default='ce')
    parser.add_argument('--hidden_size', type=int, default=784)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_kernel', type=int, default=27)
    parser.add_argument('--eval_every', type=int, default=10)

    args = parser.parse_args()

    main(args)
