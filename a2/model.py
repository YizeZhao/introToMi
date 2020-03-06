import torch.nn as nn
import torch.nn.functional as F
import torch


class NotMinstClassifier(nn.Module):

    def __init__(self, hidden_size, num_kernel, r_dropout):

        super(NotMinstClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_kernel = num_kernel
        self.conv1 = nn.Conv2d(1, num_kernel, 3)     #(channels, #kernel, kernel size)
        self.conv1_bn = nn.BatchNorm2d(num_kernel)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(num_kernel * 13 * 13, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.drops = nn.Dropout(p=r_dropout)



    def forward(self, x):
        x = torch.reshape(x, (x.size()[0], 1, x.size()[1], x.size()[2]))

        x = self.pool(self.conv1_bn(F.relu(self.conv1(x))))
        x = x.view(-1, self.num_kernel * 13 * 13)
        x = self.fc1(x)
        # x = self.drops(x)  --- uncomment if need dropouts
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x

