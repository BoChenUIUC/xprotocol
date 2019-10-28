import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

class CNN_based_Model(nn.Module):
    def __init__(self,no_of_hidden_units,out_size,weights=[0.3,1],size_list=[10,5,5,1],no_of_filters=16,filter_size=4):
        super(CNN_based_Model, self).__init__()

        self.size_list = size_list

        self.probe_conv = nn.Conv1d(1,no_of_filters,filter_size) # 16*7
        self.probe_bn = nn.BatchNorm1d(no_of_filters)

        self.frame_conv = nn.Conv1d(1,no_of_filters,filter_size) # 16*2
        self.frame_bn = nn.BatchNorm1d(no_of_filters)

        self.size_conv = nn.Conv1d(1,no_of_filters,filter_size) # 16*2
        self.size_bn = nn.BatchNorm1d(no_of_filters)

        self.fc1 = nn.Linear(177, no_of_hidden_units)

        self.fc2 = nn.Linear(no_of_hidden_units, no_of_hidden_units)


        self.fc_output = nn.Linear(no_of_hidden_units, out_size)

        class_weights = torch.FloatTensor(weights).cuda()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self,x,label):
        x_probe,x_frame,x_size,x_cursize = torch.split(torch.unsqueeze(x,1),self.size_list,dim=2) 

        x_probe = self.probe_bn(F.relu(self.probe_conv(x_probe)))
        x_frame = self.frame_bn(F.relu(self.frame_conv(x_frame)))
        x_size = self.size_bn(F.relu(self.size_conv(x_size)))

        x_probe = x_probe.view(-1,16*7)
        x_frame = x_frame.view(-1,16*2)
        x_size = x_size.view(-1,16*2)
        x_cursize = x_cursize.view(-1,1)

        x = torch.cat((x_probe,x_frame,x_size,x_cursize),dim=1)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc_output(x)

        _, pred = torch.max(x.data, 1)

        return self.loss(x,label), pred