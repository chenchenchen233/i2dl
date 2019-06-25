import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        ##############################################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:                                        #
        # 1. This network takes in a square (same width and height), grayscale image as input                        #
        # 2. It ends with a linear layer that represents the keypoints                                               #
        # it's suggested that you make this last layer output 30 values, 2 for each of the 15 keypoint (x, y) pairs  #
        #                                                                                                            #
        # Note that among the layers to add, consider including:                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or      #
        # batch normalization) to avoid overfitting.                                                                 #
        ##############################################################################################################
        height=96
        stride_conv=1
        kernel_size=5
        num_filters=32
        channel_in=1
        weight_scale=0.001
        outdim=30
        p=((height-1)*stride_conv+kernel_size-height)//2
        self.conv=nn.Conv2d(channel_in,num_filters,kernel_size,stride_conv,padding=(p,p))
        self.conv.weight.data.mul_(weight_scale)
        stride_pool=2
        pool=2
        self.maxpool=nn.MaxPool2d(pool,stride_pool)
        h_out=(height-1)//stride_conv+1
        input_size = num_filters*((height-pool)//stride_pool+1)*((height-pool)//stride_pool+1)
        dropout=0.9
        self.drop=nn.Dropout(dropout)
        self.fc=nn.Linear(input_size,outdim)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################
        x=self.maxpool(F.relu(self.conv(x)))
        x=x.view(x.size(0),-1)
        x=self.fc(self.drop(x))
       
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
