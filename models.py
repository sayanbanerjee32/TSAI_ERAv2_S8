import torch
import torch.nn as nn
import torch.nn.functional as F

class NetS6(nn.Module):
    def __init__(self,n = 8):
        super(Net, self).__init__()
        # Convolusion Block - Inital
        #input size: 28 x 28 x 1, output size: 28 x 28 x n, receptive field: 3
        self.convi = nn.Conv2d(1, n, 3, padding=1, bias = False)
        self.bni = nn.BatchNorm2d(n)

        # Convolusion Block - feature extraction
        #input size: 28 x 28 x n, output size: 28 x 28 x n*2, receptive field: 3 + (3-1) * 1 = 5
        self.conv1 = nn.Conv2d(n, n*2, 3, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(n*2)
        #input size: 28 x 28 x n*2, output size: 28 x 28 x n*4, receptive field: 5 + (3-1) * 1 = 7
        self.conv2 = nn.Conv2d(n*2, n*4, 3, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(n*4)

        # Transition block - reduction in channel size and number
        #input size: 28 x 28 x n*4, output size: 14 x 14 x n*4, receptive field: 7 + (3-1) * 1 = 8
        self.pool1 = nn.MaxPool2d(2, 2)
        #input size: 14 x 14 x n*4, output size: 14 x 14 x n, receptive field: 8 + (1-1)*2 = 8
        self.antman1 = nn.Conv2d(n*4, n, kernel_size=1,bias = False)
        self.bna1 = nn.BatchNorm2d(n)

        # Convolusion Block - feature extraction
        #input size: 14 x 14 x n, output size: 14 x 14 x n*2, receptive field: 8 + (3-1) * 2 = 12
        self.conv3 = nn.Conv2d(n, n*2, 3, padding=1, bias = False)
        self.bn3 = nn.BatchNorm2d(n*2)
        #input size: 14 x 14 x n*2, output size: 14 x 14 x n*4, receptive field: 12 + (3-1) * 2 = 16
        self.conv4 = nn.Conv2d(n*2, n*4, 3, padding=1, bias = False)
        self.bn4 = nn.BatchNorm2d(n*4)

        # Transition block - reduction in channel size and number
        #input size: 14 x 14 x n*4, output size: 7 x 7 x n*4, receptive field: 16 + (2-1) * 2 =18
        self.pool2 = nn.MaxPool2d(2, 2)
        #input size: 7 x 7 x n*4, output size: 7 x 7 x n, receptive field: 18 + (1-1) * 4 =18
        self.antman2 = nn.Conv2d(n*4, n, kernel_size=1, bias = False)
        self.bna2 = nn.BatchNorm2d(n)

        # Convolusion Block - feature extraction
        #input size: 7 x 7 x n, output size: 7 x 7 x n*2, receptive field: 18 + (3-1) * 4 = 26
        self.conv5 = nn.Conv2d(n, n*2, 3, padding=1, bias = False)
        self.bn5 = nn.BatchNorm2d(n*2)
        #input size: 7 x 7 x n*2, output size: 7 x 7 x n*4, receptive field: 26 + (3-1) * 4 = 34
        self.conv6 = nn.Conv2d(n*2, n*4, 3, padding=1, bias = False)
        self.bn6 = nn.BatchNorm2d(n*4)

        # Transition block - - reduction in channel size
        # and aligning number of channels to number of prediction classes
        #input size: 7 x 7 x n*4, output size: 3 x 3 x n*4, receptive field: 34 + (2-1) * 4 = 38
        self.pool3 = nn.MaxPool2d(2, 2)
        #input size: 3 x 3 x n*4, output size: 3 x 3 x 10, receptive field: 38 + (1-1) * 8 = 38
        self.antman3 = nn.Conv2d(n*4, 10, kernel_size=1, bias = False)

    def forward(self, x):
        x = self.bni(F.relu(self.convi(x))) #28 x 28

        x = self.bn1(F.relu(self.conv1(x))) #28 x 28
        x = self.bn2(F.relu(self.conv2(x))) #28 x 28
        x = self.bna1(self.antman1(self.pool1(x))) #14 x 14
        # dropout of 0.25 was not allowing model to train to the required level
        # thus dropout is set to 0.1
        x = F.dropout(x, 0.10) #14 x 14

        x = self.bn3(F.relu(self.conv3(x))) #14 x 14
        x = self.bn4(F.relu(self.conv4(x))) #14 x 14
        x = self.bna2(self.antman2(self.pool2(x))) #7 x 7
        # dropout of 0.25 was not allowing model to train to the required level
        # thus dropout is set to 0.1
        x = F.dropout(x, 0.10) #7 x 7

        x = self.bn5(F.relu(self.conv5(x))) #7 x 7
        x = self.bn6(F.relu(self.conv6(x))) #7 x 7
        x = self.antman3(self.pool3(x)) #3 x 3

        # Global average pooling instead of FC layer
        #input size: 3 x 3 x 10, output size: 1 x 1 x 10 > len of 10
        x = F.avg_pool2d(x, kernel_size = 3).squeeze()

        # x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        # x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        # x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # x = F.relu(self.conv7(x))
        # x = x.view(-1, 10)
        return F.log_softmax(x)

class NetS7(nn.Module):
    def __init__(self, n = 8, dropout_value = 0.1):
        super(Net, self).__init__()
        # # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n),
            nn.Dropout(dropout_value)
        ) #input size: 28 x 28 x 1, output size: 26 x 26 x n, receptive field: 1 + (3-1) * 1 = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=n*2,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n*2),
            nn.Dropout(dropout_value)
        ) #input size: 26 x 26 x n, output size: 24 x 24 x n*2, receptive field: 3 + (3-1) * 1 = 5

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) #input size: 24 x 24 x n*2, output size: 12 x 12 x n*2, receptive field: 5 + (2-1) * 1 = 6
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=n,
                      kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n),
            nn.Dropout(dropout_value)
        ) #input size: 12 x 12 x n*2, output size: 12 x 12 x n, receptive field: 6 + (1-1)*2 = 6

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=n*2,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n*2),
            nn.Dropout(dropout_value)
        ) #input size: 12 x 12 x n, output size: 10 x 10 x n*2, receptive field: 6 + (3-1) * 2 = 10

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=n*2,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n*2),
            nn.Dropout(dropout_value)
        ) #input size: 10 x 10 x n*2, output size: 8 x 8 x n*2, receptive field: 10 + (3-1) * 2 = 14

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=n*2,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n*2),
            nn.Dropout(dropout_value)
        ) #input size: 8 x 8 x n*2, output size: 6 x 6 x n*2, receptive field: 14 + (3-1) * 2 = 18

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            # nn.AvgPool2d(kernel_size=7) # 7>> 9...
            nn.AdaptiveAvgPool2d((1, 1))
        ) #input size: 6 x 6 x n*2, output size: 1 x 1 x n*2, receptive field: 18

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=10,
                      kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU()
        ) #input size: 1 x 1 x n*2, output size: 1 x 1 x 10, receptive field: 18 + (1-1) * 2 =18



    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)

        x = self.pool1(x)
        x = self.convblock3(x)

        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)

        x = self.gap(x)
        x = self.convblock7(x)

        x = x.squeeze()

        return F.log_softmax(x, dim=-1)

class NetS8BN(nn.Module):
    def __init__(self, n = 8, dropout_value = 0.1):
        super(Net, self).__init__()
        # # Input Block
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n*2,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n*2),
            nn.Dropout(dropout_value)
        ) #input size: 32 x 32 x 3, output size: 32 x 32 x n*2, receptive field: 1 + (3-1) * 1 = 3

        # CONVOLUTION BLOCK 1
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=n*4,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n*4),
            nn.Dropout(dropout_value)
        ) #input size: 32 x 32 x n*2, output size: 24 x 24 x n*4, receptive field: 3 + (3-1) * 1 = 5

        # TRANSITION BLOCK 1
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=n*4, out_channels=n,
                      kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n),
            nn.Dropout(dropout_value)
        ) #input size: 32 x 32 x n*4, output size: 32 x 32 x n, receptive field: 6 + (1-1)*2 = 6
        self.p1 = nn.MaxPool2d(2, 2) #input size: 32 x 32 x n, output size: 16 x 16 x n, receptive field: 5 + (2-1) * 1 = 6

        # CONVOLUTION BLOCK 2
        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=n*2,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n*2),
            nn.Dropout(dropout_value)
        ) #input size: 16 x 16 x n, output size: 16 x 16 x n*2, receptive field: 6 + (3-1) * 2 = 10

        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=n*4,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n*4),
            nn.Dropout(dropout_value)
        ) #input size: 16 x 16 x n*2, output size: 16 x 16 x n*4, receptive field: 10 + (3-1) * 2 = 14

        self.C6 = nn.Sequential(
            nn.Conv2d(in_channels=n*4, out_channels=n*4,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n*4),
            nn.Dropout(dropout_value)
        ) #input size: 16 x 16 x n*4, output size: 16 x 16 x n*4, receptive field: 14 + (3-1) * 2 = 18

        # TRANSITION BLOCK 2
        self.c7 = nn.Sequential(
            nn.Conv2d(in_channels=n*4, out_channels=n,
                      kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n),
            nn.Dropout(dropout_value)
        ) #input size: 16 x 16 x n*4, output size: 16 x 16 x n, receptive field: 18 + (1-1)*2 = 18
        self.p2 = nn.MaxPool2d(2, 2) #input size: 16 x 16 x n, output size: 8 x 8 x n, receptive field: 18 + (2-1) * 2 = 20


        # CONVOLUTION BLOCK 3
        self.C8 = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=n*2,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n*2),
            nn.Dropout(dropout_value)
        ) #input size: 8 x 8 x n, output size: 8 x 8 x n*2, receptive field: 20 + (3-1) * 4 = 28

        self.C9 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=n*4,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n*4),
            nn.Dropout(dropout_value)
        ) #input size: 8 x 8 x n*2, output size: 8 x 8 x n*4, receptive field: 28 + (3-1) * 4 = 36

        self.C10 = nn.Sequential(
            nn.Conv2d(in_channels=n*4, out_channels=n*4,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(n*4),
            nn.Dropout(dropout_value)
        ) #input size: 8 x 8 x n*4, output size: 8 x 8 x n*4, receptive field: 36 + (3-1) * 4 = 44

        # OUTPUT BLOCK
        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1))
        ) #input size: 8 x 8 x n*4, output size: 1 x 1 x n*4, receptive field: 44

        self.c11 = nn.Sequential(
            nn.Conv2d(in_channels=n*4, out_channels=10,
                      kernel_size=(1, 1), padding=0, bias=False),
        ) #input size: 1 x 1 x n*4, output size: 1 x 1 x 10, receptive field: 44 + (1-1) * 4 =44



    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)

        x = self.c3(x)
        x = self.p1(x)

        x = self.C4(x)
        x = self.C5(x)
        x = self.C6(x)

        x = self.c7(x)
        x = self.p2(x)

        x = self.C8(x)
        x = self.C9(x)
        x = self.C10(x)

        x = self.GAP(x)
        x = self.c11(x)

        x = x.squeeze()

        return F.log_softmax(x, dim=-1)

class NetS8GN(nn.Module):
    def __init__(self, n = 8, dropout_value = 0.05):
        super(Net, self).__init__()
        # # Input Block
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n*2,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(2, n*2),
            nn.Dropout(dropout_value)
        )  #input size: 32 x 32 x 3, output size: 32 x 32 x n*2, receptive field: 1 + (3-1) * 1 = 3

        # CONVOLUTION BLOCK 1
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=n*4,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, n*4),
            nn.Dropout(dropout_value)
        ) #input size: 32 x 32 x n*2, output size: 24 x 24 x n*4, receptive field: 3 + (3-1) * 1 = 5

        # TRANSITION BLOCK 1
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=n*4, out_channels=n,
                      kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, n),
            nn.Dropout(dropout_value)
        ) #input size: 32 x 32 x n*4, output size: 32 x 32 x n, receptive field: 6 + (1-1)*2 = 6
        self.p1 = nn.MaxPool2d(2, 2) #input size: 32 x 32 x n, output size: 16 x 16 x n, receptive field: 5 + (2-1) * 1 = 6

        # CONVOLUTION BLOCK 2
        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=n*2,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(2, n*2),
            nn.Dropout(dropout_value)
        ) #input size: 16 x 16 x n, output size: 16 x 16 x n*2, receptive field: 6 + (3-1) * 2 = 10

        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=n*4,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, n*4),
            nn.Dropout(dropout_value)
        ) #input size: 16 x 16 x n*2, output size: 16 x 16 x n*4, receptive field: 10 + (3-1) * 2 = 14

        self.C6 = nn.Sequential(
            nn.Conv2d(in_channels=n*4, out_channels=n*4,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, n*4),
            nn.Dropout(dropout_value)
        ) #input size: 16 x 16 x n*4, output size: 16 x 16 x n*4, receptive field: 14 + (3-1) * 2 = 18

        # TRANSITION BLOCK 2
        self.c7 = nn.Sequential(
            nn.Conv2d(in_channels=n*4, out_channels=n,
                      kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, n),
            nn.Dropout(dropout_value)
        ) #input size: 16 x 16 x n*4, output size: 16 x 16 x n, receptive field: 18 + (1-1)*2 = 18
        self.p2 = nn.MaxPool2d(2, 2) #input size: 16 x 16 x n, output size: 8 x 8 x n, receptive field: 18 + (2-1) * 2 = 20


        # CONVOLUTION BLOCK 3
        self.C8 = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=n*2,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(2, n*2),
            nn.Dropout(dropout_value)
        ) #input size: 8 x 8 x n, output size: 8 x 8 x n*2, receptive field: 20 + (3-1) * 4 = 28

        self.C9 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=n*4,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, n*4),
            nn.Dropout(dropout_value)
        ) #input size: 8 x 8 x n*2, output size: 8 x 8 x n*4, receptive field: 28 + (3-1) * 4 = 36

        self.C10 = nn.Sequential(
            nn.Conv2d(in_channels=n*4, out_channels=n*4,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, n*4),
            nn.Dropout(dropout_value)
        ) #input size: 8 x 8 x n*4, output size: 8 x 8 x n*4, receptive field: 36 + (3-1) * 4 = 44

        # OUTPUT BLOCK
        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1))
        ) #input size: 8 x 8 x n*4, output size: 1 x 1 x n*4, receptive field: 44

        self.c11 = nn.Sequential(
            nn.Conv2d(in_channels=n*4, out_channels=10,
                      kernel_size=(1, 1), padding=0, bias=False),
        ) #input size: 1 x 1 x n*4, output size: 1 x 1 x 10, receptive field: 44 + (1-1) * 4 =44



    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)

        x = self.c3(x)
        x = self.p1(x)

        x = self.C4(x)
        x = self.C5(x)
        x = self.C6(x)

        x = self.c7(x)
        x = self.p2(x)

        x = self.C8(x)
        x = self.C9(x)
        x = self.C10(x)

        x = self.GAP(x)
        x = self.c11(x)

        x = x.squeeze()

        return F.log_softmax(x, dim=-1)

class NetS8LN(nn.Module):
    def __init__(self, n = 8, dropout_value = 0.05):
        super(Net, self).__init__()
        # # Input Block
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n*2,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm([32,32]),
            nn.Dropout(dropout_value)
        ) #input size: 32 x 32 x 3, output size: 32 x 32 x n*2, receptive field: 1 + (3-1) * 1 = 3

        # CONVOLUTION BLOCK 1
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=n*4,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm([32,32]),
            nn.Dropout(dropout_value)
        ) #input size: 32 x 32 x n*2, output size: 24 x 24 x n*4, receptive field: 3 + (3-1) * 1 = 5

        # TRANSITION BLOCK 1
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=n*4, out_channels=n,
                      kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm([32,32]),
            nn.Dropout(dropout_value)
        ) #input size: 32 x 32 x n*4, output size: 32 x 32 x n, receptive field: 6 + (1-1)*2 = 6
        self.p1 = nn.MaxPool2d(2, 2) #input size: 32 x 32 x n, output size: 16 x 16 x n, receptive field: 5 + (2-1) * 1 = 6

        # CONVOLUTION BLOCK 2
        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=n*2,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm([16,16]),
            nn.Dropout(dropout_value)
        ) #input size: 16 x 16 x n, output size: 16 x 16 x n*2, receptive field: 6 + (3-1) * 2 = 10

        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=n*4,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm([16,16]),
            nn.Dropout(dropout_value)
        ) #input size: 16 x 16 x n*2, output size: 16 x 16 x n*4, receptive field: 10 + (3-1) * 2 = 14

        self.C6 = nn.Sequential(
            nn.Conv2d(in_channels=n*4, out_channels=n*4,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm([16,16]),
            nn.Dropout(dropout_value)
        ) #input size: 16 x 16 x n*4, output size: 16 x 16 x n*4, receptive field: 14 + (3-1) * 2 = 18

        # TRANSITION BLOCK 2
        self.c7 = nn.Sequential(
            nn.Conv2d(in_channels=n*4, out_channels=n,
                      kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm([16,16]),
            nn.Dropout(dropout_value)
        ) #input size: 16 x 16 x n*4, output size: 16 x 16 x n, receptive field: 18 + (1-1)*2 = 18
        self.p2 = nn.MaxPool2d(2, 2) #input size: 16 x 16 x n, output size: 8 x 8 x n, receptive field: 18 + (2-1) * 2 = 20


        # CONVOLUTION BLOCK 3
        self.C8 = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=n*2,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm([8,8]),
            nn.Dropout(dropout_value)
        ) #input size: 8 x 8 x n, output size: 8 x 8 x n*2, receptive field: 20 + (3-1) * 4 = 28

        self.C9 = nn.Sequential(
            nn.Conv2d(in_channels=n*2, out_channels=n*4,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm([8,8]),
            nn.Dropout(dropout_value)
        )  #input size: 8 x 8 x n*2, output size: 8 x 8 x n*4, receptive field: 28 + (3-1) * 4 = 36

        self.C10 = nn.Sequential(
            nn.Conv2d(in_channels=n*4, out_channels=n*4,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm([8,8]),
            nn.Dropout(dropout_value)
        ) #input size: 8 x 8 x n*4, output size: 8 x 8 x n*4, receptive field: 36 + (3-1) * 4 = 44

        # OUTPUT BLOCK
        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1))
        ) #input size: 8 x 8 x n*4, output size: 1 x 1 x n*4, receptive field: 44

        self.c11 = nn.Sequential(
            nn.Conv2d(in_channels=n*4, out_channels=10,
                      kernel_size=(1, 1), padding=0, bias=False),
            # nn.LayerNorm(10),
            # nn.ReLU()
        ) #input size: 1 x 1 x n*4, output size: 1 x 1 x 10, receptive field: 44 + (1-1) * 4 =44



    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)

        x = self.c3(x)
        x = self.p1(x)

        x = self.C4(x)
        x = self.C5(x)
        x = self.C6(x)

        x = self.c7(x)
        x = self.p2(x)

        x = self.C8(x)
        x = self.C9(x)
        x = self.C10(x)

        x = self.GAP(x)
        x = self.c11(x)

        x = x.squeeze()

        return F.log_softmax(x, dim=-1)

   
     
