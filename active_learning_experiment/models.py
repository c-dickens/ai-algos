import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.embDim = 128
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                out = self.conv1(x)
                out = F.relu(out)
                out = self.conv2(out)
                out = F.relu(out)
                out = F.max_pool2d(out, 2)
                out = self.dropout1(out)
                out = torch.flatten(out, 1)
                out = self.fc1(out)
                out = F.relu(out)
                e = self.dropout2(out)
        else:
            out = self.conv1(x)
            out = F.relu(out)
            out = self.conv2(out)
            out = F.relu(out)
            out = F.max_pool2d(out, 2)
            out = self.dropout1(out)
            out = torch.flatten(out, 1)
            out = self.fc1(out)
            out = F.relu(out)
            e = self.dropout2(out)
        out = self.fc2(e)
        if last:
            return out, e
        else:
            return out

    def get_embedding_dim(self):
        return self.embDim

    def get_embeddings(self, x, freeze=False):
        """
        Returns the embeddings (output of dropout2 layer) for input x.
        """
        if freeze:
            with torch.no_grad():
                out = self.conv1(x)
                out = F.relu(out)
                out = self.conv2(out)
                out = F.relu(out)
                out = F.max_pool2d(out, 2)
                out = self.dropout1(out)
                out = torch.flatten(out, 1)
                out = self.fc1(out)
                out = F.relu(out)
                e = self.dropout2(out)
        else:
            out = self.conv1(x)
            out = F.relu(out)
            out = self.conv2(out)
            out = F.relu(out)
            out = F.max_pool2d(out, 2)
            out = self.dropout1(out)
            out = torch.flatten(out, 1)
            out = self.fc1(out)
            out = F.relu(out)
            e = self.dropout2(out)
        return e

class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.embDim = 256
        self.conv1 = nn.Conv2d(3,   64,  3)
        self.conv2 = nn.Conv2d(64,  128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                out = self.pool(F.relu(self.conv1(x)))
                out = self.pool(F.relu(self.conv2(out)))
                out = self.pool(F.relu(self.conv3(out)))
                out = out.view(-1, 64 * 4 * 4)
                out = F.relu(self.fc1(out))
                e = F.relu(self.fc2(out))
        else:
            out = self.pool(F.relu(self.conv1(x)))
            out = self.pool(F.relu(self.conv2(out)))
            out = self.pool(F.relu(self.conv3(out)))
            out = out.view(-1, 64 * 4 * 4)
            out = F.relu(self.fc1(out))
            e = F.relu(self.fc2(out))
        out = self.fc3(e)
        if last:
            return out, e
        else:
            return out

    def get_embedding_dim(self):
        return self.embDim

    def get_embeddings(self, x, freeze=False):
        """
        Returns the embeddings (output of fc2 layer) for input x.
        """
        if freeze:
            with torch.no_grad():
                out = self.pool(F.relu(self.conv1(x)))
                out = self.pool(F.relu(self.conv2(out)))
                out = self.pool(F.relu(self.conv3(out)))
                out = out.view(-1, 64 * 4 * 4)
                out = F.relu(self.fc1(out))
                e = F.relu(self.fc2(out))
        else:
            out = self.pool(F.relu(self.conv1(x)))
            out = self.pool(F.relu(self.conv2(out)))
            out = self.pool(F.relu(self.conv3(out)))
            out = out.view(-1, 64 * 4 * 4)
            out = F.relu(self.fc1(out))
            e = F.relu(self.fc2(out))
        return e 