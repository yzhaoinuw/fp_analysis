import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):
    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n**2)

        return simse


class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class DifferenceLoss(nn.Module):
    def __init__(self, weight=0.08):
        super(DifferenceLoss, self).__init__()
        self.weight = weight

    def forward(self, private_samples, shared_samples):
        private_samples = private_samples - torch.mean(private_samples, dim=0)
        shared_samples = shared_samples - torch.mean(shared_samples, dim=0)
        private_samples = torch.nn.functional.normalize(
            private_samples.clone(), p=2, dim=1
        )
        shared_samples = torch.nn.functional.normalize(
            shared_samples.clone(), p=2, dim=1
        )
        correlation_matrix = torch.matmul(private_samples.t(), shared_samples)
        cost = torch.mean(torch.square(correlation_matrix)) * self.weight
        cost = torch.where(cost > 0, cost, torch.tensor(0.0).cuda())
        return cost


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, x):
        attention_weights = F.softmax(x, dim=-1)
        attended_features = torch.sum(x * attention_weights, dim=1)
        return attended_features


class DSN(nn.Module):
    def __init__(self, code_size=None, n_class=3):
        super(DSN, self).__init__()
        self.code_size = code_size

        ##########################################
        # private source encoder
        ##########################################
        self.conv1 = nn.Conv2d(1, 128, (1, 64))
        self.dropout1 = nn.Dropout(0.5)
        self.conv1_1 = nn.Conv2d(128, 128, (1, 32))
        self.dropout1_1 = nn.Dropout(0.5)
        self.conv1_2 = nn.Conv2d(128, 64, (1, 16))
        self.maxpool1 = nn.MaxPool2d((1, 2))
        self.dropout1_2 = nn.Dropout(0.5)
        self.lstm1 = nn.LSTM(576, 32, batch_first=True, dropout=0.4)
        self.dense1 = nn.Linear(32, 32)

        self.lstm2 = nn.LSTM(1, 64, batch_first=True, bidirectional=True, dropout=0.4)
        self.lstm3 = nn.LSTM(128, 32, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = Attention()
        self.dense2 = nn.Linear(64, 32)

        self.conv2 = nn.Conv2d(1, 64, (1, 64))
        self.maxpool2 = nn.MaxPool2d((1, 2))
        self.dropout2 = nn.Dropout(0.5)
        self.lstm4 = nn.LSTM(2048, 32, batch_first=True, dropout=0.4)
        self.dense3 = nn.Linear(32, 32)

        self.dense4 = nn.Linear(512, 32)

        #########################################
        # private target encoder
        #########################################

        self.targetconv1 = nn.Conv2d(1, 128, (1, 64))
        self.targetdropout1 = nn.Dropout(0.5)
        self.targetconv1_1 = nn.Conv2d(128, 128, (1, 32))
        self.targetdropout1_1 = nn.Dropout(0.5)
        self.targetconv1_2 = nn.Conv2d(128, 64, (1, 16))
        self.targetmaxpool1 = nn.MaxPool2d((1, 2))
        self.targetdropout1_2 = nn.Dropout(0.5)
        self.targetlstm1 = nn.LSTM(576, 32, batch_first=True, dropout=0.4)
        self.targetdense1 = nn.Linear(32, 32)

        self.targetlstm2 = nn.LSTM(
            1, 64, batch_first=True, bidirectional=True, dropout=0.4
        )
        self.targetlstm3 = nn.LSTM(
            128, 32, batch_first=True, bidirectional=True, dropout=0.3
        )
        self.targetattention = Attention()
        self.targetdense2 = nn.Linear(64, 32)

        self.targetconv2 = nn.Conv2d(1, 64, (1, 64))
        self.targetmaxpool2 = nn.MaxPool2d((1, 2))
        self.targetdropout2 = nn.Dropout(0.5)
        self.targetlstm4 = nn.LSTM(2048, 32, batch_first=True, dropout=0.4)
        self.targetdense3 = nn.Linear(32, 32)

        self.targetdense4 = nn.Linear(512, 32)

        ################################
        # shared encoder (dann_mnist)
        ################################

        self.sharedconv1 = nn.Conv2d(1, 256, (1, 64))
        self.shareddropout1 = nn.Dropout(0.5)
        self.sharedconv1_1 = nn.Conv2d(256, 128, (1, 32))
        self.shareddropout1_1 = nn.Dropout(0.4)
        self.sharedconv1_2 = nn.Conv2d(128, 64, (1, 16))
        self.sharedmaxpool1 = nn.MaxPool2d((1, 2))
        self.shareddropout1_2 = nn.Dropout(0.4)
        self.sharedlstm1 = nn.LSTM(576, 64, batch_first=True, dropout=0.4)
        self.sharedlstm1_2 = nn.LSTM(64, 64, batch_first=True, dropout=0.2)
        self.shareddense1 = nn.Linear(64, 32)

        self.sharedlstm2 = nn.LSTM(
            1, 64, batch_first=True, bidirectional=True, dropout=0.4
        )
        self.sharedlstm3 = nn.LSTM(
            128, 64, batch_first=True, bidirectional=True, dropout=0.3
        )
        self.sharedattention = Attention()
        self.shareddense2 = nn.Linear(128, 32)

        self.sharedconv2 = nn.Conv2d(1, 64, (1, 64))
        self.sharedmaxpool2 = nn.MaxPool2d((1, 2))
        self.shareddropout2 = nn.Dropout(0.4)
        self.sharedlstm4 = nn.LSTM(2048, 64, batch_first=True, dropout=0.4)
        self.shareddense3 = nn.Linear(64, 32)

        self.shareddense4 = nn.Linear(512, 32)

        # classify 10 numbers
        self.shared_encoder_pred_class = nn.Sequential(nn.Linear(self.code_size, 3))

        # classify two domain
        self.shared_encoder_pred_domain = nn.Sequential(nn.Linear(self.code_size, 2))

    def forward(
        self, input_flow1, input_flow2, input_flow3, input_flow4, mode, signaling, p=0.0
    ):
        result = []

        if mode == "source":
            # source private encoder

            x = F.relu(self.conv1(input_flow1))
            # x = self.maxpool1(x)
            x = self.dropout1(x)
            x = F.relu(self.conv1_1(x))
            x = self.dropout1_1(x)
            x = F.relu(self.conv1_2(x))
            x = self.maxpool1(x)
            x = self.dropout1_2(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(2), x.size(1) * x.size(3))
            # print(x.size())
            x, _ = self.lstm1(x)
            # print(x.size())
            x = F.relu(self.dense1(x[:, -1, :]))
            # print(x.size())

            y, _ = self.lstm2(input_flow2)
            y, _ = self.lstm3(y)
            # print(y.size())
            y = self.attention(y)
            # print(y.size())
            y = F.relu(self.dense2(y))

            z = F.relu(self.conv2(input_flow3))
            z = self.maxpool2(z)
            z = self.dropout2(z)
            z = z.view(z.size(0), z.size(2), z.size(1) * z.size(3))
            z, _ = self.lstm4(z)
            z = F.relu(self.dense3(z[:, -1, :]))

            w = F.relu(self.dense4(input_flow4))

            if signaling == 100:
                private_code = torch.cat((x, y, z, w), dim=1)
            elif signaling == 200:
                private_code = torch.cat((x, z, w), dim=1)
            elif signaling == 400:
                private_code = torch.cat((x, w), dim=1)
            elif signaling == 300:
                private_code = torch.cat((x, y, w), dim=1)
            elif signaling == 600:
                private_code = y
            elif signaling == 500:
                private_code = w

        elif mode == "target":
            # target private encoder
            x = F.relu(self.targetconv1(input_flow1))
            # x = self.maxpool1(x)
            x = self.targetdropout1(x)
            x = F.relu(self.targetconv1_1(x))
            x = self.targetdropout1_1(x)
            x = F.relu(self.targetconv1_2(x))
            x = self.targetmaxpool1(x)
            x = self.targetdropout1_2(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(2), x.size(1) * x.size(3))
            x, _ = self.targetlstm1(x)
            # print(x.size())
            x = F.relu(self.targetdense1(x[:, -1, :]))
            # print(x.size())

            y, _ = self.targetlstm2(input_flow2)
            y, _ = self.targetlstm3(y)
            # print(y.size())
            y = self.targetattention(y)
            # print(y.size())
            y = F.relu(self.targetdense2(y))

            z = F.relu(self.targetconv2(input_flow3))
            z = self.targetmaxpool2(z)
            z = self.targetdropout2(z)
            z = z.view(z.size(0), z.size(2), z.size(1) * z.size(3))
            z, _ = self.targetlstm4(z)
            z = F.relu(self.targetdense3(z[:, -1, :]))

            w = F.relu(self.targetdense4(input_flow4))

            if signaling == 100:
                private_code = torch.cat((x, y, z, w), dim=1)
            elif signaling == 200:
                private_code = torch.cat((x, z, w), dim=1)
            elif signaling == 400:
                private_code = torch.cat((x, w), dim=1)
            elif signaling == 300:
                private_code = torch.cat((x, y, w), dim=1)
            elif signaling == 600:
                private_code = y
            elif signaling == 500:
                private_code = w

        result.append(private_code)

        # shared encoder
        x = F.relu(self.sharedconv1(input_flow1))
        # x = self.maxpool1(x)
        x = self.shareddropout1(x)
        x = F.relu(self.sharedconv1_1(x))
        x = self.shareddropout1_1(x)
        x = F.relu(self.sharedconv1_2(x))
        x = self.sharedmaxpool1(x)
        x = self.shareddropout1_2(x)
        # print(x.size())
        x = x.view(x.size(0), x.size(2), x.size(1) * x.size(3))
        # print('hey',x.size())
        x, _ = self.sharedlstm1(x)
        x, _ = self.sharedlstm1_2(x)
        # print(x.size())
        x = F.relu(self.shareddense1(x[:, -1, :]))
        # print(x.size())

        y, _ = self.sharedlstm2(input_flow2)
        y, _ = self.sharedlstm3(y)
        # print(y.size())
        y = self.sharedattention(y)
        # print(y.size())
        y = F.relu(self.shareddense2(y))

        z = F.relu(self.sharedconv2(input_flow3))
        z = self.sharedmaxpool2(z)
        z = self.shareddropout2(z)
        z = z.view(z.size(0), z.size(2), z.size(1) * z.size(3))
        z, _ = self.sharedlstm4(z)
        z = F.relu(self.shareddense3(z[:, -1, :]))

        w = F.relu(self.shareddense4(input_flow4))

        if signaling == 100:
            shared_code = torch.cat((x, y, z, w), dim=1)
        elif signaling == 200:
            shared_code = torch.cat((x, z, w), dim=1)
        elif signaling == 400:
            shared_code = torch.cat((x, w), dim=1)
        elif signaling == 300:
            shared_code = torch.cat((x, y, w), dim=1)
        elif signaling == 600:
            shared_code = y
        elif signaling == 500:
            shared_code = w

        result.append(shared_code)

        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = F.softmax(
            self.shared_encoder_pred_domain(reversed_shared_code), dim=1
        )
        result.append(domain_label)

        if mode == "source":
            class_label = F.softmax(self.shared_encoder_pred_class(shared_code), dim=1)
            result.append(class_label)

        return result


########################IF we want Four class preds
class DSN2(nn.Module):
    def __init__(self, code_size=None, n_class=3):
        super(DSN2, self).__init__()
        self.code_size = code_size

        ##########################################
        # private source encoder
        ##########################################
        self.conv1 = nn.Conv2d(1, 128, (1, 64))
        self.dropout1 = nn.Dropout(0.5)
        self.conv1_1 = nn.Conv2d(128, 128, (1, 32))
        self.dropout1_1 = nn.Dropout(0.5)
        self.conv1_2 = nn.Conv2d(128, 64, (1, 16))
        self.maxpool1 = nn.MaxPool2d((1, 2))
        self.dropout1_2 = nn.Dropout(0.5)
        self.lstm1 = nn.LSTM(576, 32, batch_first=True, dropout=0.4)
        self.dense1 = nn.Linear(32, 32)

        self.lstm2 = nn.LSTM(1, 64, batch_first=True, bidirectional=True, dropout=0.4)
        self.lstm3 = nn.LSTM(128, 32, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = Attention()
        self.dense2 = nn.Linear(64, 32)

        self.conv2 = nn.Conv2d(1, 64, (1, 64))
        self.maxpool2 = nn.MaxPool2d((1, 2))
        self.dropout2 = nn.Dropout(0.5)
        self.lstm4 = nn.LSTM(2048, 32, batch_first=True, dropout=0.4)
        self.dense3 = nn.Linear(32, 32)

        self.dense4 = nn.Linear(512, 32)

        #########################################
        # private target encoder
        #########################################

        self.targetconv1 = nn.Conv2d(1, 128, (1, 64))
        self.targetdropout1 = nn.Dropout(0.5)
        self.targetconv1_1 = nn.Conv2d(128, 128, (1, 32))
        self.targetdropout1_1 = nn.Dropout(0.5)
        self.targetconv1_2 = nn.Conv2d(128, 64, (1, 16))
        self.targetmaxpool1 = nn.MaxPool2d((1, 2))
        self.targetdropout1_2 = nn.Dropout(0.5)
        self.targetlstm1 = nn.LSTM(576, 32, batch_first=True, dropout=0.4)
        self.targetdense1 = nn.Linear(32, 32)

        self.targetlstm2 = nn.LSTM(
            1, 64, batch_first=True, bidirectional=True, dropout=0.4
        )
        self.targetlstm3 = nn.LSTM(
            128, 32, batch_first=True, bidirectional=True, dropout=0.3
        )
        self.targetattention = Attention()
        self.targetdense2 = nn.Linear(64, 32)

        self.targetconv2 = nn.Conv2d(1, 64, (1, 64))
        self.targetmaxpool2 = nn.MaxPool2d((1, 2))
        self.targetdropout2 = nn.Dropout(0.5)
        self.targetlstm4 = nn.LSTM(2048, 32, batch_first=True, dropout=0.4)
        self.targetdense3 = nn.Linear(32, 32)

        self.targetdense4 = nn.Linear(512, 32)

        ################################
        # shared encoder (dann_mnist)
        ################################

        self.sharedconv1 = nn.Conv2d(1, 256, (1, 64))
        self.shareddropout1 = nn.Dropout(0.5)
        self.sharedconv1_1 = nn.Conv2d(256, 128, (1, 32))
        self.shareddropout1_1 = nn.Dropout(0.4)
        self.sharedconv1_2 = nn.Conv2d(128, 64, (1, 16))
        self.sharedmaxpool1 = nn.MaxPool2d((1, 2))
        self.shareddropout1_2 = nn.Dropout(0.4)
        self.sharedlstm1 = nn.LSTM(576, 64, batch_first=True, dropout=0.4)
        self.sharedlstm1_2 = nn.LSTM(64, 64, batch_first=True, dropout=0.2)
        self.shareddense1 = nn.Linear(64, 32)

        self.sharedlstm2 = nn.LSTM(
            1, 64, batch_first=True, bidirectional=True, dropout=0.4
        )
        self.sharedlstm3 = nn.LSTM(
            128, 64, batch_first=True, bidirectional=True, dropout=0.3
        )
        self.sharedattention = Attention()
        self.shareddense2 = nn.Linear(128, 32)

        self.sharedconv2 = nn.Conv2d(1, 64, (1, 64))
        self.sharedmaxpool2 = nn.MaxPool2d((1, 2))
        self.shareddropout2 = nn.Dropout(0.4)
        self.sharedlstm4 = nn.LSTM(2048, 64, batch_first=True, dropout=0.4)
        self.shareddense3 = nn.Linear(64, 32)

        self.shareddense4 = nn.Linear(512, 32)

        # classify 10 numbers
        self.shared_encoder_pred_class = nn.Sequential(nn.Linear(self.code_size, 2))

        # classify two domain
        self.shared_encoder_pred_domain = nn.Sequential(nn.Linear(self.code_size, 2))

    def forward(
        self, input_flow1, input_flow2, input_flow3, input_flow4, mode, signaling, p=0.0
    ):
        result = []

        if mode == "source":
            # source private encoder

            x = F.relu(self.conv1(input_flow1))
            # x = self.maxpool1(x)
            x = self.dropout1(x)
            x = F.relu(self.conv1_1(x))
            x = self.dropout1_1(x)
            x = F.relu(self.conv1_2(x))
            x = self.maxpool1(x)
            x = self.dropout1_2(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(2), x.size(1) * x.size(3))
            # print(x.size())
            x, _ = self.lstm1(x)
            # print(x.size())
            x = F.relu(self.dense1(x[:, -1, :]))
            # print(x.size())

            y, _ = self.lstm2(input_flow2)
            y, _ = self.lstm3(y)
            # print(y.size())
            y = self.attention(y)
            # print(y.size())
            y = F.relu(self.dense2(y))

            z = F.relu(self.conv2(input_flow3))
            z = self.maxpool2(z)
            z = self.dropout2(z)
            z = z.view(z.size(0), z.size(2), z.size(1) * z.size(3))
            z, _ = self.lstm4(z)
            z = F.relu(self.dense3(z[:, -1, :]))

            w = F.relu(self.dense4(input_flow4))

            if signaling == 100:
                private_code = torch.cat((x, y, z, w), dim=1)
            elif signaling == 200:
                private_code = torch.cat((x, z, w), dim=1)
            elif signaling == 400:
                private_code = torch.cat((x, w), dim=1)
            elif signaling == 300:
                private_code = torch.cat((x, y, w), dim=1)
            elif signaling == 600:
                private_code = y
            elif signaling == 500:
                private_code = w

        elif mode == "target":
            # target private encoder
            x = F.relu(self.targetconv1(input_flow1))
            # x = self.maxpool1(x)
            x = self.targetdropout1(x)
            x = F.relu(self.targetconv1_1(x))
            x = self.targetdropout1_1(x)
            x = F.relu(self.targetconv1_2(x))
            x = self.targetmaxpool1(x)
            x = self.targetdropout1_2(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(2), x.size(1) * x.size(3))
            x, _ = self.targetlstm1(x)
            # print(x.size())
            x = F.relu(self.targetdense1(x[:, -1, :]))
            # print(x.size())

            y, _ = self.targetlstm2(input_flow2)
            y, _ = self.targetlstm3(y)
            # print(y.size())
            y = self.targetattention(y)
            # print(y.size())
            y = F.relu(self.targetdense2(y))

            z = F.relu(self.targetconv2(input_flow3))
            z = self.targetmaxpool2(z)
            z = self.targetdropout2(z)
            z = z.view(z.size(0), z.size(2), z.size(1) * z.size(3))
            z, _ = self.targetlstm4(z)
            z = F.relu(self.targetdense3(z[:, -1, :]))

            w = F.relu(self.targetdense4(input_flow4))

            if signaling == 100:
                private_code = torch.cat((x, y, z, w), dim=1)
            elif signaling == 200:
                private_code = torch.cat((x, z, w), dim=1)
            elif signaling == 400:
                private_code = torch.cat((x, w), dim=1)
            elif signaling == 300:
                private_code = torch.cat((x, y, w), dim=1)
            elif signaling == 600:
                private_code = y
            elif signaling == 500:
                private_code = w

        result.append(private_code)

        # shared encoder
        x = F.relu(self.sharedconv1(input_flow1))
        # x = self.maxpool1(x)
        x = self.shareddropout1(x)
        x = F.relu(self.sharedconv1_1(x))
        x = self.shareddropout1_1(x)
        x = F.relu(self.sharedconv1_2(x))
        x = self.sharedmaxpool1(x)
        x = self.shareddropout1_2(x)
        # print(x.size())
        x = x.view(x.size(0), x.size(2), x.size(1) * x.size(3))
        # print('hey',x.size())
        x, _ = self.sharedlstm1(x)
        x, _ = self.sharedlstm1_2(x)
        # print(x.size())
        x = F.relu(self.shareddense1(x[:, -1, :]))
        # print(x.size())

        y, _ = self.sharedlstm2(input_flow2)
        y, _ = self.sharedlstm3(y)
        # print(y.size())
        y = self.sharedattention(y)
        # print(y.size())
        y = F.relu(self.shareddense2(y))

        z = F.relu(self.sharedconv2(input_flow3))
        z = self.sharedmaxpool2(z)
        z = self.shareddropout2(z)
        z = z.view(z.size(0), z.size(2), z.size(1) * z.size(3))
        z, _ = self.sharedlstm4(z)
        z = F.relu(self.shareddense3(z[:, -1, :]))

        w = F.relu(self.shareddense4(input_flow4))

        if signaling == 100:
            shared_code = torch.cat((x, y, z, w), dim=1)
        elif signaling == 200:
            shared_code = torch.cat((x, z, w), dim=1)
        elif signaling == 400:
            shared_code = torch.cat((x, w), dim=1)
        elif signaling == 300:
            shared_code = torch.cat((x, y, w), dim=1)
        elif signaling == 600:
            shared_code = y
        elif signaling == 500:
            shared_code = w

        result.append(shared_code)

        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = F.softmax(
            self.shared_encoder_pred_domain(reversed_shared_code), dim=1
        )
        result.append(domain_label)

        if mode == "source":
            class_label = F.softmax(self.shared_encoder_pred_class(shared_code), dim=1)
            result.append(class_label)

        return result


class DSN3(nn.Module):
    def __init__(self, code_size=None, n_class=3):
        super(DSN3, self).__init__()
        self.code_size = code_size

        ##########################################
        # private source encoder
        ##########################################
        self.conv1 = nn.Conv2d(1, 128, (1, 64))
        self.dropout1 = nn.Dropout(0.5)
        self.conv1_1 = nn.Conv2d(128, 128, (1, 32))
        self.dropout1_1 = nn.Dropout(0.5)
        self.conv1_2 = nn.Conv2d(128, 64, (1, 16))
        self.maxpool1 = nn.MaxPool2d((1, 2))
        self.dropout1_2 = nn.Dropout(0.5)
        self.lstm1 = nn.LSTM(576, 32, batch_first=True, dropout=0.4)
        self.dense1 = nn.Linear(32, 32)

        self.conv2 = nn.Conv2d(1, 64, (1, 64))
        self.maxpool2 = nn.MaxPool2d((1, 2))
        self.dropout2 = nn.Dropout(0.5)
        self.lstm4 = nn.LSTM(2048, 32, batch_first=True, dropout=0.4)
        self.dense3 = nn.Linear(32, 32)

        self.dense4 = nn.Linear(512, 32)

        #########################################
        # private target encoder
        #########################################

        self.targetconv1 = nn.Conv2d(1, 128, (1, 64))
        self.targetdropout1 = nn.Dropout(0.5)
        self.targetconv1_1 = nn.Conv2d(128, 128, (1, 32))
        self.targetdropout1_1 = nn.Dropout(0.5)
        self.targetconv1_2 = nn.Conv2d(128, 64, (1, 16))
        self.targetmaxpool1 = nn.MaxPool2d((1, 2))
        self.targetdropout1_2 = nn.Dropout(0.5)
        self.targetlstm1 = nn.LSTM(576, 32, batch_first=True, dropout=0.4)
        self.targetdense1 = nn.Linear(32, 32)

        self.targetconv2 = nn.Conv2d(1, 64, (1, 64))
        self.targetmaxpool2 = nn.MaxPool2d((1, 2))
        self.targetdropout2 = nn.Dropout(0.5)
        self.targetlstm4 = nn.LSTM(2048, 32, batch_first=True, dropout=0.4)
        self.targetdense3 = nn.Linear(32, 32)

        self.targetdense4 = nn.Linear(512, 32)

        ################################
        # shared encoder (dann_mnist)
        ################################

        self.sharedconv1 = nn.Conv2d(1, 256, (1, 64))
        self.shareddropout1 = nn.Dropout(0.5)
        self.sharedconv1_1 = nn.Conv2d(256, 128, (1, 32))
        self.shareddropout1_1 = nn.Dropout(0.4)
        self.sharedconv1_2 = nn.Conv2d(128, 64, (1, 16))
        self.sharedmaxpool1 = nn.MaxPool2d((1, 2))
        self.shareddropout1_2 = nn.Dropout(0.4)
        self.sharedlstm1 = nn.LSTM(576, 64, batch_first=True, dropout=0.4)
        self.sharedlstm1_2 = nn.LSTM(64, 64, batch_first=True, dropout=0.2)
        self.shareddense1 = nn.Linear(64, 32)

        self.sharedconv2 = nn.Conv2d(1, 64, (1, 64))
        self.sharedmaxpool2 = nn.MaxPool2d((1, 2))
        self.shareddropout2 = nn.Dropout(0.4)
        self.sharedlstm4 = nn.LSTM(2048, 64, batch_first=True, dropout=0.4)
        self.shareddense3 = nn.Linear(64, 32)

        self.shareddense4 = nn.Linear(512, 32)

        # classify 10 numbers
        self.shared_encoder_pred_class = nn.Sequential(nn.Linear(self.code_size, 3))

        # classify two domain
        self.shared_encoder_pred_domain = nn.Sequential(nn.Linear(self.code_size, 2))

    def forward(self, input_flow1, input_flow3, input_flow4, mode, signaling, p=0.0):
        result = []

        if mode == "source":
            # source private encoder

            x = F.relu(self.conv1(input_flow1))
            # x = self.maxpool1(x)
            x = self.dropout1(x)
            x = F.relu(self.conv1_1(x))
            x = self.dropout1_1(x)
            x = F.relu(self.conv1_2(x))
            x = self.maxpool1(x)
            x = self.dropout1_2(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(2), x.size(1) * x.size(3))
            # print(x.size())
            x, _ = self.lstm1(x)
            # print(x.size())
            x = F.relu(self.dense1(x[:, -1, :]))

            z = F.relu(self.conv2(input_flow3))
            z = self.maxpool2(z)
            z = self.dropout2(z)
            z = z.view(z.size(0), z.size(2), z.size(1) * z.size(3))
            z, _ = self.lstm4(z)
            z = F.relu(self.dense3(z[:, -1, :]))

            w = F.relu(self.dense4(input_flow4))

            if signaling == 200:
                private_code = torch.cat((x, z, w), dim=1)
            elif signaling == 400:
                private_code = torch.cat((x, w), dim=1)

            elif signaling == 500:
                private_code = w

        elif mode == "target":
            # target private encoder
            x = F.relu(self.targetconv1(input_flow1))
            # x = self.maxpool1(x)
            x = self.targetdropout1(x)
            x = F.relu(self.targetconv1_1(x))
            x = self.targetdropout1_1(x)
            x = F.relu(self.targetconv1_2(x))
            x = self.targetmaxpool1(x)
            x = self.targetdropout1_2(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(2), x.size(1) * x.size(3))
            x, _ = self.targetlstm1(x)
            # print(x.size())
            x = F.relu(self.targetdense1(x[:, -1, :]))

            z = F.relu(self.targetconv2(input_flow3))
            z = self.targetmaxpool2(z)
            z = self.targetdropout2(z)
            z = z.view(z.size(0), z.size(2), z.size(1) * z.size(3))
            z, _ = self.targetlstm4(z)
            z = F.relu(self.targetdense3(z[:, -1, :]))

            w = F.relu(self.targetdense4(input_flow4))

            if signaling == 200:
                private_code = torch.cat((x, z, w), dim=1)
            elif signaling == 400:
                private_code = torch.cat((x, w), dim=1)

            elif signaling == 500:
                private_code = w

        result.append(private_code)

        # shared encoder
        x = F.relu(self.sharedconv1(input_flow1))
        # x = self.maxpool1(x)
        x = self.shareddropout1(x)
        x = F.relu(self.sharedconv1_1(x))
        x = self.shareddropout1_1(x)
        x = F.relu(self.sharedconv1_2(x))
        x = self.sharedmaxpool1(x)
        x = self.shareddropout1_2(x)
        x = x.view(x.size(0), x.size(2), x.size(1) * x.size(3))

        x, _ = self.sharedlstm1(x)
        x, _ = self.sharedlstm1_2(x)

        x = F.relu(self.shareddense1(x[:, -1, :]))

        z = F.relu(self.sharedconv2(input_flow3))
        z = self.sharedmaxpool2(z)
        z = self.shareddropout2(z)
        z = z.view(z.size(0), z.size(2), z.size(1) * z.size(3))
        z, _ = self.sharedlstm4(z)
        z = F.relu(self.shareddense3(z[:, -1, :]))

        w = F.relu(self.shareddense4(input_flow4))

        # if signaling== 100:
        #     shared_code = torch.cat((x, y, z, w), dim=1)
        if signaling == 200:
            shared_code = torch.cat((x, z, w), dim=1)
        elif signaling == 400:
            shared_code = torch.cat((x, w), dim=1)

        elif signaling == 500:
            shared_code = w

        result.append(shared_code)

        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = F.softmax(
            self.shared_encoder_pred_domain(reversed_shared_code), dim=1
        )
        result.append(domain_label)

        if mode == "source":
            class_label = F.softmax(self.shared_encoder_pred_class(shared_code), dim=1)
            result.append(class_label)

        return result
