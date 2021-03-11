import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_model(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, target_size, batchsize, isTraining=True, use_dropout=False, gpu=0,
                 lstm_layers=1, att_dim=0):
        super(LSTM_model,self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.target_size=target_size
        self.target_size=target_size
        self.isTraining = isTraining
        self.use_dropout = use_dropout
        self.gpu = gpu
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(self.embedding_dim,self.hidden_dim,lstm_layers,batch_first=True)
        self.out2label = nn.Linear(self.hidden_dim,self.target_size)
        self.att_dim = att_dim
        if self.att_dim > 0:
            self.W1 = nn.Linear(self.embedding_dim, att_dim)
            self.W2 = nn.Linear(self.embedding_dim, att_dim)
            self.W3 = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.hidden = self.init_hidden(batchsize)

    def init_hidden(self,minibatch_size):
        if self.gpu >= 0:
            return (torch.zeros(self.lstm_layers, minibatch_size, self.hidden_dim).cuda(self.gpu),
                torch.zeros(self.lstm_layers, minibatch_size, self.hidden_dim).cuda(self.gpu))
        else:
            return (torch.zeros(self.lstm_layers, minibatch_size, self.hidden_dim).cpu(),
                    torch.zeros(self.lstm_layers, minibatch_size, self.hidden_dim).cpu())

    def forward(self,inputs): # input: batchsize(16) * time sequence (15) * vector (13)
        if self.att_dim > 0:
            att1 = F.softmax(self.W1(inputs),dim=2)  # 15 * 13 --> 15 * m
            att2 = F.softmax(self.W2(inputs),dim=2)  # 15 * 13 --> 15 * m
            mat_h = torch.matmul(att1, att2.transpose(1, 2))  # (15 * m) * (m * 15) --> (15 * 15)
            x_b = torch.matmul(mat_h, self.W3(inputs))  # (15 * 15) * (15 * 13 --> 15 * 13) -- > 15 * 13
            anc_outputs = inputs + x_b
        else:
            anc_outputs = inputs
        out, self.hidden = self.lstm(anc_outputs,self.hidden)

        labels = self.out2label(out)
        if self.use_dropout:
            labels = F.dropout(labels, p=0.5, training=self.isTraining)
        return labels.view(-1,self.target_size)

