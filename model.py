from networks import LSTM_model
import torch,os
import torch.nn as nn
from utils import weight_init
from torch.optim import lr_scheduler

class FDI_model(torch.nn.Module):
    def __init__(self,opts,checkpoint_dir):
        super(FDI_model,self).__init__()
        self.gpu = opts.gpu
        self.checkpoint_dir = checkpoint_dir
        self.step_lr_epoch = opts.step_lr_epoch
        self.gamma_lr = opts.gamma_lr
        self.LSTM_model = LSTM_model(opts.embedding_dim,opts.hidden_dim,opts.target_size,batchsize=opts.batchSize,
                                     isTraining=opts.isTrain,use_dropout=opts.use_dropout,gpu=opts.gpu,
                                     lstm_layers=opts.lstm_layers, att_dim=opts.att_dim)

        self.LSTM_model.apply(weight_init('orthogonal'))
        if not opts.no_weight_12:
            class_weight = torch.tensor([1.0,12.0,12.0,12.0,12.0])/49 # weighted balanced for different classes of each steps
            self.loss = nn.CrossEntropyLoss(weight=class_weight)
        else:
            self.loss = nn.CrossEntropyLoss()
        self.val_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(list(self.LSTM_model.parameters()),
                                        lr=opts.lr, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=opts.step_lr_epoch,
                                        gamma=opts.gamma_lr, last_epoch=-1)


    def set_input(self, data):
        if self.gpu >= 0:
            self.input = data['input'].cuda(self.gpu).float()
            self.target_label = data['label'].cuda(self.gpu).view(-1)
        else:
            self.input = data['input'].cpu().float()
            self.target_label = data['label'].cpu().view(-1)



    def optimize(self):
        self.LSTM_model.train()
        self.optimizer.zero_grad()
        self.LSTM_model.hidden = self.LSTM_model.init_hidden(self.input.shape[0])
        self.pred_labels = self.LSTM_model(self.input)
        self.train_loss = self.loss(self.pred_labels, self.target_label)

        self.train_loss.backward()
        self.optimizer.step()

    def save_model(self, epoch):
        ck_name = os.path.join(self.checkpoint_dir, 'lstm_%03d.pt' % (epoch))
        opt_name = os.path.join(self.checkpoint_dir, 'optimizer_%03d.pt' % (epoch))
        torch.save({'lstm': self.LSTM_model.state_dict()}, ck_name)
        torch.save({'opt': self.optimizer.state_dict()}, opt_name)

    def load_model(self,epoch):
        # Load lstm model
        if self.gpu >= 0:
            state_dict = torch.load(os.path.join(self.checkpoint_dir, 'lstm_%03d.pt' % (epoch)), map_location=torch.device('cuda:' + str(self.gpu)))
        else:
            state_dict = torch.load(os.path.join(self.checkpoint_dir, 'lstm_%03d.pt' % (epoch)), map_location=torch.device('cpu'))

        self.LSTM_model.load_state_dict(state_dict['lstm'])
        # Load optimizer
        if self.gpu >= 0:
            state_dict = torch.load(os.path.join(self.checkpoint_dir, 'optimizer_%03d.pt' % (epoch)), map_location=torch.device('cuda:' + str(self.gpu)))
        else:
            state_dict = torch.load(os.path.join(self.checkpoint_dir, 'optimizer_%03d.pt' % (epoch)), map_location=torch.device('cpu'))
        self.optimizer.load_state_dict(state_dict['opt'])
        # Reinitilize scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.step_lr_epoch,
                                            gamma=self.gamma_lr, last_epoch=epoch)
        print('Load models from epoch %d' % epoch)

    def validate(self):
        with torch.no_grad():
            self.LSTM_model.eval()
            self.LSTM_model.hidden = self.LSTM_model.init_hidden(self.input.shape[0])
            self.pred_labels = self.LSTM_model(self.input)
            return self.val_loss(self.pred_labels, self.target_label).mean().item()

    def test(self):
        with torch.no_grad():
            self.LSTM_model.eval()
            self.LSTM_model.hidden = self.LSTM_model.init_hidden(self.input.shape[0])
            self.pred_labels = self.LSTM_model(self.input)
            return self.pred_labels,self.target_label




