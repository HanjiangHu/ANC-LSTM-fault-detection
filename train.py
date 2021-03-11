from model import FDI_model
from data import FDI_train_dataset, FDI_val_dataset
from utils import save_opts,mkdirs_now,write_loss_on_tb
import time
import torch.utils.data
import torch.nn.functional as F

import argparse
import os

import tensorboardX


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help="name of the model")
    parser.add_argument('--input_path', type=str,default="./dataset", help="the path of dataset")
    parser.add_argument('--continue_train',  action='store_true', help="whether to continue training from --checkpoint_epoch ")
    parser.add_argument('--checkpoint_epoch', type=int, default=0,help="load from which iteration")

    parser.add_argument('--batchSize', type=int, default=16, help="batch size for training")
    parser.add_argument('--nThreads', type=int, default=2, help="num of workers when loading dataset")
    parser.add_argument('--show_tb_freq',type=int, default=10, help="the frequency of updating tensorboard")

    parser.add_argument('--output_path', type=str, default='./outputs', help="path for logs, checkpoints, and options")
    parser.add_argument('--embedding_dim', type=int, default=13, help="the dimension of input samples from json files")
    parser.add_argument('--hidden_dim', type=int, default=60, help="the dimension of hidden layers")
    parser.add_argument('--target_size', type=int, default=5, help="the number of classification candidates, "
                                                                   "including healthy mode")
    parser.add_argument('--lr', type=float, default=0.0001, help="initial learning rate")
    parser.add_argument('--beta1', type=float, default=0.9, help="hyperparameter in Adam")
    parser.add_argument('--beta2', type=float, default=0.999, help="hyperparameter in Adam")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="the regularization term")
    parser.add_argument('--total_epoch', type=int, default=20000, help="the total iteration for training")
    parser.add_argument('--step_lr_epoch', type=int, default=20000, help="how many iterations to reduce learning rate")
    parser.add_argument('--gamma_lr', type=float, default=0.5, help="the ratio of decreasing learning rate")
    parser.add_argument('--save_checkpoint_freq', type=int, default=5000, help="the frequency of saving checkpoints")
    parser.add_argument('--no_weight_12', action='store_true', help="whether to use weighted loss")
    parser.add_argument('--use_dropout', action='store_true', help=" whether to use dropout in the model")
    parser.add_argument('--gpu', type=int, default=0, help="GPU CUDA index, -1 for cpu only")
    parser.add_argument('--lstm_layers', type=int, default=6, help="the num of layers for LSTM module")
    parser.add_argument('--att_dim', type=int, default=15, help="the size of attention weight matrix in ANC module, "
                                                               "0 for no ANC module")


    opts = parser.parse_args()

    opts.isTrain = True

    # Setup logger, chechkpoint folders and save options
    log_dir = os.path.join(opts.output_path, opts.name,'tb_logs')
    checkpoint_dir = os.path.join(opts.output_path, opts.name, 'checkpoints')
    save_opts(opts)
    mkdirs_now([log_dir,checkpoint_dir])
    train_writer = tensorboardX.SummaryWriter(log_dir)

    train_dataset = FDI_train_dataset(opts)
    train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=opts.batchSize,
                num_workers=int(opts.nThreads))
    print('# training images = %d' % len(train_dataset))

    val_dataset = FDI_val_dataset(opts)
    val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                shuffle=True,
                batch_size=1,
                num_workers=1)

    model = FDI_model(opts,checkpoint_dir)
    if opts.gpu >= 0:
        model = model.cuda(opts.gpu)
    else:
        model = model.cpu()
    if opts.continue_train and opts.isTrain and opts.checkpoint_epoch > 0:
        model.load_model(opts.checkpoint_epoch)
    elif not opts.isTrain and opts.checkpoint_epoch > 0:
        model.load_model(opts.checkpoint_epoch)
    if opts.checkpoint_epoch > 0 and not opts.continue_train:
        opts.checkpoint_epoch = 0
    for epoch in range(opts.checkpoint_epoch+1, opts.total_epoch+1):
        # epoch starts from 1
        epoch_start_time = time.time()
        epoch_total_loss = 0
        for i, data in enumerate(train_dataloader):
            model.set_input(data)

            model.optimize()
            epoch_total_loss += model.train_loss.mean().item()
        model.scheduler.step()
        print('End of epoch %d / %d \t Time Taken: %f sec \t Training Loss: %f' %
              (epoch, opts.total_epoch, time.time() - epoch_start_time, epoch_total_loss / len(train_dataloader)))
        if epoch % opts.show_tb_freq == 0:
            print('Training loss:',epoch_total_loss / len(train_dataloader))
            total_val = 0
            for i, data in enumerate(val_dataloader):
                model.set_input(data)
                total_val += model.validate()
            print('Validating loss:', total_val / len(val_dataloader))
            print('prob:', F.softmax(model.pred_labels, 1))
            print("Target:", model.target_label)
            print("Predic:", F.softmax(model.pred_labels, 1).argmax(1))
            write_loss_on_tb(train_writer, epoch_total_loss / len(train_dataloader),
                             total_val / len(val_dataloader), epoch)
        if epoch % opts.save_checkpoint_freq == 0:
            print('Save models at epoch %d' % (epoch))
            model.save_model(epoch)




