import os
import torch.nn.init as init

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs_now(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def save_opts(opts):
    # save the options to the disk
    expr_dir = os.path.join(opts.output_path, opts.name)
    mkdirs_now(expr_dir)
    if opts.isTrain:
        file_name = os.path.join(expr_dir, 'train_opt.txt')
    else:
        file_name = os.path.join(expr_dir, 'test_opt.txt')
    args = vars(opts)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

def weight_init(init_type='orthogonal'):
    def init_fun(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data) # gain = 1
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 1.0)
        if hasattr(m,'_all_weights'):
            for nth in range(m.num_layers * m.bidirectional):
                # w_ih, (4 * hidden_size x input_size)
                init.orthogonal_(m._all_weights[nth][0], gain=1)
                # w_hh, (4 * hidden_size x hidden_size)
                init.orthogonal_(m._all_weights[nth][1], gain=1)
                # b_ih, (4 * hidden_size)
                init.zeros_(m._all_weights[nth][2])
                # b_hh, (4 * hidden_size)
                init.zeros_(m._all_weights[nth][3])
    return init_fun

def write_loss_on_tb(train_writer,train_loss, val_loss, iteration):
    train_writer.add_scalar('train_loss', train_loss, iteration)
    train_writer.add_scalar('val_loss', val_loss, iteration)