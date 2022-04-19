import sys
import os
import argparse
import time
import datetime

from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

from dataset import load_fMRI_dataset
from model import GraphLearning

from utils import label_accuracy, label_cross_entropy, edge_emb


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=-1, help='Random seed.')  # 随机种子
    parser.add_argument('--dataset', type=str, default='fMRI', help='Dataset type fMRI')
    parser.add_argument('--data-path', type=str, default='/data/GL/split_data_augm', help='Path for data.')
    parser.add_argument('--out', type=str, default='out.txt', help='Output file.')
    parser.add_argument('--save-folder', type=str, default='logs', help='Where to save the trained model, leave '
                                                                        'empty to not save anything.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of samples per batch.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta factor for mean')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta factor for standard deviation')
    parser.add_argument('--lr-decay', type=int, default=200, help='After how many epochs to decay LR by a factor of '
                                                                  'gamma.')
    parser.add_argument('--weight-decay', type=float, default=0.07, help='L2 penalty of model')
    parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')
    parser.add_argument('--num-region', type=int, default=400, help='Number of region.')
    parser.add_argument('--timesteps', type=int, default=26, help='The number of time steps per region.')
    parser.add_argument('--timesteps-reserved', type=int, default=26, help='Length of time series data used to '
                                                                           'construct node features.')
    parser.add_argument('--graph-loss', type=str, default='', help='Add regularization loss for graph')
    parser.add_argument('--temp', type=float, default=0.5, help='Temperature for Gumbel softmax.')
    parser.add_argument('--encoder-hidden', type=int, nargs='+', default=256, help='Number of hidden units in encoder.')
    parser.add_argument('--reserved-hidden', type=int, nargs='+', default=256,
                        help='Number of hidden units in reserved '
                             'time series encoder.')
    parser.add_argument('--class-num', type=int, nargs='+', default=4, help='Number of categories in the final graph '
                                                                            'classification.')
    parser.add_argument('--node-labels', type=int, nargs='+', default=7, help='Number of categories of nodes.')
    parser.add_argument('--gat-dropout', type=int, nargs='+', default=0.6, help='Probability of gat dropout.')
    parser.add_argument('--mlp-dropout', type=int, nargs='+', default=0.5, help='Probability of mlp dropout.')
    parser.add_argument('--num-heads', type=int, nargs='+', default=3, help='Number of head for multi-head attention.')

    arguments = parser.parse_args()
    if arguments.seed == -1:
        arguments.seed = int.from_bytes(os.urandom(4), sys.byteorder)
    else:
        arguments.seed = arguments.seed
    return arguments


def train(epoch, best_val_loss, best_val_acc, args, params):
    train_loader, valid_loader, _ = params['loaders']
    model = params['model']
    optimizer = params['optimizer']
    scheduler = params['scheduler']
    model_file = params['model_file']
    node_class = params['node_class']
    receive = params['receive']
    send = params['send']

    t_start = time.time()

    acc_train = []
    ent_train = []
    loss_train = []

    model.train()

    train_iterable = tqdm(train_loader)
    for index, (data, labels) in enumerate(train_iterable):
        train_iterable.set_description("Epoch{:3} Train".format(epoch + 1))

        labels = F.one_hot(labels, num_classes=4).float()
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        output = model(data, node_class, receive, send, args.temp)
        loss_ent = label_cross_entropy(output, labels)
        label_acc, _ = label_accuracy(output, labels)
        loss = loss_ent

        loss.backward()
        optimizer.step()

        acc_train.append(label_acc.detach().item())
        ent_train.append(loss_ent.detach().item())
        loss_train.append(loss.detach().item())
    scheduler.step()

    acc_valid = []
    ent_valid = []
    loss_valid = []

    model.eval()

    valid_iterable = tqdm(valid_loader)
    for index, (data, labels) in enumerate(valid_iterable):
        valid_iterable.set_description("Epoch{:3} Valid".format(epoch + 1))
        labels = F.one_hot(labels, num_classes=4).float()
        data = data.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            output = model(data, node_class, receive, send, args.temp)
            loss_ent = label_cross_entropy(output, labels)
            label_acc, _ = label_accuracy(output, labels)
            loss = loss_ent

        acc_valid.append(label_acc.detach().item())
        ent_valid.append(loss_ent.detach().item())
        loss_valid.append(loss.detach().item())

    acc_train_m = np.mean(acc_train)
    ent_train_m = np.mean(ent_train)
    loss_train_m = np.mean(loss_train)

    acc_valid_m = np.mean(acc_valid)
    ent_valid_m = np.mean(ent_valid)
    loss_valid_m = np.mean(loss_valid)
    print('acc_train_m:{} ent_train_m:{} loss_train_m:{}'.format(acc_train_m, ent_train_m, loss_train_m))
    print('acc_valid_m:{}, ent_valid_m:{}, loss_valid_m:{}'.format(acc_valid_m, ent_valid_m, loss_valid_m))

    out_string = ''.join(('Epoch: {:04d}\n', 'acc_train: {:.6f}, ', 'ent_train: {:.6f}, ',
                          'loss_train: {:.6f}\n', 'acc_valid: {:.6f}, ',
                          'ent_valid: {:.6f}, ', 'loss_valid: {:.6f}, ', 'time: {:.4f}s'))
    print(out_string.format(epoch, acc_train_m, ent_train_m, loss_train_m,
                            acc_valid_m, ent_valid_m, loss_valid_m,
                            time.time() - t_start), file=log)
    print('--------------------------------\n', file=log)
    # if loss_valid_m < best_val_loss:
    if loss_valid_m < best_val_acc:
        torch.save(model.state_dict(), model_file)
        print('Best model so far, saving...', file=log)
    return loss_valid_m, acc_valid_m


def test(args, params):
    _, _, test_loader = params['loaders']
    model = params['model']
    model_file = params['model_file']
    node_class = params['node_class']
    receive = params['receive']
    send = params['send']

    acc_test = []
    ent_test = []
    loss_test = []

    model.eval()
    model.load_state_dict(torch.load(model_file))
    real_label = []
    pred_label = []
    same_count = 0
    for index, (data, labels) in enumerate(tqdm(test_loader)):
        labels = F.one_hot(labels, num_classes=4).float()
        data = data.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output = model(data, node_class, receive, send, args.temp)
            loss_ent = label_cross_entropy(output, labels)
            label_acc, count = label_accuracy(output, labels)
            same_count += count
            loss = loss_ent

        acc_test.append(label_acc.detach().item())
        ent_test.append(loss_ent.detach().item())
        loss_test.append(loss.detach().item())

    metric_dict = {}

    acc_test_m = np.mean(acc_test)
    ent_test_m = np.mean(ent_test)
    loss_test_m = np.mean(loss_test)

    metric_dict['acc_test'] = acc_test_m
    metric_dict['ent_test'] = ent_test_m
    metric_dict['loss_test'] = loss_test_m

    print('--------------------------------', file=log)
    print('------------Testing-------------', file=log)
    print('--------------------------------', file=log)
    print('acc_test: {:.10f}\n'.format(acc_test_m),
          'ent_test: {:.10f}\n'.format(ent_test_m),
          'loss_test: {:.10f}\n'.format(loss_test_m), file=log)
    print('--------------------------------')
    print('------------Testing-------------')
    print('--------------------------------')
    print('acc_test: {:.10f}\n'.format(acc_test_m),
          'ent_test: {:.10f}\n'.format(ent_test_m),
          'loss_test: {:.10f}\n'.format(loss_test_m))
    return metric_dict, real_label, pred_label


if __name__ == "__main__":

    args = parse()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    time1 = datetime.datetime.now() + datetime.timedelta(hours=8)
    times = time1.isoformat().replace(':', '-')
    save_folder = '{}/{}/'.format('logs', times)
    os.makedirs(save_folder)
    encoder_file = os.path.join(save_folder, 'GAT_encoder.pt')
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'a')
    print(str(args) + '\n', file=log)
    pickle.dump({'args': args}, open(meta_file, "wb"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaders = load_fMRI_dataset(fMRI_root=args.data_path, batch_size=args.batch_size, shuffle=True)

    node_class = pd.read_excel("/data/GL/node_class.xlsx", header=0, usecols={'YeoNetwork': int})
    node_class = torch.from_numpy(node_class.values)
    node_class = F.one_hot(node_class, num_classes=7).float().squeeze(1)
    node_class = node_class.to(device)

    model = GraphLearning(args.timesteps, args.timesteps_reserved, args.encoder_hidden, args.class_num,
                          args.reserved_hidden, args.node_labels, args.num_region, args.gat_dropout, args.num_heads,
                          args.mlp_dropout)


    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)
    receive, send = edge_emb(args.num_region)

    model.to(device)
    receive = receive.to(device)
    send = send.to(device)


    param_dict = dict()
    param_dict.update({
        'save_folder': save_folder,
        'model_file': encoder_file,
        'loaders': loaders,
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'node_class': node_class,
        'receive': receive,
        'send': send
    })

    best_val_loss = np.inf
    best_val_acc = 0
    best_epoch = 0
    for epoch in range(args.epochs):

        val_loss, val_acc = train(epoch, best_val_loss, best_val_acc, args, param_dict)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            print('best epoch:{},best val_loss:{}\n'.format(epoch, val_loss), file=log)

    print(best_val_acc)
    print("Optimization Finished!", file=log)
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    print("Best Epoch: {:04d}".format(best_epoch + 1))

    metric_dict, real_l, pred_l = test(args, param_dict)

    print('weight_decay:{}'.format(args.weight_decay))
    print('weight_decay:{}'.format(args.weight_decay), file=log)
    log.flush()
    if log is not None:
        print(save_folder)
        log.close()
