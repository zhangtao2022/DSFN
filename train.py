# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torch.utils.data.distributed import DistributedSampler
from models.classification_heads import ClassificationHead
from models.R2D2_embedding import R2D2Embedding
from models.protonet_embedding import ProtoNetEmbedding
from models.ResNet12_embedding import resnet12
from models.wideres_embedding import wideresEmbedding
import torch.nn as nn
from utils import set_gpu, Timer, count_accuracy, check_dir, log


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
    elif options.network == 'R2D2':
        network = R2D2Embedding().cuda()
    elif options.network == 'ResNet':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5).cuda()
            network = torch.nn.DataParallel(network, device_ids=[0,1,2,3])
        else:
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2).cuda()
    elif options.network == "wideres":
        network = wideresEmbedding(remove_linear=True).cuda()
    else:
        print ("Cannot recognize the network type")
        assert(False)
        
    # Choose the classification head
    if options.head == 'Subspace':
        cls_head = ClassificationHead(base_learner='Subspace').cuda()
    elif options.head == 'Shrinkage':
        cls_head = ClassificationHead(base_learner='Shrinkage').cuda()
    elif options.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()
    elif options.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif options.head == 'R2D2':
        cls_head = ClassificationHead(base_learner='R2D2').cuda()
    elif options.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_train = MiniImageNet(phase='trainval') ## change it to train only, this is including the validation set
        dataset_val = MiniImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_train = tieredImageNet(phase='trainval')
        #dataset_train1 = DistributedSampler(dataset_train)
        dataset_val = tieredImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_train = CIFAR_FS(phase='trainval')
        dataset_val = CIFAR_FS(phase='test')
        data_loader = FewShotDataloader

    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_train, dataset_val, data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=80,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=5,
                            help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=15,
                            help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=6,
                            help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=10,
                            help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=600,
                            help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=5,
                            help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,
                            help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=5,
                            help='number of classes in one test (or validation) episode')
    parser.add_argument('--save-path', default='./experiments/exp_1')
    #parser.add_argument('--gpu', type=int,  default=-1, help='GPU id to use.')
    parser.add_argument('--gpu', default='0,1,2,3') #using 4 gpus
    parser.add_argument('--network', type=str, default='ResNet',
                            help='choose which embedding network to use. ResNet')
    parser.add_argument('--head', type=str, default='Shrinkage',
                            help='choose which classification head to use. Subspace, Shrinkage, ProtoNet, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='CIFAR_FS',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                            help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')


    opt = parser.parse_args()
    print(opt)

    #torch.distributed.init_process_group(backend="nccl")
    (dataset_train, dataset_val, data_loader) = get_dataset(opt)

    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot, # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=4,
        epoch_size=opt.episodes_per_batch * 1000, # num of batches per epoch
    )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot, # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode, # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)
    
    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)

    if opt.network == "ConvNet":
        optimizer = torch.optim.Adam([{'params': embedding_net.parameters()},
                                      {'params': cls_head.parameters()}], lr=0.005)

        lambda_epoch = lambda e: 1.0 if e < 8 else (0.5 if e < 30 else 0.25 if e < 45 else (0.1 if e < 50 else (0.05)))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    else:
            optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                                         {'params': cls_head.parameters()}], lr=0.1, momentum=0.9,
                                        weight_decay=5e-4, nesterov=True)
            ## miniimagenet###
            lambda_epoch = lambda e: 1.0 if e < 12 else (0.025 if e < 30 else 0.0032 if e < 45 else (0.0014 if e < 57 else (0.00052)))

            ## tieredimagenet###
            #lambda_epoch = lambda e: 1.0 if e < 20 else (
            #0.012 if e < 45 else 0.0052 if e < 59 else (0.00054 if e < 68 else (0.00012)))

            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, opt.num_epoch + 1):

        
        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']
            
        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                            epoch, epoch_learning_rate))
        
        _, _ = [x.train() for x in (embedding_net, cls_head)]
        
        train_accuracies = []
        train_losses = []

        for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

            train_n_support = opt.train_way * opt.train_shot
            train_n_query = opt.train_way * opt.train_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)

            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot)

            smoothed_one_hot = one_hot(labels_query.reshape(-1), opt.train_way)
            smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (opt.train_way - 1)

            log_prb = F.log_softmax(logit_query.reshape(-1, opt.train_way), dim=1)
            loss = -(smoothed_one_hot * log_prb).sum(dim=1)
            loss = loss.mean()

            acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))

            train_accuracies.append(acc.item())
            train_losses.append(loss.item())

            if (i % 100 == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                            epoch, i, len(dloader_train), loss.item(), train_acc_avg, acc))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Train on the training split
        lr_scheduler.step()

        # Evaluate on the validation split
        _, _ = [x.eval() for x in (embedding_net, cls_head)]

        val_accuracies = []
        val_losses = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
                data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

                test_n_support = opt.test_way * opt.val_shot
                test_n_query = opt.test_way * opt.val_query

                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(1, test_n_support, -1)
                emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(1, test_n_query, -1)

                logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot)

                loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
                acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

                val_accuracies.append(acc.item())
                val_losses.append(loss.item())
            
        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},\
                       os.path.join(opt.save_path, 'best_model.pth'))
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

        torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                   , os.path.join(opt.save_path, 'last_epoch.pth'))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                       , os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))
