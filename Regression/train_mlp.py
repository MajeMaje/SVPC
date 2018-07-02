#!/usr/bin/env python
"""Chainer example: train a VAE on MNIST
"""
import argparse
import os

import chainer
from chainer import cuda
import numpy as np

import net
import set_data_cpu
import cv2
import pandas as pd

from sklearn.model_selection import KFold
seed = 1024


def main():
    parser = argparse.ArgumentParser(description='Multi perceptron')
    parser.add_argument('--gpu', '-g', default= 0, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    parser.add_argument('--epoch', '-e', default=200, type=int, help='number of epochs to learn')
    parser.add_argument('--dimz', '-z', default=100, type=int, help='dimention of encoded vector')
    parser.add_argument('--batchsize', '-b', type=int, default=256, help='learning minibatch size')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# dim z: {}'.format(args.dimz))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')
        
    #出力用フォルダの作成
    if not os.path.exists(args.out):
        os.mkdir(args.out)

    # Load dataset
    X_test,X_train,Y_train = set_data_cpu.set_data()
    print("train shape",X_train.shape) 
    print("tests shape",X_test.shape)   
    

    # cross Validation
    batchsize = args.batchsize
    kf = KFold(5, True, seed)
    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
        # Prepare VAE model, defined in net.py
        model = net.MLP(n_in=int(X_train.shape[1]))
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()
            model.to_gpu()

        # Setup an optimizer
        optimizer = chainer.optimizers.SGD(lr=0.01)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.05))
        
        # split data
        x_train, x_valid = X_train[train_idx], X_train[valid_idx]
        y_train, y_valid = Y_train[train_idx], Y_train[valid_idx]
        x_train, x_valid = np.asarray(x_train).astype(np.float32), np.asarray(x_valid).astype(np.float32)
        y_train, y_valid = np.asarray(y_train).astype(np.float32), np.asarray(y_valid).astype(np.float32)
        train_datanum, val_datanum = x_train.shape[0],  x_valid.shape[0]
        print("valnum:{}===========".format(fold_idx))
        for epoch in range(1,args.epoch+1):
            #学習
            perm = np.random.permutation(train_datanum)#0~datanumの数のランダムな並び替え
            sum_loss =0
            for i in range(0,train_datanum,batchsize):
                x_batch = chainer.Variable(cuda.to_gpu(x_train[perm[i:i + batchsize]]))
                y_batch = chainer.Variable(cuda.to_gpu(y_train[perm[i:i + batchsize]]))
                model.zerograds()
                loss = model(x_batch,y_batch)
                sum_loss += float(model.loss.data) * len(x_batch.data)
                loss.backward()
                optimizer.update()
            average_loss = sum_loss / train_datanum
            print ("")
            print ("train:  epoch: {} ,loss: {} ".format( epoch, average_loss))
            #評価（validation ）
            sum_loss =0
            for i in range(0,val_datanum,batchsize):
                x_batch = chainer.Variable(cuda.to_gpu(x_valid[i:i + batchsize]))
                y_batch = chainer.Variable(cuda.to_gpu(y_valid[i:i + batchsize]))
                with chainer.using_config('train', False):
                    loss = model(x_batch,y_batch)
                sum_loss += float(model.loss.data) * len(x_batch.data)
            average_loss = sum_loss / val_datanum
            print ("validation:  epoch: {} ,loss: {} ".format( epoch, average_loss))

            #学習率の低下  
            optimizer.lr = optimizer.lr*0.995   


    # Save the model and the optimizer
    print('save the model')
    chainer.serializers.save_npz(
        os.path.join(args.out, 'mlp.model'), model)
    print('save the optimizer')
    chainer.serializers.save_npz(
        os.path.join(args.out, 'mlp.state'), optimizer)

   

if __name__ == '__main__':
    main()