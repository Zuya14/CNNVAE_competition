import argparse

# import math
import numpy as np
# import random
import os
import datetime
# from concurrent import futures

from Lidar import LidarDatasets


# from StateBuffer import StateMem, StateBuffer
# from EpisodeMemory import Episode, EpisodeMemory
# from lidar_util import imshowLocalDistance

# import pybullet as p
# import gym
# import cv2

import torch
from torchvision.utils import save_image

from model.NNVAE_trainer import NNVAE_trainer
import plot_graph

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train')

    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument("--hiddens", nargs="*", type=int, default=[512, 256])
    parser.add_argument("--latent", type=int, default=128)
    parser.add_argument('--batchnorm', action='store_true')
    parser.add_argument('--adabelief', action='store_true')
    parser.add_argument('--warmup', action='store_true')

    parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
    parser.add_argument('--id', type=str, default='')

    args = parser.parse_args()

    s_time = datetime.datetime.now()

    print("start:", s_time)

    ''' ---- Initialize ---- '''

    print("load dataset")

    train_filenames = ['./data/vaernnEnv0/id-{}.npy'.format(id) for id in range(10)]
    test_filenames  = ['./data/vaernnEnv0/id-{}.npy'.format(id) for id in range(10, 12)]

    lidarTrainDatasets = LidarDatasets(train_filenames)

    lidarTestDatasets = LidarDatasets(test_filenames)

    train_loader = torch.utils.data.DataLoader(lidarTrainDatasets, batch_size = args.batch_size, shuffle = True,  num_workers=2, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(lidarTestDatasets,  batch_size = args.batch_size, shuffle = False, num_workers=2, pin_memory=True)

    ''' ---- Train ---- '''

    print("Train", datetime.datetime.now())

    out_dir = './result-NNVAE' 

    if args.id != '':
        out_dir += '/' + args.id

    out_dir += '/hiddens'

    for hidden in args.hiddens:
        out_dir += '_{}'.format(hidden)
    out_dir += '_latent_{}'.format(args.latent)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    vae_train = NNVAE_trainer(1081, args.hiddens, args.latent, bn_flag=args.batchnorm, adaBelief=args.adabelief, device=device)

    if args.models is not '' and os.path.exists(args.models):
        vae_train.load(args.models)

    train_plot_data = plot_graph.Plot_Graph_Data(out_dir, 'train_loss', {'train_loss': [], 'mse_loss': [], 'KLD_loss': []})
    test_plot_data  = plot_graph.Plot_Graph_Data(out_dir, 'test_loss',  {'test_loss': []})
    plotGraph = plot_graph.Plot_Graph([train_plot_data, test_plot_data])

    for epoch in range(1, args.epochs+1):

        if args.warmup:
            k = epoch/args.epochs
        else:
            k = 1.0

        mse_loss, KLD_loss = vae_train.train(train_loader, k)
        loss = mse_loss + KLD_loss

        test_loss = vae_train.test(test_loader, k)

        plotGraph.addDatas('train_loss', ['train_loss', 'mse_loss', 'KLD_loss'], [loss, mse_loss, KLD_loss])
        plotGraph.addDatas('test_loss', ['test_loss'], [test_loss])

        if epoch%10 == 0:
            vae_train.save(out_dir+'/vae.pth')

            plotGraph.plot('train_loss')
            plotGraph.plot('test_loss')

            print('epoch [{}/{}], loss: {:.4f} test_loss: {:.4f}, k: {}'.format(
                epoch + 1,
                args.epochs,
                loss,
                test_loss,
                k))       

        if epoch % (args.epochs//10) == 0:
            vae_train.save(out_dir+'/vae{}.pth'.format(epoch))
            
            vae_train.vae.eval()
            data = lidarTrainDatasets.data[:100].to(device, non_blocking=True).view(-1, 1, 1081)

            recon_x, mu, logvar = vae_train.vae(data)
            save_image(torch.cat([data.view(-1,1081), recon_x.view(-1,1081)], dim=1), '{}/result{}.png'.format(out_dir, epoch))
            
            print("save:epoch", epoch)

    e_time = datetime.datetime.now()

    print("start:", s_time)
    print("end:", e_time)
    print("end-start:", e_time-s_time)
