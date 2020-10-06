from __future__ import print_function
import os, sys, time, argparse
import h5py
import numpy as np
from collections import defaultdict
from hdf5_deeplearn_utils import MultiHDF5VisualIterator
import torch
import torchvision
import torchvision.transforms as transforms
import Nets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trai a drving network.')
    # File and path naming stuff
    parser.add_argument('--h5files',    nargs='+', default='/home/dneil/h5fs/driving/rec1487864316_bin5k.hdf5', help='HDF5 File that has the data.')
    parser.add_argument('--run_id',       default='default', help='ID of the run, used in saving.')
    parser.add_argument('--filename',     default='driving_cnn_19.4_multi', help='Filename to save model and log to.')
    # Control meta parameters
    parser.add_argument('--seed',         default=42, type=int, help='Initialize the random seed of the run (for reproducibility).')
    parser.add_argument('--batch_size',   default=128, type=int, help='Batch size.')
    parser.add_argument('--num_epochs',   default=100, type=int, help='Number of epochs to train for.')
    parser.add_argument('--patience',     default=4, type=int, help='How long to wait for an increase in validation error before quitting.')
    parser.add_argument('--patience_key', default='test_acc', help='What key to look at before quitting.')
    parser.add_argument('--wait_period',  default=10, type=int, help='How long to wait before looking for early stopping.')
    parser.add_argument('--dataset_keys',  nargs='+', default='aps_frame_48x64', help='Which dataset key (APS, DVS, etc.) to use.')
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Set the save name
    comb_filename = '_'.join([args.filename, args.run_id])

    # Load dataset
    h5fs = [h5py.File(h5file, 'r') for h5file in args.h5files]

    # Create symbolic vars
    # vid_in = T.ftensor4('vid_in')
    # targets = T.fmatrix('targets')
    #network = Nets.VGG16()
    network = Nets.ResNet34()
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    network.to(device)

    # Precalc for announcing
    num_train_batches = int(np.ceil(float(np.sum([len(h5f['train_idxs']) for h5f in h5fs]))/args.batch_size))
    num_test_batches = int(np.ceil(float(np.sum([len(h5f['test_idxs']) for h5f in h5fs]))/args.batch_size))
    print(num_train_batches, num_test_batches)

    # Dump some debug data if we like
    print(network)
    temp = MultiHDF5VisualIterator()
    for data in temp.flow(h5fs, args.dataset_keys, 'train_idxs', batch_size=args.batch_size, shuffle=True):
        vid_in, bY = data
        break
    print("Input Shape: {}, Output Shape: {}".format(vid_in.shape, bY.shape))
    # print(vid_in, bY)

    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    for t in range(args.num_epochs):
        train_loss = 0
        for data in temp.flow(h5fs, args.dataset_keys, 'train_idxs', batch_size=args.batch_size, shuffle=True):
            vid_in, bY = data
            y_pred = network(torch.from_numpy(vid_in).to(device))
            # print("y", bY)
            # print("y_pred", y_pred)
            loss = loss_fn(y_pred, torch.from_numpy(bY).to(device))
            # print(t, loss.item())
            train_loss += loss.item()/args.batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = np.sqrt(train_loss/num_train_batches)
        print("Epoch: {}, Train Avg RMSE: {}".format(t, train_loss))
        test_loss = 0
        for data in temp.flow(h5fs, args.dataset_keys, 'test_idxs', batch_size=args.batch_size, shuffle=True):
            vid_in, bY = data
            y_pred = network(torch.from_numpy(vid_in).to(device))
            # print("y", bY)
            # print("y_pred", y_pred)
            loss = loss_fn(y_pred, torch.from_numpy(bY).to(device))
            # print(t, loss.item())
            test_loss += loss.item()/args.batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        test_loss = np.sqrt(test_loss/num_test_batches)
        print("Epoch: {}, Test Avg RMSE: {}".format(t, test_loss))

