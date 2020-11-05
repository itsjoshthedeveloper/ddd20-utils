from __future__ import print_function
import os, sys, time, argparse
import h5py
import numpy as np
from collections import defaultdict
from hdf5_deeplearn_utils import MultiHDF5VisualIterator, MultiHDF5EncoderDecoderVisualIterator
import torch
import torchvision
import torchvision.transforms as transforms
import Nets
import Nets_Spiking
import Nets_Spiking_BNTT
import fnmatch

def evaluate_encoder_decoder(file_path, h5fs_aps, h5fs_dvs, keys_aps, keys_dvs, timesteps = 20, batch_size = 16, exp_id = "default_exp"):
    data_iterator = MultiHDF5EncoderDecoderVisualIterator()
    model_args = {'timesteps': args.timesteps,
                  'img_size': 80,
                  'inp_maps': 2,
                  'num_cls': 1,
                  'inp_type': 'dvs',
                  'encoder_decoder': True}
    model = Nets_Spiking_BNTT.SNN_VGG9_TBN(**model_args)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = torch.nn.MSELoss()

    train_loss = 0
    for data in data_iterator.flow(h5fs_aps, h5fs_dvs, keys_aps, keys_dvs, 'train_idxs', batch_size=int(args.batch_size/2), shuffle=True, seperate_dvs_channels = True):
        vid_aps, vid_dvs = data
        vid_aps_pred = model(torch.from_numpy(vid_dvs).to(device))
        loss = loss_fn(vid_aps_pred, torch.from_numpy(vid_aps).to(device))
        train_loss += loss.item()/batch_size
    train_loss = np.sqrt(train_loss/num_test_batches)
    print("Encoder Decoder: Train Avg Error: {}".format(train_loss))
    test_loss = 0
    for data in data_iterator.flow(h5fs_aps, h5fs_dvs, keys_aps, keys_dvs, 'test_idxs', batch_size=int(args.batch_size/2), shuffle=True, seperate_dvs_channels = True):
        vid_aps, vid_dvs = data
        vid_aps_pred = model(torch.from_numpy(vid_dvs).to(device))
        loss = loss_fn(vid_aps_pred, torch.from_numpy(vid_aps).to(device))
        test_loss += loss.item()/batch_size
    test_loss = np.sqrt(test_loss/num_test_batches)
    print("Encoder Decoder: Test Avg Error: {}".format(test_loss))

def evaluate_ann(file_path, h5fs, keys, batch_size = 16, exp_id = "default_exp"):
    data_iterator = MultiHDF5VisualIterator()
    model = Nets.VGG9(num_channels = 1)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = torch.nn.MSELoss()

    train_loss = 0
    for data in data_iterator.flow(h5fs, keys, 'train_idxs', batch_size = batch_size, shuffle=True):
        vid_in, bY = data
        y_pred = aps_model(torch.from_numpy(vid_in).to(device))
        loss = loss_fn(y_pred, torch.from_numpy(bY).to(device))
        train_loss += loss.item()/batch_size
    train_loss = np.sqrt(train_loss/num_test_batches)
    print("ANN {}: Train Avg RMSE (degrees): {}".format(exp_id, train_loss))

    test_loss = 0
    for data in data_iterator.flow(h5fs, keys, 'test_idxs', batch_size = batch_size, shuffle=True):
        vid_in, bY = data
        y_pred = aps_model(torch.from_numpy(vid_in).to(device))
        loss = loss_fn(y_pred, torch.from_numpy(bY).to(device))
        test_loss += loss.item()/batch_size
    test_loss = np.sqrt(test_loss/num_test_batches)
    print("ANN {}: Test Avg RMSE (degrees): {}".format(exp_id, test_loss))

def main():
    parser = argparse.ArgumentParser(description='Train a driving network.')
    parser.add_argument('--batch_size',   default=32, type=int, help='Batch size.')
    parser.add_argument('--h5files_aps',    nargs='+', help='HDF5 File that has APS data.')
    parser.add_argument('--dataset_keys_aps',  nargs='+', default='aps_frame_80x80', help='Dataset key for APS.')
    parser.add_argument('--h5files_dvs_frames',    nargs='+', help='HDF5 File that has DVS data.')
    parser.add_argument('--dataset_keys_dvs_frames',  nargs='+', default='dvs_frame_80x80', help='Dataset key for DVS.')
    parser.add_argument('--h5files_combined_frames',    nargs='+', help='HDF5 File that has DVS data.')
    parser.add_argument('--dataset_keys_combined_frames',  nargs='+', default='dvs_frame_80x80', help='Dataset key for DVS.')
    parser.add_argument('--h5files_dvs_timesteps',    nargs='+', help='HDF5 File that has DVS data.')
    parser.add_argument('--dataset_keys_dvs_timesteps',  nargs='+', default='dvs_frame_80x80', help='Dataset key for DVS.')
    parser.add_argument('--h5files_dvs_encoded',    nargs='+', help='HDF5 File that has DVS data.')
    parser.add_argument('--dataset_keys_dvs_encoded',  nargs='+', default='encoded_frame_80x80', help='Dataset key for encoded DVS.')
    parser.add_argument('--h5files_combined_aps_dvs_enc_frames',    nargs='+', help='HDF5 File list containing APS and encoded DVS data.')
    parser.add_argument('--dataset_keys_combined_aps_dvs_enc_frames',  nargs='+', default='encoded_frame_80x80', help='Dataset key for APS and encoded DVS.')
    parser.add_argument('--timesteps', type=int, default=10, help='Number of timesteps to split')
    parser.add_argument('--model_dir', default='./saved_models', help='Directory of saved models')
    args = parser.parse_args()

    # ---- Read all h5f files ---- #
    h5fs_aps = [h5py.File(h5file, 'r') for h5file in args.h5files_aps]
    h5fs_dvs_frames = [h5py.File(h5file, 'r') for h5file in args.h5files_dvs_frames]
    h5fs_combined_frames = [h5py.File(h5file, 'r') for h5file in args.h5files_combined_frames]
    h5fs_dvs_timesteps = [h5py.File(h5file, 'r') for h5file in args.h5files_dvs_timesteps]
    h5fs_dvs_encoded = [h5py.File(h5file, 'r') for h5file in args.h5files_dvs_encoded]
    h5fs_combined_aps_dvs_encoded = [h5py.File(h5file, 'r') for h5file in args.h5files_combined_aps_dvs_enc_frames]

    num_train_batches_aps = int(np.ceil(float(np.sum([len(h5f['train_idxs']) for h5f in h5fs_aps]))/args.batch_size))
    num_test_batches_aps = int(np.ceil(float(np.sum([len(h5f['test_idxs']) for h5f in h5fs_aps]))/args.batch_size))
    num_train_batches_dvs_frames = int(np.ceil(float(np.sum([len(h5f['train_idxs']) for h5f in h5fs_dvs_frames]))/args.batch_size))
    num_test_batches_dvs_frames = int(np.ceil(float(np.sum([len(h5f['test_idxs']) for h5f in h5fs_dvs_frames]))/args.batch_size))
    num_train_batches_combined_frames = int(np.ceil(float(np.sum([len(h5f['train_idxs']) for h5f in h5fs_combined_frames]))/args.batch_size))
    num_test_batches_combined_frames = int(np.ceil(float(np.sum([len(h5f['test_idxs']) for h5f in h5fs_combined_frames]))/args.batch_size))
    num_train_batches_dvs_timesteps = int(np.ceil(float(np.sum([len(h5f['train_idxs']) for h5f in h5fs_dvs_timesteps]))/args.batch_size))
    num_test_batches_dvs_timesteps = int(np.ceil(float(np.sum([len(h5f['test_idxs']) for h5f in h5fs_dvs_timesteps]))/args.batch_size))
    num_train_batches_combined_aps_dvs_encoded = int(np.ceil(float(np.sum([len(h5f['train_idxs']) for h5f in h5fs_combined_aps_dvs_encoded]))/args.batch_size))
    num_test_batches_combined_aps_dvs_encoded = int(np.ceil(float(np.sum([len(h5f['test_idxs']) for h5f in h5fs_combined_aps_dvs_encoded]))/args.batch_size))
    assert (num_train_batches_aps == num_train_batches_dvs_frames)
    assert (num_test_batches_aps == num_test_batches_dvs_frames)
    assert (num_train_batches_aps == num_train_batches_dvs_timesteps)
    assert (num_test_batches_aps == num_test_batches_dvs_timesteps)

    for filename in os.listdir(args.model_dir):
        if fnmatch.fnmatch(filename, '*encoder_decoder*'):
            evaluate_encoder_decoder(os.path.join(args.model_dir, filename),
                                     h5fs_aps,
                                     h5fs_dvs_timesteps,
                                     args.dataset_keys_aps,
                                     args.dataset_keys_dvs_timesteps,
                                     timesteps = args.timesteps,
                                     batch_size = args.batch_size,
                                     exp_id = "Encoder Decoder")
        elif fnmatch.fnmatch(filename, '*ann_only_aps*'):
            evaluate_ann(os.path.join(args.model_dir, filename),
                         h5fs_aps,
                         args.dataset_keys_aps,
                         batch_size = args.batch_size,
                         exp_id = "Only APS")
        elif fnmatch.fnmatch(filename, '*ann_only_acc_dvs*'):
            evaluate_ann(os.path.join(args.model_dir, filename),
                         h5fs_dvs_frames,
                         args.dataset_keys_dvs_frames,
                         batch_size = args.batch_size,
                         exp_id = "Only DVS (accumulated)")
        elif fnmatch.fnmatch(filename, '*ann_combined_aps_and_acc_dvs*'):
            evaluate_ann(os.path.join(args.model_dir, filename),
                         h5fs_combined_frames,
                         args.dataset_keys_combined_frames,
                         batch_size = args.batch_size,
                         exp_id = "Combined APS and DVS (accumulated)")
        elif fnmatch.fnmatch(filename, '*ann_only_encoded_dvs*'):
            evaluate_ann(os.path.join(args.model_dir, filename),
                         h5fs_dvs_encoded,
                         args.dataset_keys_dvs_encoded,
                         batch_size = args.batch_size,
                         exp_id = "Only Encoded DVS")
        elif fnmatch.fnmatch(filename, '*ann_combined_aps_and_encoded_dvs*'):
            evaluate_ann(os.path.join(args.model_dir, filename),
                         h5fs_combined_aps_dvs_encoded,
                         args.dataset_keys_combined_aps_dvs_enc_frames,
                         batch_size = args.batch_size,
                         exp_id = "Combined APS and Encoded DVS")
        elif fnmatch.fnmatch(filename, '*ann_back_prop_encoder_only_dvs*'):
            pass
        elif fnmatch.fnmatch(filename, '*ann_back_prop_encoder_combined_aps_and_dvs*'):
            pass



if __name__ == '__main__':
    main()
