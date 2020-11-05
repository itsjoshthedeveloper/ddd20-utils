from __future__ import print_function
import numpy as np
import h5py
import os, sys, time, argparse
from hdf5_deeplearn_utils import calc_data_mean, calc_data_std, build_train_test_split, check_and_fix_timestamps, resize_data_into_new_key, run_dvs_to_aps_into_new_key

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename')
    parser.add_argument('--dataset_key', default="dvs_frame_80x80")
    parser.add_argument('--pretrained_model_path', default="./saved_models/driving_cnn_19.4_multi_encoder_decoder")
    parser.add_argument('--train_length', default=5*60, type=float)
    parser.add_argument('--test_length', default=2*60, type=float)
    parser.add_argument('--rewrite', default=0, type=int)
    parser.add_argument('--new_height', default=80, type=int)
    parser.add_argument('--new_width', default=80, type=int)
    parser.add_argument('--timesteps', type=int, default=10)
    args = parser.parse_args()


    # Set new resize
    new_size = (args.new_height, args.new_width)

    dataset = h5py.File(args.filename, 'a')
    # print('Checking timestamps...')
    # check_and_fix_timestamps(dataset)

    print('Calculating train/test split...')
    sys.stdout.flush()
    build_train_test_split(dataset, train_div=args.train_length, test_div=args.test_length, force=args.rewrite)

    if np.any(dataset[args.dataset_key][0]):
        new_dvs_key = '{}_{}x{}'.format('encoded_frame', new_size[0], new_size[1])
        print('Encoding DVS frames to {}...'.format(new_dvs_key))
        sys.stdout.flush()
        start_time = time.time()
        run_dvs_to_aps_into_new_key(dataset, args.dataset_key, new_dvs_key, new_size, args.pretrained_model_path)
        print('Finished in {}s.'.format(time.time()-start_time))

    print('Done.  Preprocessing complete.')
    filesize = os.path.getsize(args.filename)
    print('Final size: {:.1f}MiB to {}.'.format(filesize/1024**2, args.filename))
