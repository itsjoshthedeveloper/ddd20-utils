import h5py
import argparse
import numpy as np
from tqdm import tqdm
import os

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('filename',                          type=str,   help='Filename of recording')
    p.add_argument('bsize',                             type=float, help='Bin size')
    p.add_argument('--min',         default=400,        type=int,   help='Min num of events in frame')
    p.add_argument('--max',         default=0.5,        type=float, help='Max num of events in frame (% of range of total events per frame in recording)')
    p.add_argument('--out_dir',     default='./ddd20_processed/', type=str, help='Path of output directory processed data')
    args = p.parse_args()

    export_dir = './exports/'
    channels = 2
    dvs_name = 'dvs_channels'

    export_data = {}

    path = '{}{}_export_clip_separate_bsize-{}.hdf5'.format(export_dir, args.filename, args.bsize) # TODO: get date and night/day info
    print('opening ' + path)
    with h5py.File(path, 'r') as f:
        for name, dataset in tqdm(f.items(), desc='reading data'):
            if name not in ('dvs_accum', 'dvs_split'):
                if name == dvs_name:
                    name = 'dvs_frame'
                export_data[name] = dataset.value
    num_frames = len(export_data['dvs_frame'])
    print('{}: found {} frames'.format(dvs_name, num_frames))

    counts = []
    for frame in tqdm(export_data['dvs_frame'], desc='counting frames'):
        counts.append(np.sum(frame))
    frame_intensity_range = np.ptp(counts)
    print('found range of {} [{}-{}]'.format(frame_intensity_range, np.min(counts), np.max(counts)))

    temp_data = []
    for frame in tqdm(export_data['dvs_frame'], desc='removing frames'):
        event_count = np.sum(frame)
        if event_count >= args.min and event_count <= (args.max * frame_intensity_range):
            temp_data.append(frame)
    num_removed = num_frames-len(temp_data)
    print('removed {}% ({} frames) with {} frames left'.format(round(num_removed/num_frames*100, 2), num_removed, len(temp_data)))

    export_data['dvs_frame'] = np.stack(temp_data)

    try:
        os.mkdir(args.out_dir)
    except OSError:
        pass

    path = '{}{}.hdf5'.format(args.out_dir, args.filename)
    with h5py.File(path, 'w') as f:
        for name, data in tqdm(export_data.items(), 'writing data'):
            f.create_dataset(name, data=data)
    print('saved preprocessed data to {}'.format(path))