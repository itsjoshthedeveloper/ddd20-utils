import h5py
from glob import glob
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib import colors
import argparse
import numpy as np
from math import ceil
from tqdm import tqdm

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('filenames', nargs='+')
    p.add_argument('--bsizes',      default=[0.01], type=float, help='Bin sizes', nargs='+')
    p.add_argument('--mode',        default='pixel',    type=str,   help='Mode', choices=['pixel','frame'])
    p.add_argument('--channels',    default=1,          type=int,   help='Number of channels')
    p.add_argument('--separate',    action='store_true')
    p.add_argument('--clip',        action='store_true')
    args = p.parse_args()

    if len(args.filenames) > 1 and len(args.bsizes) > 1:
        raise RuntimeError('If using multiple recordings, you must use one binsize')

    if len(args.filenames) > 1 and args.separate:
        raise RuntimeError('If using separate DVS channels, you must use one recording')

    if args.separate and args.channels == 1:
        args.channels = 2

    model_dir = './exports/'

    export_data = {}

    for filename in args.filenames:

        query = '{}{}_export_{}{}bsize-*.hdf5'.format(model_dir, filename, ('clip_' if args.clip else ''), ('separate_' if args.separate else ''))
        print('querying ' + query, end=' ', flush=True)

        temp_data = {}

        files = sorted(glob(query))
        for path in files:
            binsize = float(path[(path.find('bsize-')+len('bsize-')):(-1*len('.hdf5'))])
            if binsize in args.bsizes:
                with h5py.File(path, 'r') as f:
                    export_key = 'dvs_channels' if args.separate else 'dvs_accum'
                    data = f[export_key].value
                    print('{}s[{}]'.format(binsize, len(data)), end=' ', flush=True)
                    temp_data[binsize] = data

        export_data[filename] = temp_data
        print()

    num_files = len(export_data)
    multi_files = num_files > 1
    hist = (not multi_files) and len(args.bsizes) == 1

    if hist and args.mode == 'frame':
        counts = [[]] * args.channels
    else:
        counts = [{}] * args.channels
    bins = 0

    if hist and args.mode == 'frame':
        data = export_data[args.filenames[0]][args.bsizes[0]]
        tqdm_desc = '{} [{}]'.format(args.filenames[0], args.bsizes[0])
        for i, ch_counts in enumerate(counts):
            maxsum = 0
            for frame in tqdm(data, desc=tqdm_desc):
                framesum = np.sum(frame if args.channels == 1 else frame[i])
                ch_counts.append(framesum)
                maxsum = framesum if framesum > maxsum else maxsum

        # print(len(counts))
        # bins = np.ptp(counts)//100
        # print('bins: ' + str(bins))
        # counts = iter(counts)
    else:
        for i, ch_counts in enumerate(counts):
            for j, (filename, rec_data) in enumerate(export_data.items()):
                ch_counts[filename] = {}

                for k, (binsize, data) in enumerate(rec_data.items()):
                    ch_counts[filename][binsize] = {}
                    
                    tqdm_desc = '{} [{}]'.format(filename, binsize)
                    for frame in tqdm(data, desc=tqdm_desc):
                        frame = frame if args.channels == 1 else frame[i]
                        if args.mode == 'pixel':
                            unique, pixel_counts = np.unique(frame, return_counts=True)
                            for val, count in zip(unique, pixel_counts):
                                if val not in ch_counts[filename][binsize]:
                                    ch_counts[filename][binsize][val] = 0
                                ch_counts[filename][binsize][val] += count
                        elif args.mode == 'frame':
                            val = np.sum(frame)
                            if val not in ch_counts[filename][binsize]:
                                ch_counts[filename][binsize][val] = 0
                            ch_counts[filename][binsize][val] += 1

                    if args.mode == 'pixel':
                        total = 260*346*len(data)
                    elif args.mode == 'frame':
                        total = len(data)

                    for val in ch_counts[filename][binsize]:
                        ch_counts[filename][binsize][val] = ch_counts[filename][binsize][val]/total
    print('finished counting')

    cols = len(counts) if len(counts) <= 2 else 2
    rows = 1 if len(counts) <= 2 else ceil(len(counts)/2)

    window_title = ('{}_{}'.format(args.filenames[0], args.bsizes[0]) if hist else args.bsizes[0] if multi_files else args.filenames[0]) + ('_separate' if args.separate else '')
    window_title = '{}_{}'.format(window_title, args.mode)

    fig, axs = plt.subplots(rows, cols, num=window_title, sharey=True)  # Create a figure and an axes.
    if type(axs) is not np.ndarray:
        axs = np.array([axs])

    if hist and args.mode == 'frame':
        for ch, ch_counts in enumerate(counts):
            N, bins, patches = axs[ch%2].hist(ch_counts, bins=40, density=True)
            # We'll color code by height, but you could use any scalar
            fracs = N / N.max()
            # we need to normalize the data to 0..1 for the full range of the colormap
            norm = colors.Normalize(fracs.min(), fracs.max())
            # Now, we'll loop through our objects and set the color of each accordingly
            for thisfrac, thispatch in zip(fracs, patches):
                color = plt.cm.viridis(norm(thisfrac))
                thispatch.set_facecolor(color)
    else:
        for i, ch_counts in enumerate(counts):
            for filename, binsizes in ch_counts.items():
                for binsize, cts in binsizes.items():
                    x, y = [], []

                    for val, count in sorted(cts.items()):
                        x.append(val)
                        y.append(count)

                    label = filename if multi_files else '{}s'.format(binsize)
                    axs[i].plot(x, y, label=label)  # Plot some data on the axes.
    
    for j in range(cols):
        axs[j].yaxis.set_major_formatter(PercentFormatter(1))
        axs[j].set_xlabel('Number of events in {}'.format(args.mode))  # Add an x-label to the axes.
        axs[j].set_ylabel('Frequency')  # Add a y-label to the axes.
        if multi_files:
            title = '{} event intensity frequency with binsize={}'.format(args.mode.capitalize(), args.bsizes[0])
        elif hist:
            if args.separate:
                title = '{} event intensity frequency for {} ({}s) [ch {}]'.format(args.mode.capitalize(), args.filenames[0], args.bsizes[0], j)
            else:
                title = '{} event intensity frequency for {} ({}s)'.format(args.mode.capitalize(), args.filenames[0], args.bsizes[0])
        else:
            title = '{} event intensity frequency for {}'.format(args.mode.capitalize(), args.filenames[0])
        axs[j].set_title(title) # Add a title to the axes.
        if not hist:
            axs[j].legend()  # Add a legend.

    plt.show()