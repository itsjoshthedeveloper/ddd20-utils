import h5py
from glob import glob
from matplotlib import pyplot as plt
import argparse
import numpy as np

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('filenames', nargs='+')
    p.add_argument('--start',       default=0,      type=float, help='Start time in secs', nargs='+')
    p.add_argument('--length',      default=10,     type=int,   help='Length of window')
    p.add_argument("--bsizes",      default=None,   type=float, help='Bin sizes', nargs='+')
    p.add_argument('--mode',        default='plot', type=str,   help='Mode', choices=['plot','analyze'])
    args = p.parse_args()

    if len(args.filenames) > 1 and len(args.bsizes) > 1:
        raise RuntimeError('If using multiple recordings, you must use one binsize')

    if len(args.filenames) == 1 and len(args.start) > 1:
        raise RuntimeError('If using one recording, you must use one start time')
    elif len(args.filenames) > 1 and len(args.start) != 1 and len(args.start) != len(args.filenames):
        raise RuntimeError('If using multiple recordings, you must use one start time or a start time for each recording')

    model_dir = './exports/'

    export_data = {}

    for filename in args.filenames:

        query = model_dir + filename + '*.hdf5'
        print('querying ' + query, end=' ', flush=True)

        temp_data = {}

        files = sorted(glob(query))
        for path in files:
            binsize = float(path[(path.find('bsize-')+len('bsize-')):(-1*len('.hdf5'))])
            if binsize in args.bsizes:
                with h5py.File(path, 'r') as f:
                    data = f['dvs_accum'].value
                    if args.mode == 'analyze':
                        print(binsize, data.shape)
                    else:
                        print(binsize, end=' ', flush=True)
                    temp_data[binsize] = data

        export_data[filename] = temp_data
        print()

    multi_files = len(export_data) > 1

    cols = args.length
    rows = len(export_data) if multi_files else len(args.bsizes)

    for j, (filename, rec_data) in enumerate(export_data.items()):
        for k, (binsize, data) in enumerate(rec_data.items()):
            start = (args.start[j] if len(args.start) > 1 else args.start[0])
            start_i = int(start / binsize)
            indexes = range(start_i, start_i + args.length)
            row = j if multi_files else k

            for i in indexes:
                ax = plt.subplot(rows, cols, (i-start_i)+1+(row*cols))
                plt.xticks([], [])
                plt.yticks([], [])
                
                ax.set_title('frame {}'.format(i))
                if (i-start_i) == 0:
                    if multi_files:
                        plt.ylabel("{}\n[{}s] ({}s)".format(filename, binsize, start), rotation=0, labelpad=45)
                    else:
                        plt.ylabel("{}s ({}s)".format(binsize, start), rotation=0, labelpad=20)

                img = data[i]

                if args.mode == 'analyze':
                    print(filename, binsize, i, img.shape, np.unique(img), np.mean(img))

                np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.0f}'.format}, linewidth=36)
                plt.xlabel("{} ({:.6f})".format(np.unique(img), np.mean(img)), fontsize=4)
                np.set_printoptions(suppress=False, formatter=None, linewidth=75)

                ax.imshow(img, cmap='gray')

    if args.mode == 'plot':
        plt.show()