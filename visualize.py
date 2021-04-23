import h5py
from glob import glob
from matplotlib import pyplot as plt
import argparse
import numpy as np

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('filename')
    p.add_argument('--start',               default=0,      type=float, help='Start secs')
    p.add_argument('--length',              default=10,     type=int,   help='Length of window')
    p.add_argument("--bsizes",              default=None,   type=float, help='Bin sizes', nargs='+')
    p.add_argument('--mode',                default='plot', type=str,   help='Mode', choices=['plot','analyze'])
    args = p.parse_args()

    model_dir = './exports/'

    query = model_dir + args.filename + '*.hdf5'
    print('querying ' + query)

    export_data = {}

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
                export_data[binsize] = data
    print()

    cols = args.length
    rows = len(export_data)

    for row, (binsize, data) in enumerate(export_data.items()):
        start = int(args.start / binsize)
        indexes = range(start, start + args.length)

        for i in indexes:
            ax = plt.subplot(rows, cols, (i-start)+1+(row*cols))
            plt.xticks([], [])
            plt.yticks([], [])
            
            ax.set_title('frame {}'.format(i))
            if (i-start) == 0:
                plt.ylabel("{}s".format(binsize))

            img = data[i]

            if args.mode == 'analyze':
                print(binsize, i, img.shape, np.unique(img), np.mean(img))
            
            ax.imshow(img, cmap='gray')

    if args.mode == 'plot':
        plt.show()