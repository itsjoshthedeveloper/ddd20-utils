import h5py
import argparse
import numpy as np
from tqdm import tqdm
import os

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('path',                              type=str,   help='Path of recording')
    args = p.parse_args()

    print('opening ' + args.path)
    with h5py.File(args.path, 'r') as f:

        def printinfo(name):
            print(name, f[name], f[name].dtype)

        f.visit(printinfo)