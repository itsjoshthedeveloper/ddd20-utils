
Software released as part of the publication

* Hu, Y., Binas, J., Neil, D., Liu, S.-C., and Delbruck, T. (2018).  "DDD20 End-to-End Event Camera Driving Dataset: Fusing Frames andEvents with Deep Learning for Improved Steering Prediction". Submitted to Special session Beyond Traditional Sensing for Intelligent Transportation, The 23rd IEEE International Conference on Intelligent Transportation Systems, September 20 – 23, 2020, Rhodes, Greece

 * Binas, J., Neil, D., Liu, S.-C., and Delbruck, T. (2017). DDD17: End-To-End DAVIS Driving Dataset. in ICML’17 Workshop on Machine Learning for Autonomous Vehicles (MLAV 2017) (Sydney, Australia).  Available at: arXiv:1711.01458 [cs]  http://arxiv.org/abs/1711.01458 

See https://sites.google.com/view/davis-driving-dataset-2020/home for details.

Note: the software has been tested with python 2.7, support for newer versions will follow.


# Prerequisites

These tools require
 * openCV (pip install opencv-python),
 * h5py (pip install h5py).


# Usage:

## viewing

### Play a file from the beginning
$ python view.py <recorded_file.hdf5>

### Play a file, starting at X percent
$ python view.py <recorded_file.hdf5> X%

### Play a file starting at second X
$ python view.py <recorded_file.hdf5> Xs


## Exporting to frame-based representation

$ python export.py [-h] [--tstart TSTART] [--tstop TSTOP] [--binsize BINSIZE]
                 [--update_prog_every UPDATE_PROG_EVERY]
                 [--export_aps EXPORT_APS] [--export_dvs EXPORT_DVS]
                 [--out_file OUT_FILE]
                 filename


# License

This software is released under the GNU LESSER GENERAL PUBLIC LICENSE Version 3.

