import subprocess
import argparse

p = argparse.ArgumentParser()
p.add_argument('--time',     default='10:00', type=str, help='Duration of each sbatch job')
args = p.parse_args()

DAY=( 'jul16/rec1500220388.hdf5', 'jul18/rec1500383971.hdf5', 'jul18/rec1500402142.hdf5', 'jul28/rec1501288723.hdf5', 'jul29/rec1501349894.hdf5', 'aug01/rec1501614399.hdf5', 'aug08/rec1502241196.hdf5', 'aug15/rec1502825681.hdf5', 'jul02/rec1499023756.hdf5', 'jul05/rec1499275182.hdf5', 'jul08/rec1499533882.hdf5', 'jul16/rec1500215505.hdf5', 'jul17/rec1500314184.hdf5', 'jul17/rec1500329649.hdf5', 'aug05/rec1501953155.hdf5' ) # Day data
NIGHT=( 'jul09/rec1499656391.hdf5', 'jul09/rec1499657850.hdf5', 'aug01/rec1501649676.hdf5', 'aug01/rec1501650719.hdf5', 'aug05/rec1501994881.hdf5', 'aug09/rec1502336427.hdf5', 'aug09/rec1502337436.hdf5', 'jul01/rec1498946027.hdf5', 'aug01/rec1501651162.hdf5', 'jul02/rec1499025222.hdf5', 'aug09/rec1502338023.hdf5', 'aug09/rec1502338983.hdf5', 'aug09/rec1502339743.hdf5', 'jul01/rec1498949617.hdf5', 'aug12/rec1502599151.hdf5' ) # Night data
# DAY_NIGHT=( 'jul09/rec1499656391.hdf5', 'jul09/rec1499657850.hdf5', 'aug01/rec1501649676.hdf5', 'aug01/rec1501650719.hdf5', 'aug05/rec1501994881.hdf5', 'aug09/rec1502336427.hdf5', 'aug09/rec1502337436.hdf5', 'jul01/rec1498946027.hdf5', 'aug01/rec1501651162.hdf5', 'jul02/rec1499025222.hdf5', 'aug09/rec1502338023.hdf5', 'aug09/rec1502338983.hdf5', 'aug09/rec1502339743.hdf5', 'jul01/rec1498949617.hdf5', 'aug12/rec1502599151.hdf5', 'jul16/rec1500220388.hdf5', 'jul18/rec1500383971.hdf5', 'jul18/rec1500402142.hdf5', 'jul28/rec1501288723.hdf5', 'jul29/rec1501349894.hdf5', 'aug01/rec1501614399.hdf5', 'aug08/rec1502241196.hdf5', 'aug15/rec1502825681.hdf5', 'jul02/rec1499023756.hdf5', 'jul05/rec1499275182.hdf5', 'jul08/rec1499533882.hdf5', 'jul16/rec1500215505.hdf5', 'jul17/rec1500314184.hdf5', 'jul17/rec1500329649.hdf5', 'aug05/rec1501953155.hdf5' ) # Day + Night data

files = {'day': DAY, 'night': NIGHT}

i = 0
for time_of_day, file_list in files.items():
    for f in file_list:
        rec = f[:-5]
        cmd = 'sbatch --job-name={} --time={} create.sh {} {}'.format(rec, args.time, f, time_of_day)
        print(cmd)
        subprocess.run(cmd.split())
        i += 1