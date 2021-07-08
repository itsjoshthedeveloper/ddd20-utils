# DDD20 Pre-processed Dataset

A spreadsheet detailing the attributes of each recording can be found [here](https://docs.google.com/spreadsheets/d/1k-Gm74Crad_ip9V6S9RFdLQW4hC0nHdlMVdRJZrXy30/edit?usp=sharing).

The following steps were taken to transform the original event data into standardized frames.

1. Sum the number of positive and negative events at each pixel within 25ms
2. Stack each accumulated event channel into a frame
3. Clip values in frame to [0,10]
4. Repeat steps 1-3 until no more events and all frames are generated
5. Count the sum of pixel values for all frames
6. Calculate the range of sums
7. Remove all frames which have a sum less than 400 events and greater than half of the range of sums

The data is organized in the following manner:

```
ddd20_processed/
├── day
│   ├── aug01
│   │   └── rec1501614399.hdf5
│   ├── aug05
│   │   └── rec1501953155.hdf5
│   ├── aug08
│   │   └── rec1502241196.hdf5
│   ├── aug15
│   │   └── rec1502825681.hdf5
│   ├── jul02
│   │   └── rec1499023756.hdf5
│   ├── jul05
│   │   └── rec1499275182.hdf5
│   ├── jul08
│   │   └── rec1499533882.hdf5
│   ├── jul16
│   │   ├── rec1500215505.hdf5
│   │   └── rec1500220388.hdf5
│   ├── jul17
│   │   ├── rec1500314184.hdf5
│   │   └── rec1500329649.hdf5
│   ├── jul18
│   │   ├── rec1500383971.hdf5
│   │   └── rec1500402142.hdf5
│   ├── jul28
│   │   └── rec1501288723.hdf5
│   └── jul29
│       └── rec1501349894.hdf5
└── night
    ├── aug01
    │   ├── rec1501649676.hdf5
    │   ├── rec1501650719.hdf5
    │   └── rec1501651162.hdf5
    ├── aug05
    │   └── rec1501994881.hdf5
    ├── aug09
    │   ├── rec1502336427.hdf5
    │   ├── rec1502337436.hdf5
    │   ├── rec1502338023.hdf5
    │   ├── rec1502338983.hdf5
    │   └── rec1502339743.hdf5
    ├── aug12
    │   └── rec1502599151.hdf5
    ├── jul01
    │   ├── rec1498946027.hdf5
    │   └── rec1498949617.hdf5
    ├── jul02
    │   └── rec1499025222.hdf5
    └── jul09
        ├── rec1499656391.hdf5
        └── rec1499657850.hdf5
```