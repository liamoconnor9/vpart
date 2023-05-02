import numpy as np
import h5py
import os
path = os.path.dirname(os.path.abspath(__file__))
import sys



with h5py.File(path + '/checkpoints/checkpoints_s1.h5', mode='r') as file:
    udata = file['tasks']['u']
    print(udata.shape)
     