# Program to convert the depth image data to ndarray data

# In this program we use UTKinect-Action dataset. This dataset was captured using
# a stationary Kinect sensor. It consists of 10 actions performed by
# 10 different subjects. Each subject performed every action twice.

from skimage import io
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
%matplotlib inline

from os import listdir
from os.path import isfile, join
import os
import pickle
import gzip

# The directory name where data is found
path_dir = '/Users/leeqdew/Desktop/skeletal_action_recognition_code/data/UTKinect'
action_label = 'actionLabel.txt'

# list to save data per subject
subjects = []

n_actions = 0
with open(join(path_dir, action_label)) as f:
    content = f.readlines()
for x in content:
    if ':' in x:
        x_split = x.rstrip('\n').split()
        if 'NaN' in x:
            sj.append((-1, -1)) # -1 indicates this action is invalid
        else:
            # Image with its serial number falling between lo and hi belongs to this action
            lo = int(x_split[1])
            hi = int(x_split[2])
            sj.append((lo, hi))
            n_actions += 1
        if 'clapHands' in x:
            subjects.append(sj)
    else:
        sj = []

print(n_actions, 'actions in total')

# Count how many valid frames there are
n_frames = 0
for i in range(20):
    if i % 2 == 0:
        s = (i + 2) // 2
        if s < 10:
            subdir = 's0' + str(s) + '_e01'
        else:
            subdir = 's10_e01'
    else:
        s = (i + 1) // 2
        if s < 10:
            subdir = 's0' + str(s) + '_e02'
        else:
            subdir = 's10_e02'
    filepath = join(path_dir, 'depth', subdir)
    files = [ f for f in listdir(filepath) if isfile(join(filepath, f)) ]
    for x in files:
        n_id = int(x.split('.')[0][8:]) # Get image's serial number
        for j in range(10):
            lo, hi = subjects[i][j]
            if n_id >= lo and n_id <= hi: # This frame is valid, it belongs to some action
                n_frames += 1

print(n_frames, "valid frames in total")

# Allocate memory for frame data
X_train = np.zeros((n_frames, 1, 240, 320), dtype=np.uint16)
y_train = np.zeros((n_frames, 2), dtype=np.uint8)

# Load data from image into ndarray
idx = 0
for i in range(20):
    if i % 2 == 0:
        s = (i + 2) // 2
        if s < 10:
            subdir = 's0' + str(s) + '_e01'
        else:
            subdir = 's10_e01'
    else:
        s = (i + 1) // 2
        if s < 10:
            subdir = 's0' + str(s) + '_e02'
        else:
            subdir = 's10_e02'
    filepath = join(path_dir, 'depth', subdir)
    files = [ f for f in listdir(filepath) if isfile(join(filepath, f)) ]
    for x in files:
        img = io.imread(join(filepath, x))
        img_arr = np.array(img, dtype=np.uint16)
        n_id = int(x.split('.')[0][8:])
        for j in range(10):
            lo, hi = subjects[i][j]
            if n_id >= lo and n_id <= hi:
                X_train[idx][0] = img_arr
                y_train[idx][0] = j + 1
                if 'e01' in filepath:
                    y_train[idx][1] = 1
                else:
                    y_train[idx][1] = 2
                idx += 1

print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)

plt.imshow(X_train[-1][0], cmap=cm.gray)
print(y_train[-1][0], y_train[-1][1])

# Serialize data to file
data = []
data.append((X_train, y_train))
if not os.path.exists(join(path_dir, 'UTKinect.pkl.gz')):
    with gzip.open(join(path_dir, 'UTKinect.pkl.gz'), 'wb') as f:
        pickle.dump(data, f)
