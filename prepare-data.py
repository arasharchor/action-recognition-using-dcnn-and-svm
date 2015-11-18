# Program to save data from depth image into numpy.ndarray
#
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

# The name of the path where data is found
path_dir = '/Users/leeqdew/Desktop/skeletal_action_recognition_code/data/UTKinect'
action_label = 'actionLabel_modified.txt'

def count_valid_actions(filename):
    ```Count how many valid actions from a specific file

    Keyword argument:

    filename -- file name specified by user

    Returns: (n_actions, subjects)

    ```
    n_actions = 0
    # subjects will be a list of list, its first and second element cover the actions performed
    # by the first subject, its third and fourth element cover the actions performed by
    # the second subject, etc.
    subjects = []
    with open(filename) as f:
        content = f.readlines()
    for x in content:
        if ':' in x:
            x_split = x.rstrip('\n').split()
            if 'NaN' in x: # This action is invalid, here we indicate that by -1
                sj.append((-1, -1))
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
    return (n_actions, subjects)

filename = join(path_dir, action_label)
n_actions, subjects = count_valid_actions(filename)
print(n_actions, 'actions in total')

X_train = []
y_train = np.zeros((n_actions, 2), dtype=np.uint8)


# Load data from image into numpy.ndarray
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
    # fileserial = open(subdir, 'w')
    filepath = join(path_dir, 'depth', subdir)
    files = [ f for f in listdir(filepath) if isfile(join(filepath, f)) ]
    file_sns = [ int(f.split('.')[0][8:]) for f in files ]
    file_sns.sort()
    for j in range(10):
        action = []
        lo, hi = subjects[i][j]
        if lo == -1:
            continue
        try:
            k = file_sns.index(lo)
        except ValueError:
            k = file_sns.index(lo + 2)
        while file_sns[k] <= hi:
            filename = join(filepath, 'depthImg' + str(file_sns[k]) + '.png')
            img = io.imread(filename)
            img_arr = np.array(img, dtype=np.uint16)
            action.append(img_arr)
            # fileserial.write(str(file_sns[k]))
            # fileserial.write('\n')
            k += 1
        X_train.append(action)
        y_train[idx][0] = j + 1
        if 'e01' in filepath:
            y_train[idx][1] = 1
        else:
            y_train[idx][1] = 2
        idx += 1
    # fileserial.close()

X_train = np.array(X_train)
print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)

X_train_copy = []
for action in X_train:
    action_copy = action[0].reshape(-1, order='F')
    for j in np.arange(1, len(action)):
        frame = action[j]
        frame = frame.reshape(-1, order='F')
        action_copy = np.vstack((action_copy, frame))
    X_train_copy.append(action_copy.T)

X_train = np.array(X_train_copy)

# Serialize data to file
data = []
data.append((X_train, y_train))
with gzip.open(join(path_dir, 'UTKinect.pkl.gz'), 'wb') as f:
    pickle.dump(data, f)
