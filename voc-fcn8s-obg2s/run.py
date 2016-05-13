import numpy as np
import matplotlib.pyplot as plt
import pdb

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

import os
import sys
caffe_root = '/home/xujiu01/library/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe

caffe.set_mode_gpu()
caffe.set_device(2)


model_def = 'deploy.prototxt'
model_weights = 'voc-fcn8s-obg2s.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
                
mu = np.array([104.00698793, 116.66876762, 122.67891434])
print 'mean values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

image = caffe.io.load_image('2007_000129.jpg')

transformed_image = transformer.preprocess('data', image)
plt.imshow(image)
# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

mask = net.blobs['score'].data[0]

classed = np.argmax(mask, axis=0)

names = dict()
all_labels = ["0", "1","2","3", "4","5", "6","7","8", "9","10", "11","12","13", "14","15", "16","17","18", "19","20"]
scores = np.unique(classed)
labels = [all_labels[s] for s in scores]
num_scores = len(scores)

def rescore (c):
    """ rescore values from original score values (0-59) to values ranging from 0 to num_scores-1 """
    return np.where(scores == c)[0][0]
rescore = np.vectorize(rescore)

painted = rescore(classed)
plt.imshow(painted, cmap=plt.cm.get_cmap('jet', num_scores))

# setup legend
formatter = plt.FuncFormatter(lambda val, loc: labels[val])
plt.colorbar(ticks=range(0, num_scores), format=formatter)
plt.clim(-0.5, num_scores - 0.5)
plt.show()
