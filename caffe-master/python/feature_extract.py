import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '/home/xuran/caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

net = caffe.Classifier(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                       caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
net.set_phase_test()
net.set_mode_cpu()
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

scores = net.predict([caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')],'fc7')
scores=scores['fc7']
#for x in scores:
#    print x
#[(k, v.data.shape) for k, v in net.blobs.items()]
#[(k, v[0].data.shape) for k, v in net.params.items()]
# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
#feat = net.blobs['fc6'].data[4]
#for y in feat:
#    print y
#plt.imshow(net.deprocess('data', net.blobs['data'].data[4]))
filters = net.params['conv1'][0].data
#vis_square(filters.transpose(0,2,3,1))
