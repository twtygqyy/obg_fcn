# OBG-FCN

This repository is to reproduce the implementation of 'Object Boundary Guided Semantic Segmentation' in
http://arxiv.org/abs/1603.09742

    Object Boundary Guided Semantic Segmentation
    Qin Huang, Chunyang Xia, Wenchao Zheng, Yuhang Song, Hao Xu, C.-C. Jay Kuo
    arXiv:1603.09742

the paper claimed to achieve 87.5% mean IU in PASCAL VOC 2011 validation set with only the training images of VOC 2011 training set.

The code is based on the repository of https://github.com/shelhamer/fcn.berkeleyvision.org, which contains the offical code for the [paper](http://www.cs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf):

    Fully Convolutional Models for Semantic Segmentation
    Jonathan Long*, Evan Shelhamer*, Trevor Darrell
    CVPR 2015
    arXiv:1411.4038

**The implementation is just for test and could not achieve result close to Object Boundary Guided Semantic Segmentation so far.**
Any suggestion is more than welcome

Mdoels are trained using extra data from [Hariharan et al.](http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html), but excluding SBD val.
Mdoels are tested using aug_val set by excluding the overlapping images in VOC train_val dataset.

Here is the result so far: 
* [FCN-32s sbd]: mean IU 0.601230112927 on aug_val
* [FCN-16s sbd]:  mean IU 0.623964674094 on aug_val
* [FCN-8s sbd]: mean IU 0.625525553796 on aug_val
* [FCN-16s OBG-8s sbd]: mean IU 0.628746446579 on aug_val
* [FCN-8s OBG-8s sbd]: mean IU 0.630523623869 on aug_val
* [FCN-8s OBG-4s sbd]: mean IU 0.593030120308 on aug_val
* [FCN-8s OBG-2s sbd]: mean IU 0.577085377376 on aug_val

model link: 
* [FCN-8s OBG-2s sbd]: voc-fcn8s-obg2s.caffemodel: https://drive.google.com/open?id=0B5i4atpKg9EcRU9rb1lwd1VnTlE
* [FCN-8s OBG-4s sbd]: voc-fcn8s-obg4s.caffemodel: https://drive.google.com/open?id=0B5i4atpKg9EcU3U5Xy05Tm5kX0U
* [FCN-8s OBG-8s sbd]: voc-fcn8s-obg8s.caffemodel: https://drive.google.com/open?id=0B5i4atpKg9EcMWJXcFl5MGdwQ2c


There must be major bugs in the implementation since the performace decreased when combining pool2 and pool1 for object boundary.