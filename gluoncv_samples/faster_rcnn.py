"""
Object Detection with Faster RCNN pretrained model
Based on https://gluon-cv.mxnet.io/build/examples_detection/demo_faster_rcnn.html
"""

import logging

from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)

logging.debug("downloading model...")
net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)

logging.debug("downloading image...")
im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/biking.jpg?raw=true',
                          path='biking.jpg')

logging.debug("formatting image...")
x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)

logging.debug("Infering...")
box_ids, scores, bboxes = net(x)

logging.debug("Plotting...")
ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)

logging.debug("showing...")
plt.show() #requires TkInter. Else: plt.savefig('image.png')
