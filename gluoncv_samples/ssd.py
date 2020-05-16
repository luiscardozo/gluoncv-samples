"""
Object Detection with Faster RCNN pretrained model
Based on https://gluon-cv.mxnet.io/build/examples_detection/demo_ssd.html
"""

from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)

im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/street_small.jpg?raw=true',
                          path='street_small.jpg')

x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
print('Shape of preprocessed image: ', x.shape)

class_IDs, scores, bounding_boxes = net(x)

ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0],
                        class_IDs[0], class_names=net.classes)

#plt.savefig('img.png')
plt.show()
