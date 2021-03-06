# GluonCV Samples

Samples based on [GluonCV](https://gluon-cv.mxnet.io/tutorials/index.html), from where I want to elaborate more.

## Installation

```
# clone the repo
git clone https://github.com/luiscardozo/gluoncv-samples
cd gluoncv-samples

# create the Virtual Environment
python3 -m venv env
source env/bin/activate

# install the requirements
pip install -r requirements.txt
```

## Running

For now, the samples are the same of the website.
To run them, simply execute **python _file_** (inside the virtualenv).

For example, running `python gluoncv_samples/faster_rcnn.py` will download _biking.jpg_ and then show the classes it have found.

![biking.jpg](./docs/img/biking.jpg "Original image")
![biking_classes.png](./docs/img/biking_classes.png "Image with the found classes")

