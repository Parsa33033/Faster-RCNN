# Faster-RCNN
A very simple faster rcnn model

dependancies:
1) python 3.6
2) tensorflow gpu 1.12.0
3) cuda 9.0
4) cuDNN 7.1.4

# Dataset
the dataset is derived from stanford obejct detection dataset for cars. Here is the [link](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) to the dataset.
the dataset contains
1) training set containing 8,144 training images
2) test set containing 8,041 testing images
3) devkit containing the bounding boxes and class labels as a .mat file

note: the .mat files have been converted to dictionaries. cars_train and cars_test are dictionaries containing image jpg file names for training set and test set, consecutively. The cars_train_annos and cars_test_annos refer to both training set and test set bounding boxes with keys as jpg names of the images and values as list of bounding boxes of that particular image corresponding the key. Each bounding box is a list consisting of xmin, ymin, xmax, ymax values of the bounding box.

**for tutorial of Faster RCNN please refer to [Link](https://medium.com/@parsa_h_m/faster-rcnn-a-survey-f32380cdd7ed)**

# Faster RCNN model
![Image](https://github.com/Parsa33033/Faster-RCNN/blob/master/FasterRCNN-Model.png)
