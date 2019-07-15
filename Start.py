
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.misc import imresize, imread
import numpy as np
import re
from urllib.request import urlopen
from Faster_RCNN.Utils import show_image_w_bboxes, prepare_image_and_bboxes


class Start():
    """
    start contains a function that gets the image, image bounding boxes(not the ratio but the actual size of the bounding boxes)
    and image class names for each bounding box. the function should contain a notify at the end of the function and
    the function should be read in main.py at the end of the file
    """
    def start_training_dataset(self, trainging_folder):
        image_data = np.load(trainging_folder+"cars_train.npy")
        image_annotation_dict =np.load(trainging_folder+"cars_train_annos.npy").item()
        ii = 0
        try:
            file = open("./Trained_Models/image_number_learned.txt",'r')
            ii = int(file.readline())
        except:
            pass
        print("image number: ",str(ii))
        while ii < len(image_data):
            i = image_data[ii]
            image = imread(trainging_folder+"cars_train/"+i)
            image_bounding_boxes = []
            for j in image_annotation_dict[i]:
                xmin, ymin, xmax, ymax = j[0:4]
                image_bounding_boxes.append([xmin,ymin,xmax,ymax])
            image_class_names = ['car']
            image_shape = image.shape
            # show_image_w_bboxes(image, image_bounding_boxes)
            self.notify(image, image_shape, image_bounding_boxes, image_class_names)
            ii += 1
            write_file = open("./Trained_Models/image_number_learned.txt",'w')
            write_file.write(str(ii))
            write_file.close()

    def start_testing_dataset(self, testing_folder):
        image_data = np.load(testing_folder+"cars_test.npy")
        image_annotation_dict =np.load(testing_folder+"cars_test_annos.npy").item()
        for i in image_data:
            image = imread(testing_folder+"cars_test/"+i)
            image_bounding_boxes = []
            for j in image_annotation_dict[i]:
                xmin, ymin, xmax, ymax = j[0:4]
                image_bounding_boxes.append([xmin,ymin,xmax,ymax])
            image_class_names = ['car']
            image_shape = image.shape
            # show_image_w_bboxes(image, image_bounding_boxes)
            self.notify(image, image_shape, image_bounding_boxes, image_class_names)

    def notify(self, image, image_shape, image_bounding_boxes, image_class_names):
        self.observable.update_image(image, image_shape, image_bounding_boxes, image_class_names)

    def get_instace(self, train_instance):
        self.observable = train_instance
    
    def get_classes(self):
        """
        parse name of the classes as key and give an independent number to each class starting from one to (number_of_classes) 
        :return: 
        """
        return {'car':1}



