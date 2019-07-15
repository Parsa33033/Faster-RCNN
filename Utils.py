

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.misc import imresize
import tensorflow as tf

def reshape_for_CNN(array):
    """
    the shape suitable for CNN. usually is [1, height, width, channels]
    :param array: a three dimensional array
    :return: reshaped array for CNN
    """
    h, w, c = array.shape
    return np.reshape(array, [1, h, w, c])


def reshape_image_side(image, small_side = 600):
    """
    reshaping the smaller side of the image to the small_side value and modify the shape of the bigger side by
    the scale of change in the smaller side
    :param image: the image with the shape of (height, width, 3)
    :param small_side: the fix small side of the images (mensioned in the faster rcnn article to be 600)
    :return: an image with a small side of 600
    """
    height, width, channels = image.shape
    if height < width:
        resize_scale = small_side / height
        return imresize(image, [small_side, int(width*resize_scale), channels])
    else:
        resize_scale = small_side / width
        return imresize(image, [int(height*resize_scale), small_side, channels])

def get_real_bounding_boxes(image_height_change_ratio, image_width_change_ratio, bounding_boxes):
    """
    the dataset contains bounding boxes with ratio of coordinates of the bounding boxes. this method modifies these
    ratios to real height and width sizes
    :param image_height:
    :param image_width:
    :param bounding_boxes:
    :return:
    """

    bounding_boxes = np.array(bounding_boxes)
    bbox_shape = bounding_boxes.shape
    bounding_boxes = np.column_stack([bounding_boxes[:,0] * image_width_change_ratio,
                                     bounding_boxes[:,1] * image_height_change_ratio,
                                     bounding_boxes[:,2] * image_width_change_ratio,
                                     bounding_boxes[:,3] * image_height_change_ratio])
    bounding_boxes = np.reshape(bounding_boxes, [*bbox_shape])
    return bounding_boxes

def prepare_image_and_bboxes(image, bounding_boxes):
    """
    reshaping smallest side of the image to 600(and bigger side be the reshaping scale as well),
    changing the dimension of the image to [1, height, width, channels] from [height, width, channels],
    getting the bounding boxes coordinates relative to the image size
    :param image:
    :param bounding_boxes:
    :return:
    """
    height = image.shape[-3]
    width = image.shape[-2]
    image = reshape_image_side(image, small_side = 600)
    image = reshape_for_CNN(image)
    new_height = image.shape[1]
    new_width = image.shape[2]
    bounding_boxes = get_real_bounding_boxes(new_height/height, new_width/width, bounding_boxes)
    return image, new_height, new_width, bounding_boxes

def iou_calculation(a,b):
    return intersection(a,b)/union(a,b)


def intersection(a,b):
    """
    The area intersection of two bounding boxes
    :param a: a list containing xmin, ymin, xmax, ymax of a box
    :param b: a list containing xmin, ymin, xmax, ymax of a box
    :return:
    """
    axmin, aymin, axmax, aymax = a
    bxmin, bymin, bxmax, bymax = b

    xmin = np.maximum(axmin, bxmin)
    ymin = np.maximum(aymin, bymin)
    xmax = np.minimum(axmax, bxmax)
    ymax = np.minimum(aymax, bymax)
    w = xmax - xmin
    h = ymax - ymin
    if w<=0 or h<=0:
        return 0
    if w>0 and h>0:
        return (xmax - xmin) * (ymax - ymin)

def union(a,b):
    """
    The area union of two bounding boxes
    :param a: a list containing xmin, ymin, xmax, ymax of a box
    :param b: a list containing xmin, ymin, xmax, ymax of a box
    :return:
    """
    axmin, aymin, axmax, aymax = a
    bxmin, bymin, bxmax, bymax = b
    a_area = np.abs(axmax - axmin) * np.abs(aymax - aymin)
    b_area = np.abs(bxmax - bxmin) * np.abs(bymax - bymin)
    return a_area + b_area - intersection(a,b)

def draw_box(image, box):
    xmin, ymin, xmax, ymax = box
    _, ax = plt.subplots(1)
    ax.imshow(np.squeeze(image))
    ax.add_patch(patches.Rectangle((xmin, ymin),xmax-xmin, ymax-ymin, edgecolor='r', facecolor='none', linewidth=1))
    plt.show()

def xyxy_to_xywh(a):
    """
    converting xmin, ymin, xmax, ymax to x, y center coordinates and width and height
    :param a: a list containing xmin, ymin, xmax, ymax of a box
    :return:
    """
    xmin, ymin, xmax, ymax = a
    w = xmax-xmin
    h = ymax-ymin
    x = xmin + w/2
    y = ymin + h/2
    return x, y, w, h

def xywh_to_xyxy(a):
    """
    converting x, y center coordinates and width and height to xmin, ymin, xmax, ymax
    :param a:
    :return:
    """
    x, y, w, h = a
    xmin = x - w/2
    ymin = y - w/2
    xmax = x + w
    ymax = y + h
    return xmin, ymin, xmax, ymax

def show_image_w_bboxes(image, bounding_boxes):
    image, height, width, bounding_boxes = prepare_image_and_bboxes(image, bounding_boxes)
    _, ax = plt.subplots(1)
    ax.imshow(np.squeeze(image))
    for bounding_box in bounding_boxes:
        xmin, ymin, xmax, ymax = bounding_box
        ax.add_patch(patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, facecolor='none', edgecolor='r', linewidth=1))
    plt.show()


def encode_to_deltas(anchor_box, bounding_box):
    """
    encoding the differece between an anchor box and a bounding box to deltas
    :param anchor_box: a list containing xmin, ymin, xmax, ymax of a box
    :param bounding_box: a list containing xmin, ymin, xmax, ymax of a box
    :return:
    """
    ax, ay, aw, ah = xyxy_to_xywh(anchor_box)
    bx, by, bw, bh = xyxy_to_xywh(bounding_box)

    dx = (bx - ax)/aw
    dy = (by - ay)/ah
    dw = np.log(bw/aw)
    dh = np.log(bh/ah)

    return dx, dy, dw, dh

def decode_to_bbox(anchor_box, deltas):
    """
    decoding the deltas so that the anchor box gets translated to become close to the ground truth
    :param anchor_box: a list containing xmin, ymin, xmax, ymax of a box
    :param deltas: a list containing dx, dy, dw, dh of a box
    :return:
    """
    ax, ay, aw, ah = xyxy_to_xywh(anchor_box)
    dx, dy, dw, dh = deltas

    x = dx * aw + ax
    y = dy * ah + ay
    w = (np.e ** dw) * aw
    h = (np.e ** dh) * ah

    xmin, ymin, xmax, ymax = xywh_to_xyxy([x, y, w, h])
    return xmin, ymin, xmax, ymax


def l1smooth():
    """
    in case of using l1 smooth for rpn and rcnn model
    :return:
    """
    return tf.losses.huber_loss
