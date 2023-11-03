#!/usr/bin/env python3

import os
import re
import datetime
import numpy as np
from itertools import groupby
from skimage import measure
from PIL import Image
from pycocotools import mask
from math import sqrt

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }

    return image_info

def create_annotation_info(annotation_id, image_id, category_info, binary_mask=None,
                           image_size=None, tolerance=2, bounding_box=None):
    assert(binary_mask is not None or image_size is not None)

    if binary_mask is not None:
        if image_size is not None:
            binary_mask = resize_binary_mask(binary_mask, image_size)
        binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        area = mask.area(binary_mask_encoded)
        bounding_box = mask.toBbox(binary_mask_encoded)
    else:
        area = np.array(bounding_box[2] * bounding_box[3], dtype=int)
    if area < 20:
        print("Area of this annotation is less than 20, Skip it! image_id:", image_id, "area:", area, "bbox:", bounding_box)
        return None
    if category_info["is_crowd"]:
        is_crowd = 1
        segmentation = binary_mask_to_rle(binary_mask)
    else :
        is_crowd = 0
        if binary_mask is not None:
            binary_mask_encoded = mask.encode
            segmentation = binary_mask_to_polygon(binary_mask, tolerance)
            if not segmentation:
                return None
    if binary_mask is not None:
        annotation_info = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_info["id"],
            "iscrowd": is_crowd,
            "area": area.tolist(),
            "bbox": bounding_box.tolist(),
            "segmentation": segmentation,
            "width": binary_mask.shape[1],
            "height": binary_mask.shape[0],
        }
    else:
        annotation_info = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_info["id"],
            "iscrowd": is_crowd,
            "area": area.tolist(),
            "bbox": bounding_box.tolist(),
            "width": image_size[0],
            "height": image_size[1],
        }

    return annotation_info

'''
This file is for creating [MSCOCO style] json annotation files for convenient
training in popular detection frameworks, such as mmdetection, detectron,
maskrcnn-benchmark, etc. Before running this file, please follow the instructions
in https://github.com/waspinator/coco to install the COCO API or use the following
commands for installation:
    pip install cython
    pip install git+git://github.com/waspinator/coco.git@2.1.0

Usage: python3 code/make_json_anno.py --list_path /path/to/img/list/ [--tb_only]
'''

import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from argparse import ArgumentParser
import pycococreatortools


'''
ActiveTuberculosis: Active TB
ObsoletePulmonaryTuberculosis: Latent TB
PulmonaryTuberculosis: Unknown TB
'''
def cat2label(cls_name):
    x = {'ActiveTuberculosis': 1, 'ObsoletePulmonaryTuberculosis': 2, 'PulmonaryTuberculosis': 3}
    return x[cls_name]


'''
Load annotations in the XML format
Input:
       xml_path: (string), xml annoation (relative) path
       size    : (int, int), align with the actual image size
'''
def load_annotation(xml_path, resized=(512, 512)):
    if not os.path.exists(xml_path):
        return None, None
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        label = cat2label(name)
        difficult = int(obj.find('difficult').text)
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        width_ratio = width / resized[1]
        height_ratio = height / resized[0]
        ignore = False
        bbox[0] /= width_ratio; bbox[2] /= width_ratio
        bbox[1] /= height_ratio; bbox[3] /= height_ratio
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w < 1 or h < 1:
            ignore = 1
        if difficult or ignore:
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
        else:
            bboxes.append(bbox)
            labels.append(label)
    if not bboxes:
        bboxes = None #np.zeros((0, 4))
        labels = None #np.zeros((0, ))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0, ))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)
    if bboxes is not None:
        return bboxes.astype(np.float32), labels.astype(np.int64)
    else:
        return None, None


def dataset_info():
    INFO = [
        {
            'contributor': 'Yun Liu, Yu-Huan Wu, Yunfeng Ban, Huifang Wang, Ming-Ming Cheng',
            'date_created': '2020/06/22',
            'description': 'TBX11K Dataset',
            'url': 'http://mmcheng.net/tb',
            'version': '1.0',
            'year': 2020
        }
    ]


    LICENSES = [
        {
            'id': 1,
            'name': 'Attribution-NonCommercial-ShareAlike License',
            'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/'
        }
    ]

    CATEGORIES = [
        {'id': 1, 'name': 'ActiveTuberculosis', 'supercategory': 'Tuberculosis'},
        {'id': 2, 'name': 'ObsoletePulmonaryTuberculosis', 'supercategory': 'Tuberculosis'},
        {'id': 3, 'name': 'PulmonaryTuberculosis', 'supercategory': 'Tuberculosis'},
    ]

    return INFO, LICENSES, CATEGORIES

def getData (XML_filepath):
  
  return load_annotation(XML_filepath)
