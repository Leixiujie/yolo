import tensorflow as tf
import cv2
import os
import numpy as np
from xml.dom.minidom import parse
import xml.dom.minidom


#initialize variable
_dir = ''                                 #需要补充


def xml_extractor(_dir):
    domtree = parse(_dir)
    collection = domtree.documentElement
    file_name_xml = collection.getElementsByTagName('filename')[0]
    object_xml = collection.getElementsByTagName('object')[0]
    size_xml = collection.getElementsByTagName('size')[0]
    
    file_name = file_name_xml.childNodes[0].data
    
    for size in size_xml:
        width = size.getElementsByTagName('width')
