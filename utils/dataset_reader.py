import tensorflow as tf
import cv2
import os
import numpy as np
import xml.dom.minidom


#initialize variable
_dir = ''                                 #最后补充


def xml_extractor(_dir):
    domtree = xml.dom.minidom.parse(_dir)
    collection = domtree.documentElement
    file_name_xml = collection.getElementsByTagName('filename')[0]
    objects_xml = collection.getElementsByTagName('object')[0]
    size_xml = collection.getElementsByTagName('size')[0]
    
    file_name = file_name_xml.childNodes[0].data

#read the size of image    
    for size in size_xml:
        width = size.getElementsByTagName('width')[0]
        height = size.getElementsByTagName('height')[0]
        
        width = width.childNodes[0].data
        height = height.childNode[0].data

#put all object(s) and its bundary box(s) in a picture into _objects matrix  
    _objects = []
    for object_xml in objects_xml:
        object_name = object_xml.getElementsByTagName('name')[0]
        bdbox = object_xml.getElementsByTagName('bndbox')[0]
        xmin = bdbox.getElementsByTagName('xmin')[0]
        ymin = bdbox.getElementsByTagName('ymin')[0]
        xmax = bdbox.getElementsByTagName('xmax')[0]
        ymax = bdbox.getElementsByTagName('ymax')[0]
        
        _object = (object_name.childNodes[0].data,
                   xmin.childNodes[0].data,
                   ymin.childNodes[0].data,
                   xmax.childNodes[0].data,
                   ymax.childNodes[0].data
                   )
        
        _objects.append(_object)
    
    return file_name,width,height,_objects
    

#create function to tansfer
def labels_transform(batches_filenames,target_width,target_height,layerout_width,layerout_height):

#create a dictionary of label's name and label    
    class_dictionary={'person':5,
                      'bird':6,
                      'cat':7,
                      'cow':8,
                      'dog' : 9,
                      'horse' : 10,
                      'sheep' : 11,
                      'aeroplane' : 12,
                      'bicycle' : 13,
                      'boat' : 14,
                      'bus' : 15,
                      'car' : 16,
                      'motorbike' : 17,
                      'train' : 18,
                      'bottle' : 19,
                      'chair' : 20,
                      'diningtable' : 21,
                      'pottedplant': 22,
                      'sofa' : 23,
                      'tvmonitor' : 24
                      }
    
    batches_labels = []
    for batch_filenames in batches_filenames:
        batch_labels = []
        for filename in batch_filenames:
            _, width, height, objects = xml_extractor( filename )
            width_preportion = target_width/int(width)
            height_preportion = target_height/int(height)
            label = np.add(np.zeros([int(layerout_height),int(layerout_width),255]),1e-8)#1e-8 can avoid the situation of nan
            for _object in objects:
                class_label = class_dictionary[_object[0]]
                xmin = float(_object[1])
                ymin = float(_object[2])
                xmax = float(_object[3])
                ymax = float(_object[4])
                
                x = (1.0 * xmax - xmin) / 2 * width_preportion       #x of center point of object
                y = (1.0 * ymax - ymin) / 2 *height_preportion       #y of center point of object
                
                bdbox_width = (1.0 * xmax - xmin) * width_preportion    #width of bdbox
                bdbox_height = (1.0 * ymax - ymin) * height_preportion  #height of bdbox
                
                flag_width = int(target_width) / layerout_width         
                flag_height = int(target_width) / layerout_height
                box_x = x // flag_width                                 #x subscript of box
                box_y = y // flag_height                                #y subscript of box
                 
                if (box_x == layerout_width):
                    box_x -= 1
                if (box_y == layerout_height):
                    box_y -= 1
                
                for i in range(3):                            # 3 bdbox per box
                    label[int(box_y),int(box_x),i*25] = x     #point x
                    label[int(box_y),int(box_x),i*25 + 1] = y
                    label[int(box_y),int(box_x),i*25 + 2] = bdbox_width
                    label[int(box_y),int(box_x),i*25 + 3] = bdbox_height
                    label[int(box_y),int(box_x),i*25 + 4] = 1                   #objectness
                    label[int(box_y),int(box_x),i*25 + int(class_label)] = 0.9  #class_label
                
            batch_labels.append(label)
        
        batches_labels.append(batch_labels)
    
    return batches_labels
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    