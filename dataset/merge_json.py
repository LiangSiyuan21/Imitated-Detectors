
import os
import datetime
import random
import sys
import operator
import math
import numpy as np
import skimage.io
import matplotlib
from matplotlib import pyplot as plt
import cv2
from collections import defaultdict, OrderedDict
import json
#待合并的路径
FILE_DIR = "/srv/hdd/data/VOCdevkit/"
 
def load_json(filenamejson):
    with open(filenamejson) as f:
        raw_data = json.load(f)
    return raw_data
 
file_count = 0
new_file_name = []
files = ['MTIDIFGSM_5_voc0712_train.json', 'voc0712_train.json']
for x in range(len(files)):
    #解析文件名字和后缀
    #print(str(files[x]))
    file_suffix = str(files[x]).split(".")[1]
    file_name = str(files[x]).split(".")[0]
#     #过滤类型不对的文件
#     if file_suffix != "json":
#         continue
    #计数
    file_count = file_count + 1
    #组合文件路径
    filenamejson = FILE_DIR + str(files[x])
    #读取文件
    if x == 0:
        #第一个文件作为root
        root_data = load_json(filenamejson)
        new_file_name = file_name
        start_img_id = root_data['images'][len(root_data['images'])-1]['id'] + 1
        start_anno_id = root_data['annotations'][len(root_data['annotations'])-1]['id'] + 1
    else:
        new_file_name = new_file_name + '_' + file_name
        raw_data = load_json(filenamejson)
        #追加images的数据
        for index in range(len(raw_data['images'])):
            raw_data['images'][index]['id'] = int(raw_data['images'][index]['id']) + start_img_id
        root_data['images'].extend(raw_data['images'])
        #统计区域个数
        annotation_count = str(raw_data["annotations"]).count('image_id',0,len(str(raw_data["annotations"])))
        for i in range(annotation_count):
            #追加annotations的数据
            raw_data['annotations'][i]['image_id'] = int(raw_data['annotations'][i]['image_id']) + start_img_id
            raw_data['annotations'][i]['id'] = int(raw_data['annotations'][i]['id']) + start_anno_id
            root_data['annotations'].append(raw_data['annotations'][i])
        #统计根文件类别数
        #root_categories_count = str(root_data["categories"]).count('name',0,len(str(root_data["categories"])))
        #统计这个文件的类别数
        start_img_id = start_img_id + len(root_data['images'])
        start_anno_id = start_anno_id + len(root_data['annotations'])
        raw_categories_count = str(raw_data["categories"]).count('name',0,len(str(raw_data["categories"])))
        for j in range(raw_categories_count):
            root_data["categories"].append(raw_data['categories'][j])

new_file_name = new_file_name + '.json'
temp = []
for m in root_data["categories"]:
    if m not in temp:
        temp.append(m)
root_data["categories"] = temp
print("共处理 {0} 个json文件".format(file_count))
print("共找到 {0} 个类别".format(str(root_data["categories"]).count('name',0,len(str(root_data["categories"])))))
 
json_str = json.dumps(root_data)
with open(FILE_DIR + new_file_name, 'w') as json_file:
    json_file.write(json_str)
#写出合并文件
 
print("Done!")
