import os
import random

ILSVRC_CLASSES_DICT = {
'n02672831': 'accordion', 
'n02219486': 'ant', 
'n02454379': 'armadillo', 
'n02764044': 'axe', 
'n02777292': 'balance beam', 
'n02786058': 'band aid',
'n02807133': 'bathing cap',
'n02815834': 'beaker',
'n02206856': 'bee',
'n07720875': 'bell pepper',
'n02840245': 'binder',
'n02879718': 'bow',
'n02892767': 'brassiere',
'n07880968': 'burrito',
'n02274259': 'butterfly',
'n02437136': 'camel',
'n02992211': 'cello',
'n01784675': 'centipede',
'n03017168': 'chime',
'n03063338': 'coffee maker',
'n03128519': 'cream',
'n03141823': 'crutch',
'n07718472': 'cucumber',
'n03188531': 'diaper',
'n02268443': 'dragonfly',
'n03249569': 'drum',
'n03255030': 'dumbbell',
'n03271574': 'electric fan',
'n03314780': 'face powder',
'n07753113': 'fig',
'n03372029': 'flute',
'n02118333': 'fox',
'n03394916': 'french horn',
'n01639765': 'frog',
'n02510455': 'giant panda',
'n01443537': 'goldfish',
'n07583066': 'guacamole',
'n03467517': 'guitar',
'n07697100': 'hamburger',
'n03481172': 'hammer',
'n02342885': 'hamster',
'n03494278': 'harmonica',
'n03495258': 'harp',
'n07714571': 'head cabbage',
'n03513137': 'helmet',
'n02398521': 'hippopotamus',
'n03584254': 'iPod',
'n01882714': 'koala bear',
'n02165456': 'ladybug',
'n07749582': 'lemon',
'n02129165': 'lion',
'n03676483': 'lipstick',
'n01982650': 'lobster',
'n03759954': 'microphone',
'n03770439': 'miniskirt',
'n02484322': 'monkey',
'n07734744': 'mushroom',
'n03804744': 'nail',
'n03814639': 'neck brace',
'n03838899': 'oboe',
'n02444819': 'otter',
'n03916031': 'perfume',
'n03928116': 'piano',
'n07753275': 'pineapple',
'n07768694': 'pomegranate',
'n07615774': 'popsicle',
'n02346627': 'porcupine',
'n04004767': 'printer',
'n04023962': 'punching bag',
'n04026417': 'purse',
'n02324045': 'rabbit',
'n04039381': 'racket',
'n02509815': 'red panda',
'n01770393': 'scorpion',
'n02076196': 'seal',
'n01944390': 'snail',
'n01726692': 'snake',
'n02355227': 'squirrel',
'n04317175': 'stethoscope',
'n04330267': 'stove',
'n07745940': 'strawberry',
'n04336792': 'stretcher',
'n04376876': 'syringe',
'n04392985': 'tape player',
'n01776313': 'tick',
'n02129604': 'tiger',
'n04487394': 'trombone',
'n03110669': 'trumpet',
'n01662784': 'turtle',
'n04517823': 'vacuum',
'n04536866': 'violin',
'n02062744': 'whale',
'n04542943': 'waffle iron'
}


import os
# from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import shutil

Dataset_path = '//data/ILSVRC'
ImgPath = '/Data/DET/train'
AnnoPath = '/Annotations/DET/train'
SubDataset = '/ILSVRC2014_train_000'

new_dataset_path = '//data/VOCdevkit/ILSVRC2014'


if not os.path.exists(new_dataset_path):
    os.makedirs(new_dataset_path)
    os.makedirs(new_dataset_path+'/Annotations/')
    os.makedirs(new_dataset_path+'/JPEGImages/')

# prepare datasets
# cls_counter = {}
# for cls_name in ILSVRC_CLASSES_DICT:
#     cls_counter[cls_name] = 0
# for i in range(6):
#     annos_subdataset_path = Dataset_path + AnnoPath + SubDataset + str(i) + '/'
#     imgs_subdataset_path = Dataset_path + ImgPath + SubDataset + str(i) + '/'
#     for anno_name in os.listdir(annos_subdataset_path):
#         file_name = anno_name.split('.')[0]
#         img_name = file_name + '.JPEG'

#         tree = ET.parse(annos_subdataset_path+anno_name)
#         root = tree.getroot()
#         save_flag = False
#         for elem in root.iter('name'):
#             if elem.text in ILSVRC_CLASSES_DICT and cls_counter[elem.text] <= 400:
#                 new_elem = ILSVRC_CLASSES_DICT[elem.text]
#                 cls_counter[elem.text] = cls_counter[elem.text] + 1
#                 elem.text = new_elem
#                 save_flag = True
#         if save_flag:
#             tree.write(new_dataset_path+'/Annotations/'+anno_name)
#             ori_img = imgs_subdataset_path + img_name
#             new_img = new_dataset_path+'/JPEGImages/' + img_name
#             print(ori_img+'->'+new_img)
#             shutil.copy(ori_img, new_img)
#     print('subset dataset ' + str(i) + 'is done.')


# prepare trainval.txt
file_name = new_dataset_path + '/ImageSets/Main/' + 'trainval.txt'
for anno_name in os.listdir(new_dataset_path+'/Annotations/'):
    with open(file_name, 'w') as file:
        file.write(anno_name.split('.')[0]+'\n')

print('trainval.txt is done.')