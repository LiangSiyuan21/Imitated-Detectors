import os, sys
import shutil
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import random
from fvcore.common.file_io import PathManager


src_dataset_dir = "//selenaliang/data/VOCdevkit/"
src_train_txt_path = 'datasets/vocsplit/'
txt_path = '//selenaliang/data/SODdet/retrieval_vocsplit_txt/'
train_txt_path = '//selenaliang/data/SODdet/retrieval_vocsplit/'
save_path = "//selenaliang/data/SODdet/retrieval_annos_imgs/"



VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']

data = []
data_per_cat = {c: [] for c in VOC_CLASSES}
for year in [2007, 2012]:
    data_file = '//selenaliang/data/VOCdevkit/VOC{}/ImageSets/Main/trainval.txt'.format(year)
    with PathManager.open(data_file) as f:
        fileids = np.loadtxt(f, dtype=np.str).tolist()
    data.extend(fileids)
for fileid in data:
    year = "2012" if "_" in fileid else "2007"
    dirname = os.path.join("//selenaliang/data/VOCdevkit", "VOC{}".format(year))
    anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
    img_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")
    tree = ET.parse(anno_file)
    clses = []
    for obj in tree.findall("object"):
        cls = obj.find("name").text
        clses.append(cls)
    for cls in set(clses):
        data_per_cat[cls].append(img_file)

save_imgs_path = save_path+'JPEGImages/'
save_annos_path = save_path+'Annotations/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_imgs_path):
    os.makedirs(save_imgs_path)
if not os.path.exists(save_annos_path):
    os.makedirs(save_annos_path)

train_txt = dict(list())
src_img_list = []
for img_name in os.listdir(src_dataset_dir+'VOC2007/JPEGImages/'):
    src_img_list.append(src_dataset_dir+'VOC2007/JPEGImages/'+img_name)
for img_name in os.listdir(src_dataset_dir+'VOC2012/JPEGImages/'):
    src_img_list.append(src_dataset_dir+'VOC2012/JPEGImages/'+img_name)

for shot in os.listdir(txt_path):
    shot_num = int(shot.split('shot')[-1])
    for seed in os.listdir(txt_path+shot+'/'):
        seed_num = int(seed.split('seed')[-1])
        
        for cls_name in VOC_CLASSES:
            src_train_file = 'box_'+str(shot_num)+'shot_'+cls_name+'_train.txt'
            with open(src_train_txt_path+seed+'/'+src_train_file) as f:
                train_txt[cls_name] = f.readlines()
                # new_img_names = []
                # for src_file in train_txt[cls_name]:
                #     src_xml = src_file.replace('JPEGImages', 'Annotations').split('.')[-1] + '.xml'
                #     img_name = src_file.split('.')[-1] + '_' + cls_name +'.jpg'
                #     img_xml = src_file.split('.')[-1] + '_' + cls_name +'.xml'
                #     shutil.copy(src_file, save_imgs_path+img_name)
                #     new_img_names.append(img_name + '\n')
                #     shutil.copy(src_xml, img_xml)
                # train_txt[cls_name] = new_img_names

        for file in os.listdir(txt_path+shot+'/'+seed):  
            cls_name = file.split('_')[1]
            retrieval_num = int(file.split('_')[3].split('.')[0])

            with open(txt_path+shot+'/'+seed+'/'+file, 'r') as f:
                retrieval_imgs = [each_line.strip('\n') for each_line in f]
            for retrieval_img in retrieval_imgs:
                retrieval_img = retrieval_img.split(' ')
                src_img_name = retrieval_img[0]+ '.jpg'
                img_name = retrieval_img[0]+'_'+cls_name+'_seed'+str(seed_num)+'_shot'+str(shot_num)+'.jpg'
                xml_name = retrieval_img[0]+'_'+cls_name+'_seed'+str(seed_num)+'_shot'+str(shot_num)+'.xml'
                gts = []
                for gts_index in range(1, len(retrieval_img), 5):
                    gt = retrieval_img[gts_index:gts_index+5]
                    gts.append(gt)
                
                for img in src_img_list:
                    if src_img_name == img.split('/')[-1]:
                        im = Image.open(img)
                        w, h = im.size
                        shutil.copyfile(img, save_imgs_path+img_name)
                        print('ori img path:{} --> new img path:{}'.format(img, save_imgs_path+img_name))
                        train_txt[cls_name].append(save_imgs_path+img_name+'\n')
                        with open((save_annos_path+xml_name), 'w') as file:
                            file.write('<annotation>\n')
                            file.write('    <folder>RETRIEVAL</folder>\n')
                            file.write('    <filename>' + str(img) + '</filename>\n')
                            file.write('    <size>\n')
                            file.write('        <width>' + str(w) + '</width>\n')
                            file.write('        <height>' + str(h) + '</height>\n')
                            file.write('        <depth>3</depth>\n')
                            file.write('    </size>\n')
                        
                            # write the region of image on xml file
                            for img_each_label in gts:
                                spt = img_each_label
                                file.write('    <object>\n')
                                file.write('        <name>' + str(spt[0]) + '</name>\n')
                                file.write('        <pose>Unspecified</pose>\n')
                                file.write('        <truncated>0</truncated>\n')
                                file.write('        <difficult>0</difficult>\n')
                                file.write('        <bndbox>\n')
                                file.write('            <xmin>' + str(spt[1]) + '</xmin>\n')
                                file.write('            <ymin>' + str(spt[2]) + '</ymin>\n')
                                file.write('            <xmax>' + str(spt[3]) + '</xmax>\n')
                                file.write('            <ymax>' + str(spt[4]) + '</ymax>\n')
                                file.write('        </bndbox>\n')
                                file.write('    </object>\n')
                        
                            file.write('</annotation>')

        if not os.path.exists('datasets/retrieval_vocsplit/'+seed+'/'):
            os.makedirs('datasets/retrieval_vocsplit/'+seed+'/')
        print('------------------------------------------------')
        print('annos and imgs have been done.')
        for key in train_txt:
            if len(train_txt[key]) < (shot_num+retrieval_num):
                for _ in range(retrieval_num):
                    train_txt[key].append(random.sample(data_per_cat[key], 1)[0]+'\n')
        
        for key in train_txt:
            train_txt_name = 'box_' + str(shot_num) + 'shot_'+key+'_train.txt'
            if not os.path.exists(train_txt_path+seed+'/'):
                os.makedirs(train_txt_path+seed+'/')
            with open((train_txt_path+seed+'/'+train_txt_name), 'w') as f:
                for line in train_txt[key]:
                    f.writelines(line)
        print('------------------------------------------------')
        print('train_txt have been done.')

