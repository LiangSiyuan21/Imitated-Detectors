
from pycocotools.coco import COCO
 
annoationDir='/srv/hdd/data/VOCdevkit/'
# annoationDir='/srv/hdd/data/coco2017/annotations/'
dataType='train'
# annFile='{}voc0712_{}.json'.format(annoationDir, dataType)
# annFile='{}coco20_{}.json'.format(annoationDir, dataType)
annFile='{}coco19756_{}.json'.format(annoationDir, dataType)
 
# initialize COCO api for instance annotations
coco=COCO(annFile)
 
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
cat_nms=[cat['name'] for cat in cats]
print('number of categories: ', len(cat_nms))
print('COCO categories: \n', cat_nms)
 
# 统计各类的图片数量和标注框数量
for cat_name in cat_nms:
    catId = coco.getCatIds(catNms=cat_name)     # 1~90
    imgId = coco.getImgIds(catIds=catId)        # 图片的id  
    annId = coco.getAnnIds(catIds=catId)        # 标注框的id
 
    print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))
