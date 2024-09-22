import re
import os
import time

cls_list = ["bird","bus","cow","motorbike","soft"];
per_list = ["60", "100"];
for seed in range(1):
    for i, cls_index in enumerate(cls_list):
        for j, per_index in enumerate(per_list):
                f=open('./run_semidet_test.json','r')
                alllines=f.readlines()
                f.close()
                f=open('./run_semidet_test.json','w+')
                for eachline in alllines:
                    a=re.sub(r'"test_semircnn_os_base1_bs64_mr3_seed(.*)"','"test_semircnn_os_base1_bs64_mr3_seed' + str(seed) + '_' + cls_index + '_' + per_index +'"',eachline)
                    f.writelines(a)
                f.close()

                f=open('./fsdet/modeling/meta_arch/semircnn.py','r',encoding='utf-8')
                alllines=f.readlines()
                f.close()
                f=open('./fsdet/modeling/meta_arch/semircnn.py','w+',encoding='utf-8')
                for eachline in alllines:
                    a=re.sub(r'index_list.append(.*);','index_list.append(int(cat_len/5)*' + str(seed) + ');',eachline)
                    f.writelines(a)
                f.close()

                f=open('./fsdet/data/datasets/meta_pascal_voc.py','r',encoding='utf-8')
                alllines=f.readlines()
                f.close()
                f=open('./fsdet/data/datasets/meta_pascal_voc.py','w+',encoding='utf-8')
                for eachline in alllines:
                    a=re.sub(r'alter_cls_percentage = (.*);','alter_cls_percentage = "'+ cls_index + '_' + per_index + '";',eachline)
                    f.writelines(a)
                f.close()
                time.sleep(120)
