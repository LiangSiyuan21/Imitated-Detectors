import re
import os
import time

method_name = '_os_fpn_voc2012_iter10';
config_file_name = './ex1/SemiRcnn-cls5/configs/PASCAL_VOC/semi-base-training/R101_FPN_os_base_training_split1.yml';
ex1_sub = 'ex1/';

ex1_path = './ex1/'
ex1_path = ex1_path + ex1_sub

for files_name in os.listdir(ex1_path):
    if files_name.split('-')[0] != 'SemiRcnn':
        continue
    files_path = ex1_path + files_name

    with open(files_path+'/run_semidet_test.json', 'r') as f:
        alllines = f.readlines()
    
    with open(files_path+'/run_semidet_test.json', 'w+') as f:
        for eachline in alllines:
            if 'test_semircnn_' in eachline:
                # print(eachline.split("\""))
                old_task_name = eachline.split("\"")[3]
                old_method_name = str()
                old_task_name_list = old_task_name.split('_')
                for i in range(2, len(old_task_name_list)-1):
                    old_method_name = old_method_name + "_" + old_task_name_list[i]
                eachline = eachline.replace(old_method_name, method_name)
            f.writelines(eachline)
    
    with open(files_path+'/start.sh', 'r') as f:
        alllines = f.readlines()
    
    with open(files_path+'/start.sh', 'w+') as f:
        for eachline in alllines:
            if 'config-file' in eachline:
                old_config_file_name = eachline.split(" ")[-1]
                eachline = eachline.replace(old_config_file_name, config_file_name)
        f.writelines(eachline)
    
    # os.chdir(ex1_path)


    

# for i, cls_index in enumerate(cls_list):
#     f=open('./run_semidet_test.json','r')
#     alllines=f.readlines()
#     f.close()
#     f=open('./run_semidet_test.json','w+')
#     for eachline in alllines:
#         a=re.sub(r'"test_semircnn_v2_base1_bs16_mr3_(.*)"','"test_semircnn_v2_base1_bs16_mr3_' + cls_index + '"',eachline)
#         f.writelines(a)
#     f.close()


#     f=open('test_net_multi.py','r',encoding='utf-8')
#     alllines=f.readlines()
#     f.close()
#     f=open('test_net_multi.py','w+',encoding='utf-8')
#     for eachline in alllines:
#         a=re.sub(r'cls_name = (.*);','cls_name = "'+ cls_index + '";',eachline)
#         f.writelines(a)
#     f.close()

#     time.sleep(250)
