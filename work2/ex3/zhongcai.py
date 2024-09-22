import re
import os
import time

# method_name =
config_file_name_list = ['configs/PASCAL_VOC/split1/1shot_baseline.yml', 'configs/PASCAL_VOC/split1/3shot_CL_IoU.yml', 'configs/PASCAL_VOC/split1/5shot_CL_IoU.yml', 'configs/PASCAL_VOC/split1/10shot_CL_IoU.yml'];
ex1_sub = 'ex3/';

ex1_path = './ex3'
ex1_path = ex1_path + ex1_sub
seed_num = 'seed5'
for files_name in os.listdir(ex1_path):
    if files_name.split('-')[0] != 'SODdet':
        continue
    files_path = ex1_path + files_name


    for config_file_name in config_file_name_list:
        shot_num = config_file_name.split('/')[3].split('shot')[0]
        shot = shot_num+'shot'
        with open(files_path+'/run_semidethap2sdet_resume_base.json', 'r') as f:
            alllines = f.readlines()
        
        with open(files_path+'/run_semidethap2sdet_resume_base.json', 'w+') as f:
            for eachline in alllines:
                if 'train_semircnn1_' in eachline:
                    temp = eachline.split('_')
                    ori_shot_num = eachline.split('_')[4]
                    ori_seed_num = eachline.split('_')[-1]
                    earchlines = eachline.replace(ori_shot_num, shot)
                    earchlines = earchlines.replace(ori_seed_num, seed_num)
                    f.writelines(earchlines+'",'+"\n")
                else:    
                    f.writelines(eachline)
        
        with open(files_path+"/"+config_file_name, 'r') as f:
            alllines = f.readlines()

        with open(files_path+"/"+config_file_name, 'w+') as f:
            for eachline in alllines:
                if 'TRAIN: (\'voc_2007_trainval' in eachline:
                    ori_shot_num = eachline.split('_')[4]
                    ori_seed_num = eachline.split('_')[-1]
                    eachlines = eachline.replace(ori_seed_num, seed_num+"',)\n")
                    f.writelines(eachlines)
                else:
                    f.writelines(eachline)

        with open(files_path+"/start.sh", 'r') as f:
            alllines = f.readlines()

        with open(files_path+"/start.sh", 'w+') as f:
            for eachline in alllines:
                if 'python3 train_semihap2sdet_resume_base.py' in eachline:
                    runningConf = eachline.split(' ')[-1]
                    eachlines = eachline.replace(runningConf, config_file_name+"\n")
                    f.writelines(eachlines)
                else:
                    f.writelines(eachline)
        # print(1)
                time.sleep(60)
    

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
