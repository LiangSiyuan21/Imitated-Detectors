
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Testing Script.

This scripts reads a given config file and runs the evaluation.
It is an entry point that is made to evaluate standard models in FsDet.

In order to let one script support evaluation of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use FsDet as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import re
import argparse
import os


from subprocess import Popen, PIPE, STDOUT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--config-file', type=str, default=None)

    args = parser.parse_args()
    return args

os_path = os.getcwd()
print('e')
def main(args):
    cls_name = "sofa";
    shot_nums = [1, 3, 5, 10]
    for seed in range(5):
        for j, shot_num in enumerate(shot_nums):
            print('--------------------cls_shot------------------')
            print(cls_name+'_'+str(shot_num))
            print('---------------------seed--------------------')
            print(seed)
            
            with open(os_path+'/fsdet_alter/modeling/meta_arch/semircnn.py', 'r', encoding='utf-8') as f:
                alllines=f.readlines()


            with open(os_path+'/fsdet_alter/modeling/meta_arch/semircnn.py', 'w+', encoding='utf-8') as f:
                for eachline in alllines:
                    ret = re.search(r'self.seed = (.*);',eachline)
                    if ret :
                        a=re.sub(r'self.seed = (.*);','self.seed = ' + str(seed+1) + ';',eachline)
                    else:
                        a = eachline
                    f.writelines(a)
            # f.close()
            
            with open(os_path+'/configs/PASCAL_VOC/semi-base-training/R101_FPN_hap2s_v1_base_training_split1_resume.yml', 'r', encoding='utf-8') as f:
                alllines=f.readlines()


            with open(os_path+'/configs/PASCAL_VOC/semi-base-training/R101_FPN_hap2s_v1_base_training_split1_resume.yml', 'w+', encoding='utf-8') as f:
                for eachline in alllines:
                    #print(eachline)
                    ret=re.search(r'SHOT_NUM: \d+',eachline)
                    if ret:
                        a=re.sub(r'SHOT_NUM: \d+','SHOT_NUM: ' + str(shot_num) ,eachline)
                    else:
                        a=eachline
                    f.writelines(a)

            test_cmd = 'python3 train_semihap2sdet_resume_base.py --num-gpus {} ' \
                   '--config-file {} --eval-only'.format(args.num_gpus, args.config_file)
            
            p = Popen(test_cmd, stdout=PIPE, stderr=STDOUT, shell=True)
            while True:
                line = p.stdout.readline().decode('utf-8')
                if not line:
                    break
                print(line)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    
            

