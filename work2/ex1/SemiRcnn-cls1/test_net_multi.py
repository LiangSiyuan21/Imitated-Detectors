
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
import sys
# sys.path.insert(0, './ex1/SemiRcnn-cls1')

# print(sys.path)

from subprocess import Popen, PIPE, STDOUT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--config-file', type=str, default=None)

    args = parser.parse_args()
    return args

def main(args):
    cls_name = "bird";
    per_list = ["40",'50', "60", '70', "80"]
    for seed in range(5):
        for j, per_index in enumerate(per_list):
            print('--------------------cls_per------------------')
            print(cls_name+'_'+per_index)
            print('---------------------seed--------------------')
            print(seed)
            
            with open('./ex1/SemiRcnn-cls1/fsdet_alter/modeling/meta_arch/semircnn.py', 'r', encoding='utf-8') as f:
                alllines=f.readlines()


            with open('./ex1/SemiRcnn-cls1/fsdet_alter/modeling/meta_arch/semircnn.py', 'w+', encoding='utf-8') as f:
                for eachline in alllines:
                    a=re.sub(r'index_list.append(.*);','index_list.append(int(cat_len/5)*' + str(seed) + ');',eachline)
                    f.writelines(a)
            # f.close()

            with open('./ex1/SemiRcnn-cls1/fsdet_alter/data/datasets/meta_pascal_voc.py', 'r', encoding='utf-8') as f:
                alllines=f.readlines()


            with open('./ex1/SemiRcnn-cls1/fsdet_alter/data/datasets/meta_pascal_voc.py', 'w+', encoding='utf-8') as f:
                for eachline in alllines:
                    a=re.sub(r'alter_cls_percentage = (.*);','alter_cls_percentage = "'+ cls_name + '_' + per_index + '";',eachline)
                    f.writelines(a)

            # f=open('/code/open-world-object-detection/FSOD/SemiRcnn/fsdet/data/datasets/meta_pascal_voc.py','w+',encoding='utf-8')
            # for eachline in alllines:
            #     a=re.sub(r'alter_cls_percentage = (.*);','alter_cls_percentage = "'+ cls_name + '_' + per_index + '";',eachline)
            #     f.writelines(a)
            # f.close()

            test_cmd = 'python3 tools/test_net.py --dist-url auto --num-gpus {} ' \
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
    
            

