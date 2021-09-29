import os, glob, re

import argparse, yaml

def refine_fn(config):
    textgrid_path = config['path']['preprocessed_path']
    #filelists= glob.glob(f'{textgrid_path}/TextGrid/**/*.TextGrid', recursive=True)
    filelists= glob.glob(f'./p0020/*.TextGrid', recursive=True)
    for i in filelists:
        path_ori = os.path.split(i)[0]
        ch_name = os.path.split(i)[1]
        i1 = ch_name.split('-')[0]
        i2 = ch_name.split('-')[1]
        i3 = ch_name.split('-')[2]
        new_name = f'{i1}_{i2}_{i3}'
        #print(path_ori)
        #print(new_name)
        os.rename(i, path_ori+'/'+new_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to preprocess.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    refine_fn(config)

