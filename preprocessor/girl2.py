import os, glob

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from shutil import copyfile

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    sub_dir = config["path"]["sub_dir_name"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    fixed_text_path = config["path"]["fixed_text_path"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    fixed_text_dict = dict()
    with open(fixed_text_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            wav, fixed_text = line.split('|')[0], line.split('|')[1]
            fixed_text_dict[wav] = fixed_text.replace('\n', '')

    filelist = glob.glob(f'{in_dir}/**/*.wav')

    print('make filelist ..')
    for fn_name in filelist:
        os.makedirs(f'{out_dir}/{sub_dir}',exist_ok=True)
        y, sr =librosa.load(fn_name)
        base_name=os.path.basename(fn_name)
        y = y/max(abs(y))* max_wav_value
        wavfile.write(f'{out_dir}/{sub_dir}/{base_name}', sampling_rate, y.astype(np.int16))

        text_path = os.path.join(f'{in_dir}/wav/'+base_name.replace('.wav','.txt'))
        if base_name in fixed_text_dict:
            text = fixed_text_dict[base_name]
        else:
            with open(text_path) as f:
                text = f.readline().strip("\n")
        text = _clean_text(text, cleaners)
        with open(os.path.join(f'{out_dir}/{sub_dir}/'+base_name.replace('.wav','.lab')),'w',encoding='utf-8') as f:
            f.write(text)

    # Filelist
    os.makedirs(f'{out_dir}', exist_ok=True)
    filelist_fixed = open(f'{out_dir}/filelist.txt', 'w', encoding='utf-8')
    with open(f'{in_dir}/filelist.txt', 'r', encoding='utf-8') as filelist:
        for line in tqdm(filelist.readlines()):
            wav = line.split('|')[0]
            if wav in fixed_text_dict:
                filelist_fixed.write("|".join([line.split("|")[0]] + [fixed_text_dict[wav]] + line.split("|")[2:]))
            else:
                filelist_fixed.write(line)
    filelist_fixed.close()

    # Speaker Info
    copyfile(f'{in_dir}/speaker_info.txt', f'{out_dir}/speaker_info.txt')

