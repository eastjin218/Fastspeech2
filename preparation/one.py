import re
import argparse
import yaml
import os
import shutil
import json
import librosa
import soundfile
from glob import glob
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from text import _clean_text
from text.korean import tokenize, normalize_nonchar


def write_text(txt_path, text):
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)


def get_sorted_items(items):
    # sort by key
    return sorted(items, key=lambda x:int(x[0]))


def get_emotion(emo_dict):
    e, a, v = 0, 0, 0
    if 'emotion' in emo_dict:
        e = emo_dict['emotion']
        a = emo_dict['arousal']
        v = emo_dict['valence']
    return e, a, v


def pad_spk_id(speaker_id):
    return '{}'.format("0"*(3-len(speaker_id))+speaker_id)


def create_dataset(preprocess_config):
    in_dir = preprocess_config['path']['corpus_path']
    out_dir = os.path.join(in_dir+'_preprocessed')
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    audio_files = glob(f'{in_dir}/**/*.wav', recursive=True)
    total_duration = 0
    print('Create dataset ...')
    for au_path in tqdm(audio_files):
        orig_sr = librosa.get_samplerate(au_path)
        y, sr =librosa.load(au_path, sr=orig_sr)
        new_sr = sampling_rate
        new_y = librosa.resample(y, sr, new_sr)
        duration = librosa.get_duration(new_y, sr=new_sr)
        total_duration += duration

        speaker_id = pad_spk_id(str(preprocess_config['person_info']['name']))
        tmp_fname = os.path.basename(au_path)
        tmp_fname = tmp_fname.split('.')[0]
        file_name = tmp_fname.split('_',1)[1]
        basename = f'{speaker_id}_{file_name}'
        wav_path = os.path.join(os.path.dirname(au_path).replace(in_dir,out_dir),f'{basename}.wav')
        os.makedirs(os.path.dirname(wav_path),exist_ok=True)
        soundfile.write(wav_path, new_y, new_sr)
    with open(in_dir+'/re_script.txt', 'r', encoding='utf-8') as f:
        tmp_s = f.readlines()
    for li in tqdm(tmp_s):
        script = refine_text(li.split('|')[2].strip())
        tmp_name = li.split('|')[0]
        tmp_name = tmp_name.split('.')[0]
        file_name = tmp_name.split('_',1)[1]
        gender = tmp_name.split('_')[0].upper()
        age = preprocess_config['person_info']['age']
        # write_text(os.path.join(os.path.dirname(wav_path)+f'/{speaker_id}_{file_name}.txt'),script)
        write_text(os.path.join(os.path.dirname(wav_path)+f'/{file_name}.txt'),script)
        # with open(f'{out_dir}/filelist.txt','a',encoding='utf-8') as f:
        #     f.write(f'{speaker_id}_{file_name}|{script}|{speaker_id}|others|SD|neutral|4|5|neutral|4|5|neutral|5|5|neutral|5|5|neutral|5|5\n')
        with open(f'{out_dir}/filelist.txt','a',encoding='utf-8') as f:
            f.write(f'{file_name}|{script}|{speaker_id}|others|SD|neutral|4|5|neutral|4|5|neutral|5|5|neutral|5|5|neutral|5|5\n')
        with open(f'{out_dir}/speaker_info.txt','w',encoding='utf-8') as f:
            f.write(f'{speaker_id}|{gender}|{age}')
    print(f'End parsing, total duration : {total_duration}')



def refine_text(text):
    # Fix invalid characters in text
    text = text.replace('…', ',')
    text = text.replace('\t', '')
    text = text.replace('-', ',')
    text = text.replace('–', ',')
    text = ' '.join(text.split())
    return text


def extract_audio(preprocess_config):
    in_dir = preprocess_config["path"]["corpus_path"]
    out_dir = os.path.join(os.path.dirname(in_dir), os.path.basename(in_dir)+"_tmp")
    video_files = glob(f'{in_dir}/**/*.mp4', recursive=True)

    print("Extract audio...")
    for video_path in tqdm(video_files):
        audio_path = video_path.replace(in_dir, out_dir, 1).replace('mp4', 'wav')
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)

        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, verbose=False)
        clip.close()


def extract_nonkr(preprocess_config):
    in_dir = preprocess_config["path"]["raw_path"]
    filelist = open(f'{in_dir}/nonkr.txt', 'w', encoding='utf-8')

    count = 0
    nonkr = set()
    print("Extract non korean charactors...")
    with open(f'{in_dir}/filelist.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_count = len(lines)
        for line in tqdm(lines):
            wav = line.split('|')[0]
            text = line.split('|')[1]
            reg = re.compile("""[^ ㄱ-ㅣ가-힣~!.,?:{}`"'＂“‘’”’()\[\]]+""")
            impurities = reg.findall(text)
            if len(impurities) == 0:
                count+=1
                continue
            norm = _clean_text(text, preprocess_config["preprocessing"]["text"]["text_cleaners"])
            impurities_str = ','.join(impurities)
            filelist.write(f'{norm}|{text}|{impurities_str}|{wav}\n')
            for imp in impurities:
                nonkr.add(imp)
    filelist.close()
    print('Total {} non korean charactors from {} lines'.format(len(nonkr), total_count-count))
    print(sorted(list(nonkr)))


def extract_lexicon(preprocess_config):
    """
    Extract lexicon and build grapheme-phoneme dictionary for MFA training
    See https://github.com/HGU-DLLAB/Korean-FastSpeech2-Pytorch
    """
    in_dir = preprocess_config["path"]["raw_path"]
    lexicon_path = preprocess_config["path"]["lexicon_path"]
    filelist = open(lexicon_path, 'a+', encoding='utf-8')

    # Load Lexicon Dictionary
    done = set()
    if os.path.isfile(lexicon_path):
        filelist.seek(0)
        for line in filelist.readlines():
            grapheme = line.split("\t")[0]
            done.add(grapheme)

    print("Extract lexicon...")
    for lab in tqdm(glob(f'{in_dir}/**/*.lab', recursive=True)):
        with open(lab, 'r', encoding='utf-8') as f:
            text = f.readline().strip("\n")
        assert text == normalize_nonchar(text), "No special token should be left."

        for grapheme in text.split(" "):
            if not grapheme in done:
                phoneme = " ".join(tokenize(grapheme, norm=False))
                filelist.write("{}\t{}\n".format(grapheme, phoneme))
                done.add(grapheme)
    filelist.close()


def apply_fixed_text(preprocess_config):
    in_dir = preprocess_config["path"]["corpus_path"]
    sub_dir = preprocess_config["path"]["sub_dir_name"]
    out_dir = preprocess_config["path"]["raw_path"]
    fixed_text_path = preprocess_config["path"]["fixed_text_path"]
    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

    fixed_text_dict = dict()
    print("Fixing transcripts...")
    with open(fixed_text_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            wav, fixed_text = line.split('|')[0], line.split('|')[1]
            clip_name = wav.split('_')[2].replace('c', 'clip_')
            fixed_text_dict[wav] = fixed_text.replace('\n', '')

            text = _clean_text(fixed_text, cleaners)
            with open(
                os.path.join(out_dir, sub_dir, clip_name, "{}.lab".format(wav)),
                "w",
            ) as f1:
                f1.write(text)

    filelist_fixed = open(f'{out_dir}/filelist.txt', 'w', encoding='utf-8')
    with open(f'{in_dir}/filelist.txt', 'r', encoding='utf-8') as filelist:
        for line in tqdm(filelist.readlines()):
            wav = line.split('|')[0]
            if wav in fixed_text_dict:
                filelist_fixed.write("|".join([line.split("|")[0]] + [fixed_text_dict[wav]] + line.split("|")[2:]))
            else:
                filelist_fixed.write(line)
    filelist_fixed.close()

    extract_lexicon(preprocess_config)
