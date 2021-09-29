import re
import argparse
from string import punctuation
import os
import json

import torch
import yaml
import numpy as np
from g2p_en import G2p

from utils.model import get_vocoder
from model import FastSpeech2, ScheduledOptim
from utils.tools import to_device, synth_samples
from text import text_to_sequence
from text.korean import tokenize, normalize_nonchar


class Syn:
    def __init__(self, configs, element):
        os.chdir('/home/ubuntu/ldj/Expressive-FastSpeech2')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        self.preprocess_config, self.model_config, self.train_config = configs
        self.configs = configs
        self.duration_control=1.1
        self.energy_control =1.0
        self.pitch_control = 1.21
        self.mode = 'single'
        self.restore_step = 900000
        self.speaker_id, self.emotion_id = element
        
        self.model = FastSpeech2(self.preprocess_config, self.model_config).to(self.device)
        ckpt_path = os.path.join(
            self.train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(900000),
        )
        ckpt = torch.load(ckpt_path)
        #ckpt = torch.load(ckpt_path,map_location=torch.device('cpu'))
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.model.requires_grad_ = False
        
        self.vocoder = get_vocoder(self.model_config, self.device)

    def read_lexicon(self, lex_path):
        lexicon = {}
        with open(lex_path) as f:
            for line in f:
                temp = re.split(r"\s+", line.strip("\n"))
                word = temp[0]
                phones = temp[1:]
                if word not in lexicon:
                    lexicon[word] = phones
        return lexicon

    def preprocess_korean(self, text):
        lexicon = self.read_lexicon(self.preprocess_config["path"]["lexicon_path"])
    
        phones = []
        words = filter(None, re.split(r"([,;.\-\?\!\s+])", text))
        for w in words:
            if w in lexicon:
                phones += lexicon[w]
            else:
                phones += list(filter(lambda p: p != " ", tokenize(w, norm=False)))
        phones = "{" + "}{".join(phones) + "}"
        phones = normalize_nonchar(phones, inference=True)
        phones = phones.replace("}{", " ")
    
        print("Raw Text Sequence: {}".format(text))
        print("Phoneme Sequence: {}".format(phones))
        sequence = np.array(
            text_to_sequence(
                phones, self.preprocess_config["preprocessing"]["text"]["text_cleaners"]
            )
        )
    
        return np.array(sequence)

    def preprocess_english(self, text, preprocess_config):
        text = text.rstrip(punctuation)
        lexicon = self.read_lexicon(preprocess_config["path"]["lexicon_path"])
    
        g2p = G2p()
        phones = []
        words = filter(None, re.split(r"([,;.\-\?\!\s+])", text))
        for w in words:
            if w.lower() in lexicon:
                phones += lexicon[w.lower()]
            else:
                phones += list(filter(lambda p: p != " ", g2p(w)))
        phones = "{" + "}{".join(phones) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        phones = phones.replace("}{", " ")
    
        print("Raw Text Sequence: {}".format(text))
        print("Phoneme Sequence: {}".format(phones))
        sequence = np.array(
            text_to_sequence(
                phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
            )
        )
    
        return np.array(sequence)
    
    
    def synthesize(self, model, configs, vocoder, batchs, control_values, tag):
        preprocess_config, model_config, train_config = configs
        pitch_control, energy_control, duration_control = control_values
    
        for batch in batchs:
            batch = to_device(batch, self.device)
            with torch.no_grad():
                # Forward
                output = model(
                    *(batch[2:]),
                    p_control=pitch_control,
                    e_control=energy_control,
                    d_control=duration_control
                )
                basename = synth_samples(
                    batch,
                    output,
                    vocoder,
                    model_config,
                    preprocess_config,
                    train_config["path"]["result_path"],
                    tag,
                )
        return basename

    def inference(self, text):

        emotions = arousals = valences = None
        ids = raw_texts = [text[:100]]
        with open(os.path.join(self.preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            speaker_map = json.load(f)
        speakers = np.array([speaker_map[self.speaker_id]])
        if self.model_config["multi_emotion"]:
            with open(os.path.join(self.preprocess_config["path"]["preprocessed_path"], "emotions.json")) as f:
                json_raw = json.load(f)
                emotion_map = json_raw["emotion_dict"]
                arousal_map = json_raw["arousal_dict"]
                valence_map = json_raw["valence_dict"]
            emotions = np.array([emotion_map[self.emotion_id]])
            arousals = np.array([arousal_map['5']])
            valences = np.array([valence_map['5']])
        if self.preprocess_config["preprocessing"]["text"]["language"] == "kr":
            texts = np.array([self.preprocess_korean(text)])
        elif self.preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([self.preprocess_english(text, self.preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, emotions, arousals, valences, texts, text_lens, max(text_lens))]
        tag = f"{self.speaker_id}_{self.emotion_id}"
        control_values = self.pitch_control, self.energy_control, self.duration_control

        basename = self.synthesize(self.model, self.configs, self.vocoder, batchs, control_values, tag)
        return basename






