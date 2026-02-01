# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import json
import datetime as dt
import os
import numpy as np
from scipy.io.wavfile import write
from pathlib import Path

import torch

import params
from model import GradTTS
from tools.text_processing.symbols import symbols
from tools.text_processing import global_backend, text_to_phoneme, cleaned_text_to_sequence
from utils import intersperse

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

from utils import parse_filelist
from tqdm import tqdm
import csv
import pandas as pd
import shutil


HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = '/workspace/local/checkpoints/hifigan_univ_v1'

SPEAKER = None
SAVE_DIR = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=50, help='number of timesteps of reverse diffusion')
    parser.add_argument('-s', '--speaker_id', type=int, required=False, default=None, help='speaker id for multispeaker model')
    parser.add_argument('--temperature', type=float, required=False, default=1.5, help='temperature for sampling')
    args = parser.parse_args()
    return args


def get_vocoder():
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()
    return vocoder

if __name__ == '__main__':
    args = parse_args()

    print("SCRIPT ========")
    _, _, speaker_id, train_epochs = os.path.basename(args.checkpoint).split('.')[0].split('-')
    print("Speaker ID:", speaker_id)
    print("Train Epochs:", train_epochs)

    print(os.path.basename(args.checkpoint).split('.')[0].split('-'))

    SAVE_DIR =  "/workspace/local/samples/grad-tts"
    SAVE_DIR += f"/{speaker_id}/{train_epochs}"

    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # # Create output directory
    # if 'bas' in str(args.checkpoint).lower():
    #     SPEAKER = 'bas'
    #     SAVE_DIR = './out/bas'
    #     if not os.path.exists(SAVE_DIR):
    #         os.makedirs(SAVE_DIR)
    # elif 'sgs' in str(args.checkpoint).lower():
    #     SPEAKER = 'sgs'
    #     SAVE_DIR = './out/sgs'
    #     if not os.path.exists(SAVE_DIR):
    #         os.makedirs(SAVE_DIR)
    # else:
    #     raise ValueError(f"Unknown model type in checkpoint path: {args.checkpoint}")

    if not isinstance(args.speaker_id, type(None)):
        assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
        spk = torch.LongTensor([args.speaker_id]).cuda()
    else:
        spk = None

    print('Initializing Grad-TTS...')
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint {args.checkpoint} does not exist.")

    generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
    _ = generator.cuda().eval()
    print(f'Number of parameters: {generator.nparams}')

    print('Initializing HiFi-GAN...')
    vocoder = get_vocoder()

    print(f"Reading texts from {args.file}...")
    filelist = parse_filelist(args.file, split_char='|')

    # rtf_values = []

    with torch.no_grad():
        for i, line in enumerate(tqdm(filelist, desc="Synthesizing")):
            filepath, text, speaker = line[0], line[1], line[2]

            x = text_to_phoneme(text, global_backend) # uses ro language
            x = cleaned_text_to_sequence(x)
            x = intersperse(x, len(symbols))
            x = torch.LongTensor(x).cuda()[None] # TODO: might need to remove [None]
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()

            # t = dt.datetime.now()
            y_enc, y_dec, attn = generator.forward(x, x_lengths,
                                                   n_timesteps=args.timesteps,
                                                   temperature=args.temperature,
                                                   stoc=False,
                                                   spk=spk,
                                                   length_scale=0.91
                                                )
            # t = (dt.datetime.now() - t).total_seconds()

            # RTF = t * 22050 / (y_dec.shape[-1] * 256)
            # rtf_values.append(RTF)
            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

            # Filepath is a full path, extract the base path
            base_name = os.path.basename(filepath)

            write(os.path.join(SAVE_DIR, base_name), 22050, audio)

    # print('Done. Check out `out` folder for samples.')
    # rtf_df = pd.DataFrame(rtf_values, columns=['RTF'])
    # csv_file = os.path.join(SAVE_DIR, 'rtf_values.csv')
    # rtf_df.to_csv(csv_file, index=False)

    # stats = rtf_df.describe().loc[['mean', 'max', 'min', 'std']]
    # stats.index = ['Average RTF', 'Max RTF', 'Min RTF', 'Standard Deviation RTF']
    # stats_csv_file = os.path.join(SAVE_DIR, 'rtf_stats.csv')
    # stats.to_csv(stats_csv_file)

    # print(f"RTF values saved to {csv_file}")
    # print(f"RTF statistics saved to {stats_csv_file}")

    # print(f"RTF values saved to {csv_file}")
