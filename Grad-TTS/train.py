# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import json
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import write

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params
from gradtts_ro.model import GradTTS
from data import TextMelDataset, TextMelBatchCollate
from utils import plot_tensor, save_plot

from gradtts_ro.text_processing import symbols, global_backend, text_to_phoneme, cleaned_text_to_sequence, intersperse
from gradtts_ro.vocoder import HiFiGAN, AttrDict

import argparse


HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan_univ_v1'


def get_vocoder():
    """Load HiFi-GAN vocoder for synthesis during training."""
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()
    return vocoder


checkpoint_path = params.checkpoint_path
train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
add_blank = params.add_blank

log_dir = params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size
learning_rate = params.learning_rate
random_seed = params.seed
save_every = params.save_every

nsymbols = len(symbols) + 1 if add_blank else len(symbols)
n_enc_channels = params.n_enc_channels
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale

def parse_args():
    parser = argparse.ArgumentParser(description="Train Grad-TTS")

    parser.add_argument('--checkpoint_path',
                        type=str,
                        default=checkpoint_path,
                        help='Path to the model checkpoint to load')

    parser.add_argument('--n_epochs',
                        type=int,
                        default=n_epochs,
                        help='Number of training epochs')

    parser.add_argument('--log_dir',
                        type=str,
                        default=log_dir,)

    parser.add_argument('--save_every',
                        type=int,
                        default=save_every)

    parser.add_argument('--train_filelist_path',
                        type=str,
                        default=train_filelist_path,
                        help='Path to the training filelist')
    parser.add_argument('--batch_size',
                        type=int,
                        default=batch_size,
                        help='Batch size for training')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    checkpoint_path = args.checkpoint_path
    n_epochs = args.n_epochs
    log_dir = args.log_dir
    save_every = args.save_every
    train_filelist_path = args.train_filelist_path
    batch_size = args.batch_size

    print('Using checkpoint:', checkpoint_path)
    print('Using log directory:', log_dir)
    print('Using number of epochs:', n_epochs)
    print('Using save every:', save_every)
    print('Using batch size: ', batch_size)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('Initializing logger...')
    logger = SummaryWriter(log_dir=log_dir)

    print('Initializing data loaders...')
    train_dataset = TextMelDataset(train_filelist_path, global_backend, add_blank,
                                   n_fft, n_feats, sample_rate, hop_length,
                                   win_length, f_min, f_max)
    batch_collate = TextMelBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=4, shuffle=False)
    test_dataset = TextMelDataset(valid_filelist_path, global_backend, add_blank,
                                  n_fft, n_feats, sample_rate, hop_length,
                                  win_length, f_min, f_max)

    print('Initializing model...')
    model = GradTTS(nsymbols, 1, None, n_enc_channels, filter_channels, filter_channels_dp,
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                    n_feats, dec_dim, beta_min, beta_max, pe_scale).cuda()
    print('Number of encoder + duration predictor parameters: %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters: %.2fm' % (model.decoder.nparams/1e6))
    print('Total parameters: %.2fm' % (model.nparams/1e6))

    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
    else:
        print('Initializing model from scratch...')

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    print('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=params.test_size)
    # for i, item in enumerate(test_batch):
    #     mel = item['y']
    #     logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
    #                      global_step=0, dataformats='HWC')
    #     save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')

    print('Start training...')
    iteration = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        with tqdm(loader, total=len(train_dataset)//batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                dur_loss, prior_loss, diff_loss = model.compute_loss(x, x_lengths,
                                                                     y, y_lengths,
                                                                     out_size=out_size)
                loss = sum([dur_loss, prior_loss, diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),
                                                               max_norm=1)
                optimizer.step()

                logger.add_scalar('training/duration_loss', dur_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/prior_loss', prior_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                  global_step=iteration)


                msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}'
                progress_bar.set_description(msg)

                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                iteration += 1
                # =========================== END OF TRAIN LOOP

        log_msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
        log_msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        log_msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(log_msg)

        if epoch % save_every > 0:
            continue

        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")

        # Syhthesis once every 50 epochs
        if epoch % 10 != 0:
            continue

        vocoder = get_vocoder()
        model.eval()
        print('Synthesis...')
        with torch.no_grad():
            for i, item in enumerate(test_batch):
                x = item['x'].to(torch.long).unsqueeze(0).cuda()
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()

                y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=50)
                audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
                write(f'{log_dir}/sample_{i}_epoch{epoch}.wav', 22050, audio)
                break

                # logger.add_image(f'image_{i}/generated_enc',
                #                  plot_tensor(y_enc.squeeze().cpu()),
                #                  global_step=iteration, dataformats='HWC')
                # logger.add_image(f'image_{i}/generated_dec',
                #                  plot_tensor(y_dec.squeeze().cpu()),
                #                  global_step=iteration, dataformats='HWC')
                # logger.add_image(f'image_{i}/alignment',
                #                  plot_tensor(attn.squeeze().cpu()),
                #                  global_step=iteration, dataformats='HWC')
                # save_plot(y_enc.squeeze().cpu(),
                #           f'{log_dir}/generated_enc_{i}.png')
                # save_plot(y_dec.squeeze().cpu(),
                #           f'{log_dir}/generated_dec_{i}.png')
                # save_plot(attn.squeeze().cpu(),
                #           f'{log_dir}/alignment_{i}.png')
        del vocoder



if __name__ == "__main__":
    main()