# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from tacotron2.text import text_to_sequence
import models
import torch
import argparse
import os
import numpy as np
from scipy.io.wavfile import write
import matplotlib
import matplotlib.pyplot as plt

import sys

import time
import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

from waveglow.denoiser import Denoiser # change vocoder

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='full path to the input text (phareses separated by new line)')
    #parser.add_argument('--speaker-id', type=int, default=0, # Bago ito
    #                    help='ID of the speaker to use for inference')
    # --- NEW: ADDED N-SPEAKERS HERE ---
    #parser.add_argument('--n-speakers', type=int, default=1,
    #                    help='Number of speakers in the model')
    # ----------------------------------
    parser.add_argument('--gate-threshold', type=float, default=0.5,
                        help='Lower this to 0.2 or 0.1 if model keeps mumbling at the end')
    # --- ADDED THIS ---
    #parser.add_argument('--speakers-embedding-dim', type=int, default=256,
    #                    help='Dimension size of the speaker embedding layer')
    # ------------------
    # --- ADD THIS ---
    parser.add_argument('--noise-id', type=int, default=0,
                        help='Noise ID for inference (0 for clean, 1 for noisy)')
    # ----------------
    # --- ADDED THIS ---
    parser.add_argument('--noise-embedding-dim', type=int, default=64,
                        help='Dimension size of the noise embedding layer')
    parser.add_argument('--n_noise_types', type=int, default=4,
                        help='Dimension size of the noise embedding layer')
    # ------------------
    parser.add_argument('-o', '--output', required=True,
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('--suffix', type=str, default="", help="output filename suffix")
    parser.add_argument('--tacotron2', type=str,
                        help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('--waveglow', type=str,# change vocoder
                        help='full path to the WaveGlow model checkpoint file')# change vocoder
    # --- ADD HIFI-GAN ---
    parser.add_argument('--hifigan', type=str,
                        help='full path to the HiFi-GAN model checkpoint file')
    #parser.add_argument('--hifigan_config', type=str,
    #                    help='full path to the HiFi-GAN config.json file')
    # ---------------------------
    parser.add_argument('-s', '--sigma-infer', default=0.9, type=float)
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float)
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')

    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument('--fp16', action='store_true',
                        help='Run inference with mixed precision')
    run_mode.add_argument('--cpu', action='store_true',
                        help='Run inference on CPU')

    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    parser.add_argument('--include-warmup', action='store_true',
                        help='Include warmup')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')

    return parser


def checkpoint_from_distributed(state_dict):
    """
    Checks whether checkpoint was generated by DistributedDataParallel. DDP
    wraps model in additional "module.", it needs to be unwrapped for single
    GPU inference.
    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


def unwrap_distributed(state_dict):
    """
    Unwraps model from DistributedDataParallel.
    DDP wraps model in additional "module.", it needs to be removed for single
    GPU inference.
    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict


def load_and_setup_model(model_name, parser, checkpoint, fp16_run, cpu_run,
                         forward_is_infer=False, jittable=False):
    model_parser = models.model_parser(model_name, parser, add_help=False)
    model_args, _ = model_parser.parse_known_args()

    model_config = models.get_model_config(model_name, model_args)
    model = models.get_model(model_name, model_config, cpu_run=cpu_run,
                             forward_is_infer=forward_is_infer,
                             jittable=jittable)

    if checkpoint is not None:
        if cpu_run:
            state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))['state_dict']
        else:
            #state_dict = torch.load(checkpoint)['state_dict']
            state_dict = torch.load(checkpoint, weights_only=False)['state_dict']
        if checkpoint_from_distributed(state_dict):
            state_dict = unwrap_distributed(state_dict)

        model.load_state_dict(state_dict)

    if model_name == "WaveGlow":# change vocoder
        model = model.remove_weightnorm(model)

    model.eval()

    if fp16_run:
        model.half()

    return model


# taken from tacotron2/data_function.py:TextMelCollate.__call__
def pad_sequences(batch):
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x) for x in batch]),
        dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]]
        text_padded[i, :text.size(0)] = text

    return text_padded, input_lengths


def prepare_input_sequence(texts, cpu_run=False):

    d = []
    for i,text in enumerate(texts):
        d.append(torch.IntTensor(
            text_to_sequence(text, ['english_cleaners'])[:]))

    text_padded, input_lengths = pad_sequences(d)
    if not cpu_run:
        text_padded = text_padded.cuda().long()
        input_lengths = input_lengths.cuda().long()
    else:
        text_padded = text_padded.long()
        input_lengths = input_lengths.long()

    return text_padded, input_lengths


class MeasureTime():
    def __init__(self, measurements, key, cpu_run=False):
        self.measurements = measurements
        self.key = key
        self.cpu_run = cpu_run

    def __enter__(self):
        if not self.cpu_run:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self.cpu_run:
            torch.cuda.synchronize()
        self.measurements[self.key] = time.perf_counter() - self.t0


def main():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU or CPU.
    """
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    log_file = os.path.join(args.output, args.log_file)
    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, log_file),
                            StdOutBackend(Verbosity.VERBOSE)])
    for k,v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k:v})
    DLLogger.log(step="PARAMETER", data={'model_name':'Tacotron2_PyT'})

    tacotron2 = load_and_setup_model('Tacotron2', parser, args.tacotron2,
                                     args.fp16, args.cpu, forward_is_infer=True)

    # --- NEW: HACK TO OVERRIDE THE GATE THRESHOLD ---
    try:
        tacotron2.decoder.gate_threshold = args.gate_threshold
        print(f"🔧 Gate threshold successfully set to: {args.gate_threshold}")
    except AttributeError:
        pass
    # -----------------------------------------------

    jitted_tacotron2 = tacotron2  # Just use the normal model

    # --- CONDITIONALLY LOAD THE VOCODERS ---
    if args.waveglow is not None:
        waveglow = load_and_setup_model('WaveGlow', parser, args.waveglow,
                                        args.fp16, args.cpu, forward_is_infer=True,
                                        jittable=True)
        denoiser = Denoiser(waveglow)
        if not args.cpu:
            denoiser.cuda()

        waveglow.make_ts_scriptable()
        jitted_waveglow = torch.jit.script(waveglow)
        
    if args.hifigan is not None:
        hifigan = load_and_setup_model('HiFi-GAN', parser, args.hifigan,
                                       args.fp16, args.cpu, forward_is_infer=True)
    # ---------------------------------------

    texts = []
    try:
        f = open(args.input, 'r')
        texts = f.readlines()
    except:
        print("Could not read file")
        sys.exit(1)

    if args.include_warmup:
        sequence = torch.randint(low=0, high=148, size=(1,50)).long()
        input_lengths = torch.IntTensor([sequence.size(1)]).long()
        # --- ADD THIS ---
        # Create a dummy speaker ID (0)
        #dummy_speaker_id = torch.tensor([0]).long()
        # ----------------
        # --- ADD THIS ---
        dummy_noise_id = torch.tensor([0]).long() 
        # ----------------
        if not args.cpu:
            sequence = sequence.cuda()
            input_lengths = input_lengths.cuda()
            #dummy_speaker_id = dummy_speaker_id.cuda() # Move to GPU
            dummy_noise_id = dummy_noise_id.cuda() # Move to GPU
            
        for i in range(3):
            with torch.no_grad():
                mel, mel_lengths, _ = jitted_tacotron2(sequence, input_lengths, dummy_noise_id)
                
                # --- CONDITIONALLY WARMUP ---
                if args.waveglow is not None:
                    _ = jitted_waveglow(mel)
                elif args.hifigan is not None:
                    _ = hifigan(mel)
                # ----------------------------

    measurements = {}

    sequences_padded, input_lengths = prepare_input_sequence(texts, args.cpu)

    # --- ADD THIS ---
    # Create the speaker tensor for the whole batch
    # We repeat the speaker_id for every sentence in the input text file
    batch_size = sequences_padded.size(0)
    #speaker_ids = torch.tensor([args.speaker_id] * batch_size).long()
    # --- ADD THIS ---
    noise_ids = torch.tensor([args.noise_id] * batch_size).long()
    # ----------------

    if not args.cpu:
        #speaker_ids = speaker_ids.cuda()
        noise_ids = noise_ids.cuda() # <--- ADD THIS

    with torch.no_grad(), MeasureTime(measurements, "tacotron2_time", args.cpu):
        mel, mel_lengths, alignments = jitted_tacotron2(sequences_padded, input_lengths, noise_ids)

    # --- CONDITIONALLY RUN VOCODER INFERENCE ---
    if args.waveglow is not None:
        with torch.no_grad(), MeasureTime(measurements, "waveglow_time", args.cpu):
            audios = jitted_waveglow(mel, sigma=args.sigma_infer)
            audios = audios.float()
        with torch.no_grad(), MeasureTime(measurements, "denoiser_time", args.cpu):
            audios = denoiser(audios, strength=args.denoising_strength).squeeze(1)

    elif args.hifigan is not None:
        # Properly labeled "hifigan_time", and no dummy variables needed!
        with torch.no_grad(), MeasureTime(measurements, "hifigan_time", args.cpu): 
            audios = hifigan(mel).squeeze(1)
            audios = audios.float()
    # -------------------------------------------

    print("Stopping after",mel.size(2),"decoder steps")
    tacotron2_infer_perf = mel.size(0)*mel.size(2)/measurements['tacotron2_time']

    DLLogger.log(step=0, data={"tacotron2_items_per_sec": tacotron2_infer_perf})
    DLLogger.log(step=0, data={"tacotron2_latency": measurements['tacotron2_time']})

    # --- CONDITIONALLY LOG VOCODER STATS ---
    if args.waveglow is not None:
        waveglow_infer_perf = audios.size(0)*audios.size(1)/measurements['waveglow_time']
        DLLogger.log(step=0, data={"waveglow_items_per_sec": waveglow_infer_perf})
        DLLogger.log(step=0, data={"waveglow_latency": measurements['waveglow_time']})
        DLLogger.log(step=0, data={"denoiser_latency": measurements['denoiser_time']})
        DLLogger.log(step=0, data={"latency": (measurements['tacotron2_time'] + measurements['waveglow_time'] + measurements['denoiser_time'])})

    elif args.hifigan is not None:
        hifigan_infer_perf = audios.size(0)*audios.size(1)/measurements['hifigan_time']
        DLLogger.log(step=0, data={"hifigan_items_per_sec": hifigan_infer_perf})
        DLLogger.log(step=0, data={"hifigan_latency": measurements['hifigan_time']})
        # HiFi-GAN has no denoiser, so we just add Tacotron + HiFi-GAN
        DLLogger.log(step=0, data={"latency": (measurements['tacotron2_time'] + measurements['hifigan_time'])})
    # ---------------------------------------

    for i, audio in enumerate(audios):

        plt.imshow(alignments[i].float().data.cpu().numpy().T, aspect="auto", origin="lower")
        figure_path = os.path.join(args.output,"alignment_"+str(i)+args.suffix+".png")
        plt.savefig(figure_path)

        audio = audio[:mel_lengths[i]*args.stft_hop_length]
        audio = audio/torch.max(torch.abs(audio))
        audio_path = os.path.join(args.output,"audio_"+str(i)+args.suffix+".wav")
        write(audio_path, args.sampling_rate, audio.cpu().numpy())

    DLLogger.flush()

if __name__ == '__main__':
    main()
