# *****************************************************************************
#  Multi-Speaker TextMelLoader (Based on Code 1 + Speaker logic from Code 2)
# *****************************************************************************

import random
import torch
import torch.utils.data

import tacotron2_common.layers as layers
from tacotron2_common.utils import load_wav_to_torch, load_filepaths_and_text, to_gpu
from tacotron2.text import text_to_sequence

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text, speaker_id triplets
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, dataset_path, audiopaths_and_text, args):
        # Assumes file format: filepath|text|speaker_id
        self.audiopaths_and_text = load_filepaths_and_text(dataset_path, audiopaths_and_text)
        self.text_cleaners = args.text_cleaners
        self.max_wav_value = args.max_wav_value
        self.sampling_rate = args.sampling_rate
        self.load_mel_from_disk = args.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            args.filter_length, args.hop_length, args.win_length,
            args.n_mel_channels, args.sampling_rate, args.mel_fmin,
            args.mel_fmax)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_speaker_pair(self, audiopath_text_speaker):
        # [CHANGE 1] Unpack 3 items instead of 2
        # separate filename, text, and speaker_id
        audiopath, text, noise_id = audiopath_text_speaker[0], audiopath_text_speaker[1], audiopath_text_speaker[2]
        
        # --- ADD THESE TWO LINES ---
        #speaker_id = int(speaker_id)
        noise_id = int(noise_id)
        # ---------------------------
        
        len_text = len(text)
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        
        # [CHANGE 2] Return speaker_id in the tuple
        return (text, mel, len_text, noise_id)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.load(filename)
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_speaker_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized, len_text, speaker_id, noise_id]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        # [CHANGE 3] Collect speaker_ids
        len_x = []
        #speaker_ids = []
        noise_ids = [] # <--- NEW
        for i in range(len(ids_sorted_decreasing)):
            len_x.append(batch[ids_sorted_decreasing[i]][2])
            # Extracts speaker_id (index 3 from get_mel_text_speaker_pair)
            #speaker_ids.append(batch[ids_sorted_decreasing[i]][3])
            noise_ids.append(batch[ids_sorted_decreasing[i]][3]) # <--- NEW: Extract noise_id

        len_x = torch.Tensor(len_x)
        # Convert list of speaker IDs to a Tensor
        #speaker_ids = torch.LongTensor(speaker_ids)
        noise_ids = torch.LongTensor(noise_ids) # <--- NEW: Convert to tensor

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, len_x, noise_ids

def batch_to_gpu(batch):
    # [CHANGE 4] Unpack and transfer speaker_ids
    text_padded, input_lengths, mel_padded, gate_padded, \
        output_lengths, len_x, noise_ids = batch
    
    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    max_len = torch.max(input_lengths.data).item()
    mel_padded = to_gpu(mel_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    #speaker_ids = to_gpu(speaker_ids).long()
    noise_ids = to_gpu(noise_ids).long() # <--- NEW
    
    # [CHANGE 5] Add speaker_ids to input tuple 'x'
    x = (text_padded, input_lengths, mel_padded, max_len, output_lengths, noise_ids)
    y = (mel_padded, gate_padded)
    len_x = torch.sum(output_lengths)
    
    return (x, y, len_x)
