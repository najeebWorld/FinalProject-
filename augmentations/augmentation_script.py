from audiomentations import Compose, AddGaussianNoise, TimeMask, PitchShift, BandStopFilter
import numpy as np
from datasets import load_metric
from audiomentations.augmentations.mp3_compression import Mp3Compression
import soundfile as sf  # save data as wav file
import pydub  # convert wav format to mp3
import os
import glob
from pathlib import Path
import audio2numpy as a2n

path = '/home/or/Desktop/turkish/augmentations'
new_path = path + '/augmented_BandStopFilter_audio'  # for every type of augment change name here
os.makedirs(new_path)  # only once creates directory
mp3_pre_augment = glob.glob('/home/or/Desktop/turkish/train/*mp3')  # switch with path to location of mp3 files

# Change p = 0 for augmentations you dont want to use and p = 1 to augmentation you want
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.03, p=0),
    PitchShift(min_semitones=-6, max_semitones=8, p=0),
    BandStopFilter(min_center_freq=60, max_center_freq=2500, min_bandwidth_fraction=0.1, max_bandwidth_fraction=0.4,
                   p=1)
])
print('----------- Augmenting... ----------------')
for mp3 in mp3_pre_augment:
    filename = Path(mp3).stem
    fullpath = '/home/or/Desktop/turkish/train/' + filename + '.mp3'
    x, sr = a2n.audio_from_file(fullpath)
    augmented_samples = augment(samples=x, sample_rate=48000)
    sf.write(new_path + '/' + filename + '.wav', augmented_samples, 48000)
print('----------- Augmenting complete. ----------\n\n')

print('----------- exporting to mp3... ----------------')
wav_files = glob.glob(new_path + '/*wav')
for wav in wav_files:
    # print(wav)
    mp3_file = os.path.splitext(wav)[0] + '.mp3'
    sound_2 = pydub.AudioSegment.from_wav(wav)
    sound_2.export(mp3_file, format="mp3")
    os.remove(wav)
print('----------- exporting to mp3 complete. ----------\n\n')