import librosa 
import numpy as np
from matplotlib import image
import os
import logging

def load_audio_files(path : str, sampling_rate : int, to_mono : bool) -> (dict, int):

  audio_files = {}
  max_length = 0
  i = 0
  files = os.listdir(path)
  for file in files:
    audio, rate = librosa.load(path+file, sr=sampling_rate, mono = to_mono)
    if len(audio)/rate <= 10:
      audio_files[file.split('.')[0]] = audio
      max_length = max(max_length,len(audio))
    i+=1
    if i%20 == 0:
      print('loaded',i,'audio files')
    if i == 12000:
      break
  logging.info('loaded {} audio files'.format(len(audio_files)))
  return audio_files, max_length

def load_transcripts(filepath : str) -> dict:
  transcripts = {}
  with open (filepath, encoding="utf-8")as f:
    for line in f.readlines():
      
      text, filename = line.split("</s>")
      text, filename = text.strip()[3:], filename.strip()[1:-1]
      transcripts[filename] = text
    logging.info('loaded {} transcripts'.format(len(transcripts)))
    return transcripts


def load_spectrograms_with_transcripts(mfcc_features : dict, encoded_transcripts : dict, path : str):
  X_train = []
  y_train = []
  for audio in mfcc_features:
    specgram = image.imread(path+f'{audio}.png')
    X_train.append(specgram)
    y_train.append(encoded_transcripts[audio])
  return np.array(X_train), np.array(y_train)

def load_spectrograms_with_transcripts_in_batches(mfcc_features : dict, encoded_transcripts : dict,
                                                 batch_size : int, batch_no : int, path : str):
  X_train = []
  y_train = []
  audio_names = list(mfcc_features.keys())
  i = batch_size*batch_no
  j = batch_size*(batch_no + 1)
  for audio in audio_names[i:j]:
    specgram = image.imread(path+f'{audio}.png')
    X_train.append(specgram)
    y_train.append(encoded_transcripts[audio])
  return np.array(X_train), np.array(y_train)
