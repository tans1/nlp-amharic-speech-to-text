import numpy as np
import librosa
import random
import logging

def resize_audios_mono(audios : dict, max_length : int) -> dict:
  for name in audios:
    audios[name] = np.pad(audios[name], 
                          (0, max_length-len(audios[name])),
                          mode = 'constant')
  logging.info('padded audio samples with silence')
  return audios


def augment_audio(audios : dict, sample_rate : int) -> dict:
  for name in audios:
    print(f"Processing: {name}, Audio shape: {audios[name].shape}")
    audios[name] = np.roll(audios[name], [-sample_rate//10,0,sample_rate//10][random.randint(0,2)])
    audios[name] = librosa.effects.time_stretch(audios[name],rate=[0.9,1,1.1][random.randint(0,2)])
    audios[name] = librosa.effects.pitch_shift(audios[name], sr=sample_rate, n_steps=[-3,0,3][random.randint(0,2)])
  logging.info('performed pitch shifting, time shifting and time stretching')
  return audios




def equalize_transcript_dimension(mfccs, encoded_transcripts, truncate_len):
  max_len = max([len(encoded_transcripts[trans]) for trans in mfccs])
  print("maximum number of characters in a transcript:", max_len)
  new_trans = {}
  for trans in mfccs:
    new_trans[trans] = np.pad(encoded_transcripts[trans], 
                          (0, max_len-len(encoded_transcripts[trans])),
                          mode = 'constant')[:truncate_len]
  logging.info('equalized transcript length')
  return new_trans