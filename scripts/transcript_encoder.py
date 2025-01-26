from sklearn.preprocessing import LabelEncoder
import numpy as np
import logging

def fit_label_encoder(transcripts : dict) -> LabelEncoder:
  characters = []
  for audio in transcripts:
    characters.extend(list(transcripts[audio]))
  encoder = LabelEncoder()
  encoder.fit_transform(characters)
  return encoder

def encode_transcripts(transcripts : dict, encoder : LabelEncoder) -> dict:
  transcripts_encoded = {}
  for audio in transcripts:
    transcripts_encoded[audio] = encoder.transform(list(transcripts[audio]))
  
  logging.info('encoded transcripts using integers')
  return transcripts_encoded

def decode_predicted(pred,encoder):
  dec = []
  for a in pred:
    l = [np.argmax(b) for b in a]
    newl = []
    for i in range(len(l)-1):
      if l[i]!=222 and l[i+1]!=l[i]:
        newl.append(l[i])
    if l[-1] != 222:
      newl.append(l[-1])
    dec.append(''.join(encoder.inverse_transform(newl)).strip())
    logging.info('decoded predictions')
  return dec