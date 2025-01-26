from dataset_loader import load_audio_files, load_transcripts, load_spectrograms_with_transcripts, load_spectrograms_with_transcripts_in_batches
from resize_and_augment import resize_audios_mono, augment_audio, equalize_transcript_dimension
from transcript_encoder import fit_label_encoder, encode_transcripts, decode_predicted
from jiwer import wer
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.keras
import logging
import pickle
import tensorflow as tf
from new_model import LogMelgramLayer, CTCLayer
import os


BASE_DIR = os.getenv('BASE_DIR', os.path.dirname(os.path.abspath(__file__)))


class Prediction:
    def resize_audios_mono(self, audio_files, target_length):
        resized = {}
        for filename, signal in audio_files.items():
            if len(signal.shape) > 1:
                signal = np.mean(signal, axis=0)  

            length = len(signal)
            if length > target_length:
                signal = signal[:target_length]
            else:
                signal = np.pad(signal, (0, target_length - length), mode='constant')
            resized[filename] = signal

        return resized


    def handle_df_upload(self, request, secure_filename, app):
        transcripts_path = os.path.join(BASE_DIR, "../data/test/trsTest.txt")
        encoder_path = os.path.join(BASE_DIR, "../models/encoder.pkl")
        model_path = os.path.join(BASE_DIR, "../models/new_model_v1_6000.h5")
        
        if 'file' not in request.files:
            return {"status": "fail", "error": "No file part in the request"}, 400

        file = request.files['file']
        if file.filename == '':
            return {"status": "fail", "error": "No selected file"}, 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        
        sample_rate = 44100
        EXPECTED_LENGTH = 440295

        audio_files, maximum_length = load_audio_files(f"{app.config['UPLOAD_FOLDER'] }//", sample_rate, True)
        audio_files = resize_audios_mono(audio_files, EXPECTED_LENGTH)
        logging.info('loaded audio files')
        demo_audio = list(audio_files.keys())[0]

        transcripts = load_transcripts(transcripts_path)
        logging.info('loaded transcripts')



        audio_files = resize_audios_mono(audio_files, 440295)
        audio_files = augment_audio(audio_files, sample_rate)


        enc = open(encoder_path, 'rb')
        char_encoder = pickle.load(enc)

        logging.info("Loaded Saved Encoder")

        transcripts_encoded = encode_transcripts(transcripts, char_encoder)
        enc_aug_transcripts = equalize_transcript_dimension(audio_files, transcripts_encoded, 200)


        model = tf.keras.models.load_model(model_path, 
                                            custom_objects = {
                                                'LogMelgramLayer': LogMelgramLayer ,
                                                'CTCLayer': CTCLayer}
                                            )
        logging.info("Loaded Speech To Text Model")
        
        X_test, y_test = self.load_data(audio_files, enc_aug_transcripts)

        mlflow.set_tracking_uri('../')
        mlflow.keras.autolog()

        if X_test.shape[1] != 440295:
            raise ValueError(f"Expected audio length of 440295, but got {X_test.shape[1]}")

        predicted = model.predict([X_test,y_test])
        logging.info("Completed Prediction")
        predicted_trans = decode_predicted(predicted, char_encoder)
        real_trans = [''.join(char_encoder.inverse_transform(y)) for y in y_test]
        for i in range(len(y_test)):
            print("Test", i)
            print("pridicted:",predicted_trans[i])
            print("actual:",real_trans[i])
            print("word error rate:", wer(real_trans[i], predicted_trans[i]))
        
        upload_folder = app.config['UPLOAD_FOLDER']
        for file_name in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logging.info(f"Deleted file: {file_path}")
            except Exception as e:
                logging.error(f"Failed to delete file {file_path}: {e}")
                
        return {
                    "status": "success",
                    "transcript": predicted_trans
                }

    def load_data(self, audio_files, encoded_transcripts):
        X_train = []
        y_train = []
        for audio in audio_files:
            X_train.append(audio_files[audio])
            y_train.append(encoded_transcripts[audio])
        return np.array(X_train), np.array(y_train)

