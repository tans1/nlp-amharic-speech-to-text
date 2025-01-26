import librosa
import librosa.display
import matplotlib.pyplot as plt
import logging

class FeatureExtraction:
    def extract_features(self, audios : dict, sample_rate : int) -> dict:
        if type(audios) != dict or type(sample_rate) != int:
            raise TypeError("""argument audios must be of type dict and argument sample_rate
                            must be of type int""")

        mfcc_features = {}
        for audio in audios:
            mfcc_features[audio] = librosa.feature.mfcc(audios[audio], sr=sample_rate)
        logging.info('extracted mfcc features')
        return mfcc_features

    def save_mfcc_spectrograms(self, mfccs: dict, sample_rate: int, path: str) -> int:
        if type(mfccs) != dict or type(path) != str:
            raise TypeError("""argument mfccs must be of type dict and argument path
                            must be of type string (str)""")
        for audio in mfccs:
            fig, ax = plt.subplots()
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            librosa.display.specshow(mfccs[audio], sr=sample_rate, x_axis='time')
            try:
                plt.savefig(path+f'{audio}.png')
            except FileNotFoundError:
                raise FileNotFoundError(f'The directory {path} does not exist')
            plt.close()
        logging.info('saved mfcc spectrograms')
        return 0

    def save_mel_spectrograms(self, audios: dict, sample_rate: int, path: str) -> int:
        if type(audios) != dict or type(path) != str:
            raise TypeError("""argument mfccs must be of type dict and argument path
                            must be of type string (str)""")
        for audio in audios:
            X = librosa.stft(audios[audio], n_fft = 512)
            Xdb = librosa.amplitude_to_db(abs(X))
            fig, ax = plt.subplots()
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='hz')
            try:
                plt.savefig(path+f'{audio}.png')
            except FileNotFoundError:
                raise FileNotFoundError(f'The directory {path} does not exist')
            plt.close()
        logging.info('saved mel spectrograms')
        return 0