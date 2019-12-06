import os
import numpy as np
import scipy.io as sio
from scipy.io import wavfile
import librosa
import scipy.signal
import soundfile as sf


## The URBAN-SED dataset comes with pre-sorted train,validation and test sets that we use. 

# dataset paths and filenames

train_data_path = '../audio/train' 
validation_data_path = '../audio/validate'
test_data_path = '../audio/test'

train_data_filename = np.loadtxt('../train_wav.txt',dtype='str')
validation_data_filename = np.loadtxt('../validation_wav.txt',dtype='str')
test_data_filename = np.loadtxt('../test_wav.txt',dtype='str')


def melspectrogram_feature_extract(datapath,filename,n_fft,hop_length,win_length,n_mels,fps,duration):
   feature = []
   for i in range(len(filename)):
      # wav read (fs = 44.1 kHz)
      [fs, x] = wavfile.read(os.path.join(datapath,filename[i])) 
      x = np.array(x, dtype='float')
      # STFT computation (fft_points = 2048, overlap= 50%, analysis_window=40ms (44.1kHz * 40ms = 882*2))
      D = librosa.stft(x,n_fft,hop_length,win_length,scipy.signal.hamming)
      # magnitude spectra 
      D = np.abs(D)**2
      # mel transformation (mel_bands = 40)
      S = librosa.feature.melspectrogram(S=D,n_mels=n_mels)
      # power spectrogram (amplitude squared) to decibel (dB) units 
      S=librosa.power_to_db(S,ref=np.max)
      # normalization
      normS = S-np.amin(S) 
      normS = normS/float(np.amax(normS))
      # zero padding and trimming
      if int(normS.shape[1]) < fps*duration: 
          z_pad = np.zeros((n_mels,fps*duration))
          z_pad[:,:-(fps*duration-normS.shape[1])] = normS
          feature.append(z_pad)
      else:
          feat = normS[:,np.r_[0:fps*duration]] 
          feature.append(feat)
      # melspectrogram feature
   return feature



# train feature extraction
train_feature = melspectrogram_feature_extract(train_data_path,train_data_filename,2048,882,882*2,40,50,10)
train_feature = np.reshape(train_feature,(len(train_feature),40,500,1))
np.save('urban_SED_train_feature',train_feature)


# fps (frames per second) = 50
# hop_length = 882 = 20ms*44.1kHz (50% overlap)
# win_length = 882*2 = 40ms*44.1kHz
# duration of each audio file = 10 sec


