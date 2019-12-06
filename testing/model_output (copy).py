from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Reshape
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn import metrics
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
import sklearn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

#####################################################################################################

# testing

def model_predict(model,sed_output_layer,sad_output_layer):
	
	# load model
	base_model = load_model(model)
	#opt = Adam(lr = 0.001)
	#model.compile(loss='binary_crossentropy', optimizer=adada, metrics=['accuracy'])
	base_model.summary()
	#choose the layer 
	model_sed = Model(inputs=base_model.input, outputs=base_model.get_layer(sed_output_layer).output) 
	model_sad = Model(inputs=base_model.input, outputs=base_model.get_layer(sad_output_layer).output) 
	# load test feature
	test_feature = np.load('../test_feature.npy')
	# predict output
	sed_pred = []
	sad_pred = []
	for i in range(len(test_feature)):
   		sed_pred.append(model_sed.predict(np.reshape(test_feature[i], (1,40,500,1))))
		sad_pred.append(model_sad.predict(np.reshape(test_feature[i], (1,40,500,1))))
		
	sed_pred = np.array(sed_pred)
	sed_pred = np.reshape(sed_pred, (len(sed_pred),500,10))
	sad_pred = np.array(sad_pred)
	sad_pred = np.reshape(sad_pred, (len(sad_pred),500))
	return sed_pred, sad_pred
	
##############################################################################################################	
## sed_sad_prediction_aggregation (element-wise multiplication)

sed_pred, sad_pred = model_predict('sed_sad_joint_model.hdf5','sl_out','pa_out')

def sed_sad_aggregation_function(sed_pred, sad_pred):
   agg_feature = []
   for i in range(len(sed_pred)):
      agg_vec = []
      present_feature_sed = sed_pred[i]
      present_feature_sad = sad_pred[i]
      for j in range(present_feature_sed.shape[1]):
         agg_vec.append(present_feature_sed[:,j]*present_feature_sad)
      agg_vec = np.array(agg_vec)
      agg_vec = agg_vec.T
      agg_feature.append(agg_vec)
      
   agg_feature = np.array(agg_feature)
   return agg_feature


final_sed_prediction = sed_sad_aggregation_function(sed_pred, sad_pred)

##################################################################################################################

# probability to one_hot

def prob_to_onehot(model_prediction, threshold):
   onehot_pred =[]
   for i in range(len(model_prediction)):
      present_sample = model_prediction[i]
      present_sample[present_sample >= threshold] = 1
      present_sample[present_sample < threshold] = 0
      onehot_pred.append(present_sample)
   return onehot_pred
         

final_sed_one_hot = prob_to_onehot(final_sed_prediction, 0.2)


## We investigate different threshold values on the baseline models using the validation set.  Using the best results, we chose a threshold of 0.2 for the SED (for SED model and SED_SAD_joint models) and 0.5 for the SAD predictions (for the SAD model).

