from keras.models import Model
from keras.layers import Input, merge, Multiply
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Activation, Permute, Lambda, RepeatVector
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling1D, Conv1D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
import keras
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from itertools import repeat


# input dimensions  
rows, cols , channels = 40,500,1    

#model params
epochs = 200
batch_size =  32
nb_classes = 10
lr = 0.001


def sed_sad_agg(rows,cols,channels,nb_classes,lr,sed_loss_weight,sad_loss_weight):

################################# SHARED BRANCH ########################################################

	#conv1
	conv2d_1 = Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same')(input = Input(shape=(rows,cols,channels)))
	conv2d_1 = BatchNormalization(axis=-1)(conv2d_1)
	conv2d_1 = Dropout(0.30)(conv2d_1)

	#maxpool1
	MP_1 = MaxPooling2D(pool_size=(5, 1), strides=(5, 1))(conv2d_1)

	#conv2
	conv2d_2 = Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same')(MP_1)
	conv2d_2 = BatchNormalization(axis=-1)(conv2d_2)
	conv2d_2 = Dropout(0.30)(conv2d_2)

	#maxpool2
	MP_2 = MaxPooling2D(pool_size=(4, 1), strides=(4, 1))(conv2d_2)

########## SED BRANCH #######################################################################
	#conv3
	conv2d_3_sl = Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same')(MP_2)
	conv2d_3_sl = BatchNormalization(axis=-1)(conv2d_3_sl)
	conv2d_3_sl = Dropout(0.30)(conv2d_3_sl)

	#maxpool3
	MP_3_sl = MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(conv2d_3_sl)

	#reshape
	reshape1_sl = Reshape((64,-1))(MP_3_sl)
	reshape1_sl = Permute((2,1))(reshape1_sl)

	#GRU
	gru_sl = GRU(64, activation='tanh', return_sequences=True)(reshape1_sl)
	
	#sed output
	sl_out = TimeDistributed(Dense(nb_classes, activation='sigmoid'), name = 'sl_out')(gru_sl)

##################### SAD BRANCH #############################################################
	#conv3
	conv2d_3_pa = Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same')(MP_2)
	conv2d_3_pa = BatchNormalization(axis=-1)(conv2d_3_pa)
	conv2d_3_pa = Dropout(0.30)(conv2d_3_pa)

	#maxpool3
	MP_3_pa = MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(conv2d_3_pa)

	#reshape
	reshape1_pa = Reshape((64,-1))(MP_3_pa)
	reshape1_pa = Permute((2,1))(reshape1_pa)

	#GRU
	gru_pa = GRU(64, activation='tanh', return_sequences=True)(reshape1_pa)
	

	pa_out = TimeDistributed(Dense(nb_classes, activation='sigmoid'))(gru_pa)
	# sad output
	pa_out = TimeDistributed(Dense(1, activation='sigmoid'), name = 'pa_out')(pa_out)

#################################################################################################
	model = Model(input, outputs = [sl_out,pa_out], name='sed_sad_joint')
	opt = Adam(lr = lr)
	losses = {'sl_out': 'binary_crossentropy','pa_out': 'binary_crossentropy'}
	lossWeights = {'sl_out': sed_loss_weight, 'pa_out': sad_loss_weight}
	model.compile(loss=losses, loss_weights=lossWeights, optimizer=opt, metrics=['accuracy'])
	model.summary()
	return model
################################################################################################

sed_sad_agg = sed_sad_agg(rows,cols,channels,nb_classes,lr)


# feature and label
X_train = np.load('.../train_feature.npy')
X_val = np.load('../val_feature.npy')

Y_train_SED = np.load('../train_label_SED.npy')
Y_val_SED = np.load('../val_label_SED.npy')

Y_train_SAD = np.load('../train_label_SAD.npy')
Y_val_SAD = np.load('../val_label_SAD.npy')


## training
def model_train(model,X_train,Y_train_SED,Y_train_SAD,X_val,Y_val_SED,Y_val_SAD,epochs,batch_size,model_name,output_folder):
	filepath=os.path.join(output_folder,model_name + '-{epoch:02d}-{val_sl_out_loss:.2f}.hdf5')
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	hist = model.fit(X_train, {'sl_out': Y_train_SED, 'pa_out': Y_train_SAD}, validation_data=(X_val, {'sl_out': Y_val_SED, 'pa_out': Y_val_SAD}), callbacks=callbacks_list, epochs=epochs, shuffle=False,batch_size=batch_size,verbose=2)
	return hist
	
	
train_history = model_train(sed_sad_agg,X_train,Y_train_SED,Y_train_SAD,X_val,Y_val_SED,Y_val_SAD,epochs,batch_size,sed_sad_joint,output_folder)



