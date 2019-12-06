import os
import numpy as np

import numpy as np

def prob_to_onehot(model_prediction, threshold):
   onehot_pred =[]
   for i in range(len(model_prediction)):
      present_sample = model_prediction[i]
      present_sample[present_sample >= threshold] = 1
      present_sample[present_sample < threshold] = 0
      onehot_pred.append(present_sample)
   return onehot_pred
         

## We investigate different threshold values on the baseline models using the validation set.  Using the best results, we chose a threshold of 0.2 for the SED and onset predictions and 0.5 for the SAD predictions.
## convert the onehot predictions into corresponding onsets, offsets, and labels

################################################### SED-EVAL ##############################################
import sed_eval
import dcase_util

# ref_path contains true annotations with "onset" "offset" and "event_label" 
ref_path = '../annotations/test'
# est_path contains predicted annotations with "onset" "offset" and "event_label" 
est_path = '../annotations/predict'
   
    
file_list=[]
pairfile='../file_list_sed_eval' ## filenames: 1st column - original filename, 2nd column - predited filename
with open(pairfile,'r') as pf:
    for line in pf.readlines():
        line=line.strip().split('\t')
        d={}
        d['reference_file']=ref_path+'/'+line[0]
        d['estimated_file']=est_path+'/'+line[1]
        file_list.append(d)

data = []
all_data = dcase_util.containers.MetaDataContainer()
for file_pair in file_list:
    reference_event_list = sed_eval.io.load_event_list(filename=file_pair['reference_file'])
    estimated_event_list = sed_eval.io.load_event_list(filename=file_pair['estimated_file'])
    data.append({'reference_event_list': reference_event_list, 'estimated_event_list': estimated_event_list})
    all_data += reference_event_list


print(len(data))
event_labels = all_data.unique_event_labels
print(event_labels)


###########################################################################################
# segment based metrics


def segment_based_metrics(event_labels,time_resolution):
	segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(event_label_list=event_labels,time_resolution=time_resolution)
	for file_pair in data:
    	segment_based_metrics.evaluate(reference_event_list=file_pair['reference_event_list'],estimated_event_list=file_pair['estimated_event_list'])    	 	
    print (segment_based_metrics)
    

time_resolution = 1.0 # Segment size used in the evaluation, in seconds.
segment_based_metrics(event_labels,time_resolution)

################################################################################################
# event based metrics

def event_based_metrics(event_labels,t_collar):
	event_based_metrics = sed_eval.sound_event.EventBasedMetrics(event_label_list=event_labels,t_collar=t_collar, evaluate_onset=True, evaluate_offset=False)
	for file_pair in data:
    	segment_based_metrics.evaluate(reference_event_list=file_pair['reference_event_list'],estimated_event_list=file_pair['estimated_event_list'])
    print(event_based_metrics)


t_collar = 0.250 # Time collar used when evaluating validity of the onset
event_based_metrics(event_labels,t_collar)

###########################################################################################W
