# POLYPHONIC-SOUND-EVENT-AND-SOUND-ACTIVITY-DETECTION
POLYPHONIC SOUND EVENT AND SOUND ACTIVITY DETECTION:A MULTI-TASK APPROACH


Current polyphonic SED systems fail to model the temporal structure of sound events explicitly and instead attempt to look at which sound events are present at each audio frame.  Consequently, the event-wise detection performanceis much lower than the segment-wise detection performance. In this work, we propose a joint model approach to improve the temporal localization of sound events using a multi-task learning setup.  The first task predicts which sound events are present at each time frame; we call this branch ‘Sound Event Detection (SED) model’, while the second task predicts if a sound event is present or not at each frame; we call this branch ‘Sound Activity Detection (SAD) model’. We verify the proposed joint model by comparing it with a separate implementation of both tasks aggregated together from individual task predictions. Our experiments on the URBAN-SED dataset showthat the proposed joint model can alleviate False Positive (FP) and False Negative (FN) errors and improve both the segment-wise and the event-wise metrics.



# Description

         /feature_extration - this folder contains code and associated files for feature extraction
         /training - baseline models for SED, and SAD, SED_SAD_joint model
         /testing - code for model prediction
         /evaluation - code for SED evaluation
         /best_models - best SED_SAD_joint model

 

# Publication

[Pankajakshan A, Bear H, Benetos E. Polyphonic sound event and sound activity detection: a multi-task approach. IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA 2019), New Paltz, NY, USA, 20 Oct 2019 - 23 Oct 2019. ](https://arxiv.org/abs/1907.05122)
                
                
                
# References

[1]  J. Salamon, D. MacConnell, M. Cartwright, P. Li, and J. P.Bello, “Scaper: A library for soundscape synthesis and aug-mentation,” in IEEE Workshop on Applications of Signal Processing  to  Audio  and  Acoustics  (WASPAA),  2017,  pp.  344–348.


[2] A. Mesaros, T. Heittola, and T. Virtanen, “Metrics for polyphonic sound event detection, ”Applied Sciences, vol. 6, no. 6,p. 162, 2016.
