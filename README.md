# STU-GravWave
## GW Dataset Retrieval 
- Runs loops for L1 (LIGO Livingston) and H1 (LIGO Hanford) to grab from list of signals. The signal data is then cleaned and transformed into spectrogram form which is saved to a given directory
#### Modules/Libraries used
- Matplotlib
- os
- gwosc
- requests

## GW Machine Learning 2
- An old TensorFlow-based machine learning model used to identify whether or not a spectrogram is a gravitational wave. Attempted to apply a confusion matrix to the model analysis to see how signals were being classified but was unsuccessful in implementing
#### Modules/Libraries used
- matplotlib
- tensorflow
- sklearn
- numpy

## ML Denoise Cifar10
- Uses Cifar10 dataset to try and implement a denoising autoencoder onto a color image and then display the differences between a noisy and denoised image.
#### Modules/Libraries used
- matplotlib
- numpy
- tensorflow

## ML Denoise GW 2
- CURRENT PROJECT: attempting to remove noise from a GW spectrogram and leave as much of the wave signal intact as possible.
#### Modules/Libraries Used
- os
- numpy
- matplotlib
- tensorflow
