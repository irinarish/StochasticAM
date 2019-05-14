#!/bin/bash

savePath='./testResults/'
gpuID=1

max_epochs=10
numTrials=100

adamArgs='adam.pt'
sgdArgs='sgd.pt'
am11Args='amAdam1_me1.pt'

for ((trial=1; trial<=numTrials; trial++)); do
  # Get trial ID
  trialNum=$trial #$((gpuID*numTrials + trial))
  trialID='trial'$trialNum

  saveName=$savePath'amAdam11_'$trialID
  CUDA_VISIBLE_DEVICES=$gpuID python3 ../../train_cnn_14jan.py --seed=$trialNum --epochs=$max_epochs --loadBestHyperparameters=$am11Args --first-epoch-log --save-filename=$saveName

  # SGD
  saveName=$savePath'sgd_'$trialID
  CUDA_VISIBLE_DEVICES=$gpuID python3 ../../train_cnn_14jan.py --seed=$trialNum --epochs=$max_epochs --loadBestHyperparameters=$sgdArgs --first-epoch-log --save-filename=$saveName

  # Adam
  saveName=$savePath'adam_'$trialID
  CUDA_VISIBLE_DEVICES=$gpuID python3 ../../train_cnn_14jan.py --seed=$trialNum --epochs=$max_epochs --loadBestHyperparameters=$adamArgs --first-epoch-log --save-filename=$saveName
done

