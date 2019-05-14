#!/bin/bash

# Which GPU to run it on?
gpuID=2

# Where to save results?
savePath='./lenetMnist_18jan/'

# These are fixed for all experiments.
max_epochs=10
dataName='mnist'
model='lenet'
numTrials=100
# It seems like I have to set this to get a preliminary result...?
save_interval=2000

# batches/epoch = (600, 120, 6)
bszs=(128)

# weight learning rates for batchsize 50
#             12   123   1234   12345
weight_LRs=(0.02 0.002 0.0002 0.00002)

for ((trial=1; trial<=numTrials; trial++)); do
  # Get trial ID
  trialNum=$trial #((gpuID*numTrials + trial))
  trialID='trial'$trialNum

  # Reset Counters
  comboInc=0

  for BSZ in "${bszs[@]}"; do
    for weightLR in ${weight_LRs[@]}; do

      if [ $trial -gt 2 ];
      then
        # Compute parameter combination ID
        let "(comboInc++)"
        comboID='combo'$comboInc
        
        # Compute save name
        savename=$savePath'sgd_'$comboID$trialID
        
        # Run SGD experiment!
        CUDA_VISIBLE_DEVICES=$gpuID python3 ../train_cnn_14jan.py --opt-method='sgd' --dataset=$dataName --seed=$trialNum --lr-weights=$weightLR --epochs=$max_epochs --save-filename=$savename --save-interval=$save_interval --model=$model --batch-size=$BSZ --use-validation-size=10000
  
  
        # Compute save name
        savename=$savePath'adam_'$comboID$trialID
  
        # Run alt-min experiment!
        CUDA_VISIBLE_DEVICES=$gpuID python3 ../train_cnn_14jan.py --opt-method='adam' --dataset=$dataName --seed=$trialNum --lr-weights=$weightLR --epochs=$max_epochs --save-filename=$savename --save-interval=$save_interval --model=$model --batch-size=$BSZ --use-validation-size=10000
      fi
    done
  done
done
