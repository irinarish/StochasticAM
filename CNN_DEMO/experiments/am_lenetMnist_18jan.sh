#!/bin/bash

# Which GPU to run it on?
gpuID=3

# Where to save results?
savePath='./lenetMnist_18jan/'

# These are fixed for all experiments.
max_epochs=10
dataName='mnist'
model='lenet'
numTrials=100
# It seems like I have to set this to get a preliminary result...?
save_interval=2000

# We loop through these
niters=(1)
n_weight_it=1

# LOOPED HYPERPARAMETERS---------------------------
muMults=(1 1.1)
muMaxes=(1.5)
muStart=0.01
muUpdateFreq=999999 # only update between epochs
delta_mus=(0.01 0.00001 0.0000001)             # small additive increments

# Code learning rates.
code_LRs=(0.1 1)

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
  amComboInc=0

  for n_code_it in "${niters[@]}"; do
    for BSZ in "${bszs[@]}"; do
      for weightLR in ${weight_LRs[@]}; do
        for codeLR in "${code_LRs[@]}"; do
          for muMult in "${muMults[@]}"; do
            for muFin in "${muMaxes[@]}"; do
              for dMu in "${delta_mus[@]}"; do
                # Compute parameter combination ID
                let "(amComboInc++)"
                comboID='combo'$amComboInc
          
                if [ $amComboInc -gt 19 ] || [ $trial -gt 1 ];
                then
                  # Compute save name
                  savename=$savePath'am_'$comboID$trialID
            
                  # Run alt-min experiment!
                  CUDA_VISIBLE_DEVICES=$gpuID python3 ../train_cnn_14jan.py --dataset=$dataName --seed=$trialNum --lr-codes=$codeLR --lr-weights=$weightLR --epochs=$max_epochs --n-iter-codes=$n_code_it --save-filename=$savename --save-interval=$save_interval --model=$model --batch-size=$BSZ --n-iter-weights=$n_weight_it --min-mu=$muStart --max-mu=$muFin --d-mu=$dMu --mu-update-freq=$muUpdateFreq --mult-mu=$muMult --use-validation-size=10000
                fi
              done
            done
          done
        done
      done
    done
  done
done
