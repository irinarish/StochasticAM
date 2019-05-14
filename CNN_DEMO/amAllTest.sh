#!/bin/bash
# Convolutional experiments for alt-min!
# All internal loops have just ONE iteration.

# Which GPU to run it on?
gpuID=0

# Where to save results?
savePath='./test/'

# These are fixed for all experiments.
max_epochs=2
numTrials=2
# It seems like I have to set this to get a preliminary result...?
save_interval=2000

# LOOPED HYPERPARAMETERS---------------------------
# Increment mu by this amount after each minibatch.
#              1234567   1234
muSchedules=(0.0000001 0.0001)

# Code learning rates.
code_LRs=(0.01 1)

# weight learning rates for batchsize 50
weight_LRs=(0.00005 0.0005 0.005 0.05)

# FIXED HYPERPARAMETERS-----------------------------
# Sparsity parameters.
sprs_codes=(0)
sp_weights=(0)

# Subproblem iterations.
ncodeitrs=(5)
nweightitrs=(5)

# batches/epoch = 58.59
bszs=(4096)

# RNN parameters
hidSizes=(20)
seqFacts=(1)

# For Nesterov SGD subproblem
momentumVal=0.9

for ((trial=1; trial<=numTrials; trial++)); do
  # Get trial ID
  trialNum=$trial #$((gpuID*numTrials + trial))
  trialID='trial'$trialNum

  # Reset Counters
  comboInc=0

  for BSZ in "${bszs[@]}"; do
    for LR in "${weight_LRs[@]}"; do    #active loop
      for hidSz in "${hidSizes[@]}"; do
        for seqFct in "${seqFacts[@]}"; do
          for codeLR in "${code_LRs[@]}"; do  #active loop
            for d_mu in "${muSchedules[@]}"; do  #active loop
              for spCode in "${sprs_codes[@]}"; do
                for wCode in "${sp_weights[@]}"; do
                for codeIter in "${ncodeitrs[@]}"; do
                  for weightIter in "${nweightitrs[@]}"; do
  
                    # Parameter combination ID.
                    let "(comboInc++)"
                    comboID='combo'$comboInc

                    #------------------------------------------
                    # Alt-Min-Adam
                    # Compute save name
                    savename=$savePath'amAdam_'$comboID$trialID
        
                    # Run alt-min experiment!
                    CUDA_VISIBLE_DEVICES=$gpuID python3 ../train_rnn_5jan.py --opt-method='altmin' --internalAlg='adam' --rnnHidSize=$hidSz --rnnOutSeqFact=$seqFct --lr-codes=$codeLR --lr-weights=$LR --n-iter-codes=$codeIter --n-iter-weights=$weightIter --lambda_c=$spCode --lambda_w=$wCode --batch-size=$BSZ --d-mu=$d_mu --save-filename=$savename --save-interval=$save_interval --seed=$trialNum --epochs=$max_epochs

                    #------------------------------------------
                    # Alt-Min-SGD
                    # Compute save name
                    savename=$savePath'amSgd_'$comboID$trialID

                    # Run alt-min experiment!
                    CUDA_VISIBLE_DEVICES=$gpuID python3 ../train_rnn_5jan.py --opt-method='altmin' --internalAlg='sgd' --rnnHidSize=$hidSz --rnnOutSeqFact=$seqFct --lr-codes=$codeLR --lr-weights=$LR --n-iter-codes=$codeIter --n-iter-weights=$weightIter --lambda_c=$spCode --lambda_w=$wCode --batch-size=$BSZ --d-mu=$d_mu --save-filename=$savename --save-interval=$save_interval --seed=$trialNum --epochs=$max_epochs

                    #------------------------------------------
                    # Alt-Min-Nesterov (that is SGD w/Nesterov Momentum)
                    # Compute save name
                    savename=$savePath'amNext_'$comboID$trialID

                    # Run alt-min experiment!
                    CUDA_VISIBLE_DEVICES=$gpuID python3 ../train_rnn_5jan.py --opt-method='altmin' --internalAlg='sgd' --momentum=$momentumVal --rnnHidSize=$hidSz --rnnOutSeqFact=$seqFct --lr-codes=$codeLR --lr-weights=$LR --n-iter-codes=$codeIter --n-iter-weights=$weightIter --lambda_c=$spCode --lambda_w=$wCode --batch-size=$BSZ --d-mu=$d_mu --save-filename=$savename --save-interval=$save_interval --seed=$trialNum --epochs=$max_epochs

                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
