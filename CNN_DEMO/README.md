# Step 1: HYPERPARAMETER TESTING.
Write a hyperparameter grid search in `experiments` sub-folder.
EG, from the `experiments` subfolder, run:
`./all_lenetFashionMnist_18jan.sh`   // does both alt-min (am) and SGD for fashion-mnist
`./am_lenetMnist_18jan.sh`           // does am on mnist
`./sgd_lenetMnist_18jan.sh`          // does sgd on mnist

These files call `../train_cnn_14jan.py` in a loop, which trains LeNet architectures on training sets and record error on validation set. Make sure to set "savePath" etc. within that file to an appropriate sub-directory.


# Step 2: ANALYSIS.
(a) Still from the experiments folder, use `cnnMnistViz19jan.py` to determine the hyperparameters which performed the best, on average. E.g.:
```
python -i ../cnnMnistViz19jan.py --loadPath='./lenetMnist_18jan/'
```

This will automatically save the results in `bestHypers` and `ims` subfolders within the folder specific via argument `--loadPath`.
(b) Copy the values saved in `bestHypers` subfolder (in `.pt` format) to the corresponding subfolder of `am2-paper-results/postHPexps/`. `postHPexps` stands for "POST-HYPER-PARAMETER-EXPERIMENTS", e.g. testing and visualization.

(c) To print the selected hyperparameters to terminal, execute from `am2-paper-results` e.g.:
```
python -i printHPOrez_cnn.py --loadPath='postHPexps/lenet/lenet_mnist/'
```

# Step 3: TESTING.
In the `postHPexps` subfolders, run the executables which 
(a) Load the best hyperparameters determined in Step (2a) and copied in Step (2b).
(b) Train corresponding network using entire training set.
(c) Evaluate on the test.
(d) Repeat for many random intializations.

# Step 4: VISUALIZATION.
From the `postHPexps` subfolder, use `testRezViz_19jan.py` to generate figures displaying testing results, e.g.
python ../../../testRezViz_19jan.py --loadPath='./testResults/'

