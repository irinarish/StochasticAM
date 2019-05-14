"""
Creates visualizations for CNN hyperparameter search results.

@author Benjamin Cowen
@date 17 Jan 2019
"""

############################################
# Imports ----------------------------------

# Loading
from utils import ddict, load_dataset
import glob
import torch as th 

# Plotting and Saving
# The following two lines are for the remote branch.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from shutil import copyfile
import numpy as np
import argparse


p = argparse.ArgumentParser(description='Visualize hyperparameter grid search results.')
p.add_argument('--loadPaths', type=str, default=None, metavar='path1/a/, path2/b/',
           help='A list of strings/paths to results.')
p.add_argument('--loadPath', type=str, default=None, metavar='path/to/data/',
           help='Path to directory containing data to plot.')
p.add_argument('--savePath', type=str, default=None, metavar='path/to/savedIms/',
           help='Path to directory containing images saved here.')
p.add_argument('--onlyBest', type=bool, default=False, metavar='True',
           help='Set to true if you DONT want all the imgs (only best params).')
p.add_argument('--saveBestHyperParameterPath', type=str, default=None, metavar='path/to/savedHyperParameters/',
          help='Path to directory to save the BEST hyperparameters according to the exp.')
p.add_argument('--epoch0-accs', type=str, default=None, metavar='path/to/epoch-0-accuracies/',
          help='Path to directory containing the epoch-0 accuracies... (avg result for all methods)')
p.add_argument('--reduced', type=bool, default=False, metavar='T/F',
          help='skip methods...')

args = p.parse_args()

# Check arguments.
# Assumes loadPaths or loadPath is valid...
if args.loadPaths is None:
  # Load and Save Paths
  loadPaths = [args.loadPath]
else:
  loadPaths = args.loadPaths.strip('[]').split(',')

# For saving images
if args.savePath is None:
  savePath = loadPaths[0] + "ims/"
else:
  savePath = args.savePath
if not os.path.exists(savePath):
  os.mkdir(savePath)
# For saving the best hyperparameters
if args.saveBestHyperParameterPath is None:
  saveHPpath = loadPaths[0] + "bestHypers/"
else:
  saveHPpath = args.saveBestHyperParameterPath
if not os.path.exists(saveHPpath):
  os.mkdir(saveHPpath)

print('*********************************************************')
print('Saving images to '+savePath)
print('Saving best hyperparameters to '+saveHPpath)
print('*********************************************************')

############################################
# Make a copy of this file in the savePath!
here = os.path.realpath(__file__)
there = savePath + (__file__).replace('../','')
copyfile(here,there)

############################################
# Decide tree structure here... (TODO way to automate this?)
# Stands for "all shelves"
allSh  = {"sgd":{}, "adam":{},   # Backprop-based methods
          "amAdam1_1_me1":{},
          "amAdam5_1_me1":{},
          "amAdam1_5_me1":{},
          "amAdam5_5_me1":{}
         }

# Loop thru given loadpaths.
for k,loadPath in enumerate(loadPaths):
  print('Loading files from: ' + loadPath)
  for filepath in glob.iglob(loadPath + '*.pt'):
    fileName = filepath.replace(loadPath,'')

    # Go ahead and load it.
    DD = ddict()._load(filepath)
    algBase = DD.args['opt_method']

    if not hasattr(DD,'perf'):
      continue

    # Backprop-type.
    if algBase=='sgd':
      expID = 'sgd'
    elif algBase=='adam':
      expID='adam'
    # Altmin-type.
    elif algBase=='altmin':
      nit1 = str(DD.args['n_iter_codes'])
      nit2 = str(DD.args['n_iter_weights'])
      me  = str(DD.args['mini_epochs'])
      expID='amAdam' + nit1 + '_' + nit2 + '_me' + me
    else:
      raise Exception("Unrecognized optimization method "+algBase+" (only 'altmin','sgd','adam')")

    # Now we have to determine the parameter Combination ID.
    # Location of the first digit of Combo ID. (comes right after "combo" string.)
    numStart = fileName.find('combo')+len('combo')  
    n        = 1               # Number of digits in the Combo ID.
    nextChar = fileName[numStart+n]
    while not(nextChar=="t"):  # Scrolling thru filename until the number ends.
      n += 1
      nextChar = fileName[numStart+n]
    comboNum = fileName[numStart:(numStart+n)]   # this better be an integer combo ID :)
    useKey   = "combo"+ comboNum +'_'+ str(k)        # Combination ID !!!

    # Finally, add it to results. If it's a new key, make it. 
    #  If it's just new trial, append it
    if useKey in allSh[expID]:
      # Append
      allSh[expID][useKey].append(ddict()._load(filepath))
    else:
      # Or make new
      allSh[expID][useKey] = [ddict()._load(filepath)]

# Compute the average accuracies for epoch 0 (ie, using initializations only)
epoch0_meanTr = None
if args.epoch0_accs is not None:
  epoch0_meanTr = []
  epoch0_meanTe = []
  # Compute means and std devs of the initializations.
  for filepath in glob.iglob(args.epoch0_accs + '*.pt'):
    DD = ddict()._load(filepath)
    epoch0_meanTr += DD.perf['tr']
    epoch0_meanTe += DD.perf['te']

  # Finally average.
  epoch0_stdTr  = th.Tensor( epoch0_meanTr ).std()
  epoch0_meanTr = th.Tensor( epoch0_meanTr ).mean()
  epoch0_stdTe  = th.Tensor( epoch0_meanTe ).std()
  epoch0_meanTe = th.Tensor( epoch0_meanTe ).mean()
  

############################################
# Print loss figures -----------------------
bestRez  = {}
allMeanTes={}
for expID, sh in allSh.items():
  bestRez[expID] = {"acc":-1} # Initialize each Exp ID.
  allMeanTes[expID] = {}
  for comboID, rez in sh.items():
    numTrials       = len(rez)
    numFailedTrials = 0
    
    # (0.) Collect the Training/Testing histories.
    # Compute the mean and std dev across trials.
    numEpochs = rez[0].args['epochs']              #should be same for all trials.
    allTr = None
    allTe = None
    for trial in range(numTrials):
      # Shelf for current trial.
      tShelf = rez[trial]

      # CHECK VALIDITY:-----------------
      # If no 'perf' field, then performance wasn't yet recorded.
      if not hasattr(tShelf,'perf'):
        continue
      else:
        expData = tShelf.perf

      #-------------------------------------  
      # If the number of recorded epochs << number allowed epochs,
      #  then add zeros to the end. EG force the length to be numEpochs.
      currTrAcc = th.Tensor(expData['tr'])
      currTeAcc = th.Tensor(expData['te'])

      # Double check failure to train.
#      if currTeAcc.mean()<0.2:
#        numFailedTrials+=1
#        continue
      if len(expData['tr'])<=2:
        continue
      # Otherwise you can still visualize incomplete results.
      if (len(expData['tr'])< numEpochs): 
        currTrAcc = th.cat([currTrAcc, th.zeros(numEpochs-len(expData['tr']))])
        currTeAcc = th.cat([currTeAcc, th.zeros(numEpochs-len(expData['te']))])

      # If it checks out, extract the data.
      currTrAcc = currTrAcc.view(numEpochs, 1)
      currTeAcc = currTeAcc.view(numEpochs, 1)

      # Append into a numpy array so we can get mean/std later.
      if allTr is None:
        allTr = currTrAcc
        allTe = currTeAcc
      else:
        allTr = th.cat([allTr, currTrAcc], 1)
        allTe = th.cat([allTe, currTeAcc], 1)
      # End of loop through the trials.
    # If allTr is None, NO trials were valid (for various reasons).
    # if numFailedTrials > numTrials/2, then more than half
    #    of the trials failed due to initialization randomness.
    if allTr is None or (numFailedTrials > numTrials/2):
      continue
    # (1.) Else, compute mean and std devs.
    # Update the number of trials actually used in this analysis.
    numTrials = allTr.size(1)
    # Only average the nonzero entries.
    meansTr = (allTr.sum(1)/allTr.gt(0).sum(1).float())
    meansTe = (allTe.sum(1)/allTe.gt(0).sum(1).float())
    # Initialize with zero.
    stdsTr  = th.zeros(meansTr.size()).numpy()
    stdsTe  = th.zeros(meansTe.size()).numpy()
    # If there were actually multiple trials, fill in using NONZERO entries only.
    if numTrials>1:
      for epoch, tnsr in enumerate(allTr):
        # This is my stupid way of doing ONLY NONZERO std dev's...
        stdtr  = tnsr[ tnsr.gt(0) ].std().numpy()
        if np.isnan(stdtr):
          stdtr=np.array([0])
        stdte  = allTe[epoch][ allTe[epoch].gt(0.1) ].std().numpy()
        if np.isnan(stdte):
          stdte=np.array([0])
        # Finally...
        stdsTr[epoch] = stdtr
        stdsTe[epoch] = stdte
    meansTr = meansTr.numpy()
    meansTe = meansTe.numpy()


    # (1.5) for plotting, we REMOVE the zeros added during mean-computation.
    while np.isnan(meansTr[-1]) or np.isnan(meansTe[-1]) or meansTr[-1]<1e-6 or meansTe[-1]<1e-6:
      meansTr = meansTr[:-1]
      meansTe = meansTe[:-1]
      stdsTr  = stdsTr[:-1]
      stdsTe  = stdsTe[:-1]

    if epoch0_meanTr is not None:
      meansTr = np.insert(meansTr, 0, epoch0_meanTr)
      meansTe = np.insert(meansTe, 0, epoch0_meanTe)
      stdsTr = np.insert(stdsTr, 0, epoch0_stdTr)
      stdsTe = np.insert(stdsTe, 0, epoch0_stdTe)
      x_axis = range(0, len(meansTr))
    else:
      x_axis = range(1, len(meansTr)+1)

    allMeanTes[expID][comboID]=meansTe
    ## (2.) If requested, plot the results for each parameter combination.
    if not args.onlyBest:
#    if expID=='amAdam5_me1':
      # (2.a) Extract information about this setting for titles etc.
      ar = rez[0].args
      # Model Size.

      # Alg name.
      algName = ar['opt_method']
      modelName=ar['model']
      TITLE  =modelName+' trained with '+algName

      # Print other hyperparameters
      LR      = ar['lr_weights']

      if algName in ['sgd', 'adam']:
        l1_w    = ar['lambda_w']
        TITLE2  = '\nWeight LR={}, L1={}; '.format(LR,l1_w)
      else:
        algName += '-{}_me{}'.format(ar['n_iter_weights'] , ar['mini_epochs'])
        codeLR  = ar['lr_codes']
        l1_c    = ar['lambda_c']
        muMult  = ar['mult_mu']
        muMax   = ar['max_mu']
        l1wFact = ar['lambda_w_muFact']
        l1cFact = ar['lambda_c_muFact']

        TITLE2  = '\nWeight LR={},L1={}\mu; '.format(LR,l1wFact)
        TITLE2 +=     'Code LR={},L1={}\mu; '.format(codeLR,l1cFact)
        TITLE2 += '\n\mu factor = {}; '.format(muMult)
        TITLE2 +=   '\mu-max={}'.format(muMax)

      plt.figure(1)
      plt.clf()
      plt.errorbar(x_axis, meansTr, stdsTr, marker = '.', label='Training')
      plt.errorbar(x_axis, meansTe, stdsTe, marker = '.', label='Testing')
      plt.legend()
      plt.ylim([.05,0.5])
      plt.xlabel('Epochs')
      plt.ylabel('Classification Accuracy')
      plt.title(TITLE+TITLE2)
      plt.savefig(savePath + expID + comboID + 'allTrials.png')

    ## (3.) Finally, record the best results for each setting.
    #(3.a) determine the final (nonzero) accuracy:
    if meansTe[-1] > bestRez[expID]['acc']:# and len(meansTe)>9:
      # For Plotting
      bestRez[expID]['xax']     = x_axis
      bestRez[expID]['meansTr'] = meansTr
      bestRez[expID]['stdsTr']  = stdsTr
      bestRez[expID]['meansTe'] = meansTe
      bestRez[expID]['stdsTe']  = stdsTe
      bestRez[expID]['numTr']   = numTrials
      # For later.
      bestRez[expID]['comboID'] = comboID
      bestRez[expID]['acc']     = meansTe[-1]
      bestRez[expID]['lr_weights'] = rez[0].args['lr_weights']
      bestRez[expID]['bsz'] = rez[0].args['batch_size']
      if expID.startswith('am'):
        bestRez[expID]['lr_codes'] = rez[0].args['lr_codes']
        bestRez[expID]['d_mu']     = rez[0].args['d_mu']
        bestRez[expID]['muMult']     = rez[0].args['mult_mu']
        bestRez[expID]['muMax']     = rez[0].args['max_mu']



######################################################
# Now we combine results into a plost that compare the 
#  methods using their best performing parameters.

# (0.) Each method gets a unique color.
colors  = {"sgd":'b', "adam":'k',   # Backprop-based methods
          "amAdam1_1_me1":'r',
          "amAdam5_1_me1":'y',
          "amAdam1_5_me1":'m',
          "amAdam5_5_me1":'c'
          }
markers = {"sgd":'o', "adam":'o',   
          "amAdam1_1_me1":'*',
          "amAdam5_1_me1":'d',
          "amAdam1_5_me1":'P',
          "amAdam5_5_me1":'x'
          }
plotNames={"amAdam1_1_me1":'AM-Adam11',
           "amAdam5_1_me1":'AM-Adam51',
           "amAdam1_5_me1":'AM-Adam15',
           "amAdam5_5_me1":'AM-Adam55',
           'sgd':'SGD', 'adam':"Adam"
          }
#colors = ['b','g','r','c','m','y','k']

markerSize = 4
modelName = rez[0].args['model']
resUsableTitle = modelName

dataName = rez[0].args['dataset']
trmark='.'
temark='^'
numEpochs=10
######## ALL PLOTS
for i, xLims in enumerate([[-0.1, numEpochs+0.1]]):
  for k,yLims in enumerate([[0., 1], [0.6,0.93], [0.85,1]]):
    plt.figure(i*k + k)
    plt.clf()
    colorID=0
    if yLims[0]>0 or yLims[1]<1 or xLims[1]<numEpochs:
      titleZoom = ' (zoomed) '
    else:
      titleZoom = ''
    for expID, rez in bestRez.items():
  #    print('expID = '+expID+': '+str(rez['stdsTr']))
      if rez['acc']==-1:
        continue 
      # Else, add to the axes.
      plt.errorbar(rez['xax'], rez['meansTr'], rez['stdsTr'], 
                      marker = trmark, color=colors[expID],
                      markersize = markerSize, linewidth = 1,
                      label=plotNames[expID]+' Train')
      plt.errorbar(rez['xax'], rez['meansTe'], rez['stdsTe'], 
                      marker = temark, color=colors[expID],
                      markersize = markerSize, linewidth = 1,
                      label=plotNames[expID]+' Valid (N={})'.format(rez['numTr']))
    plt.xlim(xLims)
    plt.ylim(yLims)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Optimization methods using best parameters\n'+resUsableTitle +titleZoom)
    # Get the legend off of the main plot:
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.savefig(savePath+'BEST-'+modelName+dataName+'lim'+str(i)+str(k)+'.png', bbox_inches='tight')


######################################################
# Finally, print out the best hyperparameters found.

print('***************************************************')
print('PRINTING OFFICIAL BEST HYPERPARAMETERS FOR '+resUsableTitle)
for expID, rez in bestRez.items():
  if rez['acc']==-1:
    continue 
  print('--------------------')
  print('Optimizer = '+expID)
  print('Batch-size = ' + str(bestRez[expID]['bsz']))
  print('learning rate (weights) = ' + str(bestRez[expID]['lr_weights']))
  if expID.startswith('am'):
    print('learning rate (codes) = ' + str(bestRez[expID]['lr_codes']))
    print('delta-mu= ' + str(bestRez[expID]['d_mu']))
    print('mu-Mult= ' + str(bestRez[expID]['muMult']))
    print('mu-Max= ' + str(bestRez[expID]['muMax']))
  # I was gonna just save bestRez, but it seems silly to use both pickle
  #   and ddict() even though pickle is simpler...
  # Args from an example of the winning parameter combo.
  winningArgs = allSh[expID][rez['comboID']][0].args
  SH = ddict(args=winningArgs)
  # Finally, save it.
  rezFileName = expID
  SH._save(saveHPpath + rezFileName, date=False)









