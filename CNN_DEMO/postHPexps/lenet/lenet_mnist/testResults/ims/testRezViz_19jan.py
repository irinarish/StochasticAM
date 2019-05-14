"""
Creates visualizations for RNN test results.

@author Benjamin Cowen
@date 19 Jan 2019
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
p.add_argument('--reduced', type=bool, default=False, metavar='True',
           help='Set to true if you only want SGD, Adam, and AltMin-Adam.')
p.add_argument('--epoch0-accs', type=str, default=None, metavar='path/to/epoch-0-accuracies/',
          help='Path to directory containing the epoch-0 accuracies... (avg result for all methods)')
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
  if not os.path.exists(savePath):
    os.mkdir(savePath)
print('*********************************************************')
print('Saving images to '+savePath)
print('*********************************************************')

############################################
# Make a copy of this file in the savePath!
here = os.path.realpath(__file__)
there = savePath + (__file__).replace('../','')
copyfile(here,there)

############################################
# Decide tree structure here... (TODO way to automate this?)
# Stands for "all shelves"
allSh  = {"amAdam1_me1":[], 
          "amAdam5_me1":[], 
          "sgd":[], "adam":[]
         }


# Loop thru given loadpaths.
for k,loadPath in enumerate(loadPaths):
  print('Loading files from: ' + loadPath)
  for filepath in glob.iglob(loadPath + '*.pt'):
    fileName = filepath.replace(loadPath,'')

    # Go ahead and load it.
    DD = ddict()._load(filepath)
    algBase = DD.args['opt_method']

    # If SGD, check whether it's Nesterov.
    if algBase=='sgd':
      expID = 'sgd'
    elif algBase=='adam':
      expID='adam'
    elif algBase=='altmin':
      nit = str(DD.args['n_iter_codes'])
      me  = str(DD.args['mini_epochs'])
      expID='amAdam' + nit + '_me' + me
    else:
      raise Exception("Unrecognized optimization method "+algBase+" (only 'altmin','sgd','adam')")

    # Append results.
    if args.reduced:
      if not (expID=='sgd' or expID=='adam' or expID=='amAdam5_me1'):
        continue
    allSh[expID].append(ddict()._load(filepath))


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
  

# This better be the same for all DD anyway :)
############################################
# Print all single-exp. figures -----------------------
avgRez  = {}
numE0btch = 469 #59#300  # Number of elements in 'first_epoch'
numEpochs = 10   # Number of elements in 'te' and 'tr'

for expID, rez in allSh.items():
  if len(rez)==0:
    continue
  # Initialize each Exp ID.
  avgRez[expID] = {'tr':[], 'te':[]}
  numTrials = len(rez)

  # (0.) Collect the Training/Testing histories.
  # Compute the mean and std dev across trials.
  allR0 = None
  for trial in range(numTrials):
    # Shelf for current trial.
    tShelf = rez[trial]

    # CHECK VALIDITY:-----------------
    # If no 'perf' field, then performance wasn't yet recorded.
    if hasattr(tShelf,'perf'):
      expData = tShelf.perf
      r0  = expData['first_epoch']
      rTr = expData['tr']
      rTe = expData['te']
    else:
      continue

    # If the number of recorded epochs << number allowed epochs, it's in-progress.
    # If the mean training accurac < 14%, consider this a FAILED attempt at training.
    if len(rTe)<numEpochs or np.mean(r0)<0.14 or np.mean(rTr)<0.14 or len(r0)==0:
      continue

    # Crazy occasional results for SGD here:
    if 'dataset' in tShelf.args:
      if tShelf.args['dataset']=='mnist':
        if rTe[7]<0.98:
          continue
    # Finally, might need to remove FAILED attempts.
    #----------------------------------

    # If it checks out, extract the data.
    r0  = th.Tensor(r0).view(numE0btch, 1)
    rTr = th.Tensor(rTr).view(numEpochs, 1)
    rTe = th.Tensor(rTe).view(numEpochs, 1)

    # Append into a numpy array so we can get mean/std later.
    if allR0 is None:
      allR0 = r0
      allTr = rTr
      allTe = rTe
    else:
      allR0 = th.cat([allR0, r0], 1)
      allTr = th.cat([allTr, rTr], 1)
      allTe = th.cat([allTe, rTe], 1)
    # End of loop through the trials.
  # If no trials were valid, skip!
  if allR0 is None:
    continue

#  if expID.startswith('sgd'):
#    1/0

  # Need this for later...
  # (1.) Else, compute mean and std devs.
  # Update the number of trials actually used in this analysis.
  numTrials = allR0.size(1)

  # There shouldn't be any zero entries anymore...
  meansR0 = allR0.mean(1).numpy()
  meansTr = allTr.mean(1).numpy()
  meansTe = allTe.mean(1).numpy()
  if numTrials>0:
    stdsR0 = allR0.std(1).numpy()
    stdsTr = allTr.std(1).numpy()
    stdsTe = allTe.std(1).numpy()
  else:
    stdsR0 = th.zeros(meansR0.size()).numpy()
    stdsTr = th.zeros(meansTr.size()).numpy()
    stdsTe = th.zeros(meansTe.size()).numpy()    


  ###############
  # For epoch-0 results to be tacked on..
  x_axis = np.array(range(1, len(meansTr)+1))
#  meansTr = np.insert(meansTr, 0, meansR0[0])
#  meansTe = np.insert(meansTe, 0, meansR0[0])
#  stdsTr  = np.insert(stdsTr,  0, stdsR0[0])
#  stdsTe  = np.insert(stdsTe,  0, stdsR0[0])
#  x_axis = np.array(range(0, len(meansTr)))

  # Finally divide by the sqrt number of samples:
  
  stdsR0  /= np.sqrt(numTrials)
  stdsTr  /= np.sqrt(numTrials)
  stdsTe  /= np.sqrt(numTrials)
  
  ## (2.) If requested, plot the results for each parameter combination.
  if not args.onlyBest:
    # (2.a) Extract information about this setting for titles etc.
    # These should NOT change for rez[i], i=/=0.
    LR      = str(rez[0].args['lr_weights'])
    # Data type (permuted vs sequential mnist)
    if 'data_type' in rez[0].args:           #This arg was added late so may not exist.
      dataType = rez[0].args['data_type']
    else:
      dataType = 'seq'
    # Alg name.
    algName = rez[0].args['opt_method']
    if algName=='altmin':
      algName += str(rez[0].args['n_iter_weights'])
    if 'rnnHidSize' in rez[0].args:
      modelName='RNN 1-{}'.format(rez[0].args['rnnHidSize'])
    else:
      modelName=rez[0].args['model']
    TITLE  = modelName +'; trained with '+expID

    plt.figure(1)
    plt.clf()
    plt.errorbar(x_axis, meansTr, stdsTr, marker = 'o', label='Training')
    plt.errorbar(x_axis, meansTe, stdsTe, marker = '^', label='Testing')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Classification Accuracy')
    plt.title(TITLE)
  #  plt.ylim([0.875, 1.05])
    plt.savefig(savePath + expID + 'allEpoch.png')

    x_axis0 = np.array(range(0, len(meansR0)))
    plt.figure(2)
    plt.clf()
    plt.errorbar(x_axis0, meansR0, stdsR0, marker = '.', label='Testing')
    plt.xlabel('Minibatches')
    plt.ylabel('Classification Accuracy')
    plt.title(TITLE+'\n(Epoch 0 Minibatches)')
  #  plt.ylim([0.875, 1.05])
    plt.savefig(savePath + expID + 'minibatchE0.png')

  ## (3.) Finally, record the avg for a big plot at the end.
  # For Plotting
  avgRez[expID]['xax']     = x_axis
  avgRez[expID]['xax0']    = x_axis0
  avgRez[expID]['meansTr'] = meansTr
  avgRez[expID]['meansTe'] = meansTe
  avgRez[expID]['meansR0'] = meansR0
  avgRez[expID]['stdsTr']  = stdsTr
  avgRez[expID]['stdsTe']  = stdsTe
  avgRez[expID]['stdsR0']  = stdsR0
  avgRez[expID]['numTr']   = numTrials


######################################################
# Now we combine results into a plost that compare the 
#  methods using their best performing parameters.
# (0.) Each method gets a unique color.
colors  = {"sgd":'b', "adam":'k',
           "amAdam1_me1":'r',
           "amAdam5_me1":'r'
          }
markers  = {"sgd":'o', "adam":'P',
           "amAdam1_me1":'^',
           "amAdam5_me1":'^'
           }
names  = {"sgd":'SGD', "adam":'Adam',
          "amAdam1_me1":'AM-Adam',
           "amAdam5_me1":'AM-Adam'}
models={'lenet':'LeNet', 'vgg7':'VGG7'}
datasets={'mnist':'MNIST', 'fashion_mnist':'Fashion-MNIST'}
#colors = ['b','g','r','c','m','y','k']
if 'rnnHidSize' in rez[0].args:
  modelName='RNN 1-{}'.format(rez[0].args['rnnHidSize'])
  datName='Sequential MNIST'
else:
  modelName=models[rez[0].args['model']]
  datName=datasets[rez[0].args['dataset']]
resUsableTitle = datName+', '+modelName


if datName=='MNIST':
  ylim1 = [0.96, 0.99]
  ylim2 = [0.1,1]
  forget='amAdam5_me1'
  numMini=None
elif datName=='Fashion-MNIST':
  ylim1 = [0.78,0.9]
  ylim2 = [0.1,0.8]
  forget='amAdam1_me1'
  numMini=150
elif datName=='Sequential MNIST':
  if modelName[-2:]=='50':
    ylim1 = [0.3,0.965]
    ylim2 = [0.1,1]
  elif modelName[-2:]=='15':
    ylim1 = [0.775,0.93]
    ylim2 = [0.1,0.96]
    if numEpochs==50:
      ylim1=[0.775, 0.96]
  forget=''
  numMini=None

epoch0Display=0.9
if datName=='Sequential MNIST':
  xtras1={'adam':1.001, 'amAdam1_me1':1, 'amAdam5_me1':1, 'sgd':0.99}
  xtras2={'adam':1.03, 'amAdam1_me1':1,'amAdam5_me1':1, 'sgd':0.999}
else:
  if rez[0].args['batch_size']==200:
    if datName=='Fashion-MNIST':
      zzz=0.999
      xtras1={'adam':1.005, 'amAdam1_me1':zzz, 'amAdam5_me1':zzz, 'sgd':1}
      xtras2={'adam':1.06, 'amAdam1_me1':zzz,'amAdam5_me1':zzz, 'sgd':1}
      epoch0Display = 0.8
    else:
      zzz=0.999
      xtras1={'adam':1.001, 'amAdam1_me1':zzz, 'amAdam5_me1':zzz, 'sgd':1}
      xtras2={'adam':1.06, 'amAdam1_me1':0.95,'amAdam5_me1':0.95, 'sgd':1}
      epoch0Display = 1
  else:
    xtras1={'adam':1.0001, 'amAdam1_me1':1, 'amAdam5_me1':1, 'sgd':0.99}
    xtras2={'adam':1.06, 'amAdam1_me1':1,'amAdam5_me1':1, 'sgd':0.999}

xtraAx1 = 0.1
xtraAx2 = 1
trmark='.'
temark='^'
                        #mnistAll  #fMnistAll   
plt.figure(1)
plt.clf()
# Plot all results.
for expID, rez in avgRez.items():
  if not ('meansTr' in rez) or expID==forget:
    continue
#    plt.errorbar(rez['xax'], rez['meansTr'], rez['stdsTr'], 
#                    marker = trmark, color=colors[expID],
#                    markersize = 4, linewidth = 1,
#                    label=names[expID]+' TR')
  yvals =rez['meansTe']
  plt.errorbar(rez['xax'], yvals, rez['stdsTe'], 
                  marker = temark, color=colors[expID],
                  markersize = 4, linewidth = 1,
                  label=names[expID]+' (N={})'.format(rez['numTr']))
  # Plot the final accuracy to the right of the axes.
  plt.annotate('%0.3f' % yvals[-1], xy=(1,yvals[-1]*xtras1[expID]), xytext=(8,0),
                xycoords=('axes fraction', 'data'), textcoords='offset points')
# Now plot a marker designating when TESTING phase began.
#  plt.axvline(x=59.5, linewidth=2, label='Begin testing')
plt.xlim([min(rez['xax'])-xtraAx1, max(rez['xax'])+xtraAx1])
plt.ylim(ylim1)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title(resUsableTitle)
plt.legend()#loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)
plt.savefig(savePath+'bestEpochs_22jan-'+modelName+datName+'.png', bbox_inches='tight')
###########################################
## Now do epoch-0.
plt.figure(2)
plt.clf()

END=max(rez['xax0'])
# Plot all results.
for expID, rez in avgRez.items():
  if not ('meansR0' in rez) or expID==forget:
    continue
  yvals = rez['meansR0']
  plt.errorbar(rez['xax0'], yvals, rez['stdsR0'], 
                  marker = markers[expID], color=colors[expID],
                  markersize = 4, linewidth = 1,
                  label=names[expID]+' (N={})'.format(rez['numTr']))
  # Plot the final accuracy to the right of the axes.
  plt.annotate('%0.3f' % yvals[END], xy=(1,yvals[END]*xtras2[expID]), xytext=(8,0),
                xycoords=('axes fraction', 'data'), textcoords='offset points')
# Now plot a marker designating when TESTING phase began.
#  plt.axvline(x=59.5, linewidth=2, label='Begin testing')
plt.xlim([min(rez['xax0'])-xtraAx2, END+xtraAx2])
plt.ylim(ylim2)
plt.xlabel('Minibatches')
plt.ylabel('Accuracy')
plt.title(resUsableTitle)
plt.legend()#loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)
plt.savefig(savePath+'bestMiniBatches_22jan-'+modelName+datName+'.png', bbox_inches='tight')





######################################
## Combined plots.

fig = plt.figure(23)
fig.clf()
ax1 = fig.add_subplot(111)

# Plot all results.
for expID, rez in avgRez.items():
  if not ('meansTr' in rez) or expID==forget:
    continue
#    plt.errorbar(rez['xax'], rez['meansTr'], rez['stdsTr'], 
#                    marker = trmark, color=colors[expID],
#                    markersize = 4, linewidth = 1,
#                    label=names[expID]+' TR')
  yvals = rez['meansTe']
  ax1.errorbar(rez['xax'], yvals, rez['stdsTe'], 
                  marker = temark, color=colors[expID],
                  markersize = 4, linewidth = 1,
                  label=names[expID]+' (N={})'.format(rez['numTr']))
  # Plot the final accuracy to the right of the axes.
  ax1.annotate('%0.3f' % yvals[-1], xy=(1,yvals[-1]*xtras1[expID]), xytext=(8,0),
                xycoords=('axes fraction', 'data'), textcoords='offset points')
# Now set axis  limits etc.
ax1.set_xlim([min(rez['xax'])-xtraAx1, max(rez['xax'])+xtraAx1])
ax1.set_ylim(ylim1)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.set_title(resUsableTitle)
ax1.legend(loc=(.05,.0),fontsize='small')
#loc=(0.5,0.65))
#'best')#'lower left')
#loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)

###########################################
## Now do epoch-0 inside the above axis...
ax2=plt.axes([.45, .2, .45, .4])

if numMini is not None:
  END=numMini
else:
  END=len(rez['meansR0'])-1
for expID, rez in avgRez.items():
  if not ('meansR0' in rez) or expID==forget:
    continue
  yvals = rez['meansR0']
  ax2.errorbar(rez['xax0'], yvals, rez['stdsR0'], 
                  marker = markers[expID], color=colors[expID],
                  markersize = 4, linewidth = 1,
                  label=names[expID]+' (N={})'.format(rez['numTr']))
  # Plot the final accuracy to the right of the axes.
  ax2.annotate('%0.3f' % yvals[END], xy=(1,yvals[END]*xtras2[expID]), xytext=(8,0),
                xycoords=('axes fraction', 'data'), textcoords='offset points')
ax2.set_xlim([min(rez['xax0'])-xtraAx2, END+xtraAx2])
#ax2.set_xticks([])
ax2.set_ylim(ylim2)
ax2.set_xlabel('Minibatches')
ax2.annotate('1st Epoch', xy=(12,epoch0Display), fontweight='bold')
#ax2.ylabel('Accuracy')
#ax2.title('Epoch 0 Test Accuracy\n' + resUsableTitle)
#ax2.legend()#loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)

plt.savefig(savePath+'combined'+datName+'_'+modelName+'.png', bbox_inches='tight')
plt.show()




#########################
# For computing nonzero std. devs.
#  # If there were actually multiple trials, compute standard deviations
#  #   in accuracies using NONZERO entries only.
#  if numTrials>1:
#    for epoch, tnsr in enumerate(allTr):
#      # This is my stupid way of doing ONLY NONZERO std dev's...
#      stdtr  = tnsr[ tnsr.gt(0) ].std().numpy()
#      if np.isnan(stdtr):
#        stdtr=np.array([0])
#      # Finally...
#      stdsTr[epoch] = stdtr
#      ##############################
#      # Now do Testing.
#      # This is my stupid way of doing ONLY NONZERO std dev's...
#      stdte  = allTe[epoch][ allTe[epoch].gt(0.1) ].std().numpy()
#      if np.isnan(stdte):
#        stdte=np.array([0])
#      # Finally...
#      stdsTe[epoch] = stdte
#    ##############################
#    # Finally, do Epoch-0 results..
#    # This is my stupid way of doing ONLY NONZERO std dev's...
#    for minibatch, tnsr in enumerate(allR0):
#      stdE0  = allTe[minibatch][ allTe[minibatch].gt(0.1) ].std().numpy()
#      if np.isnan(stdE0):
#        stdE0=np.array([0])
#      # Finally...
#      stdE0[epoch] = stdte



