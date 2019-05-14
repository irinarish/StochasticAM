"""@author Benjamin Cowen and Mattia Rigotti
@date 23 Jan 2019
"""
from __future__ import print_function, division
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import random as rand # needed for validation shuffle
from altmin14jan import get_mods, get_codes
from altmin14jan import update_codes, update_last_layer_, update_hidden_weights_adam_, set_mod_optimizers_
from altmin14jan import post_processing_step

from utils import ddict, load_dataset

##########################################################################################
##########################################################################################

parser = argparse.ArgumentParser(description='AltMin on mnist')
#.............................................................
# New experimental option is to load best hyperparameters from saved thing.
# Will use the Bunch class to make a "namespace" from this loaded dict, if requested.
class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)
parser.add_argument('--loadBestHyperparameters', type=str, default=None,
                    help='Path to file containing ddict() of desired arguments.')
#.............................................................

# Training settings
parser.add_argument('--delta-batch-size', type=int, default=0, metavar='N',
                    help='batch size increas after each epoch')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--postprocessing-steps', type=int, default=0, metavar='N',
                    help='number of Carreira-Perpinan post-processing steps after training')
parser.add_argument('--dataset', default='mnist', metavar='D',
                    help='name of dataset')
parser.add_argument('--use-validation-size', type=int, default=-1, metavar='N',
                    help='If <=0, uses test data. Otherwise, uses validation set of size N from training set to "test".')
parser.add_argument('--data-augmentation', action='store_true', default=False,
                    help='enables data augmentation')
parser.add_argument('--model', default='LeNet', metavar='M',
                    help='name of model')
parser.add_argument('--log-frequency', type=int, default=4, metavar='N',
                    help='how many times per epoch to log training status')
parser.add_argument('--save-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before saving test performance (only relevant if one epoch is larger than that)')
parser.add_argument('--first-epoch-log', action='store_true', default=False,
                    help='whether or not it should test and log after every mini-batch in first epoch')
# WE ARE NOT ALLOWING INDEPENDENT PARAMETER HERE:
#parser.add_argument('--lr-out', type=float, default=0.001, metavar='LR',
#                    help='learning rate for last layer weights updates')

# Logistics:
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save-filename', default='', metavar='F',
                    help='name of file where results are saved')
parser.add_argument('--opt-method', type=str, default='altmin',
                    help='"altmin" or "adam" or "sgd"')

# Things we loop over:
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training')
# CODE LEARNING PARAMS
parser.add_argument('--lr-codes', type=float, default=0.5, metavar='LR',
                    help='learning rate for codes updates')
parser.add_argument('--n-iter-codes', type=int, default=1, metavar='N',
                    help='number of internal iterations for codes optimization')
# Sparsity:
parser.add_argument('--lambda_c', type=float, default=0.0, metavar='L',
                    help='codes sparsity')
# WEIGHT LEARNING PARAMS
parser.add_argument('--n-iter-weights', type=int, default=1, metavar='N',
                    help='number of internal iterations in learning weights')
parser.add_argument('--lr-weights', type=float, default=0.002, metavar='LR',
                    help='learning rate for hidden weights updates')
# Sparsity:
parser.add_argument('--lambda_w', type=float, default=0.0, metavar='L',
                    help='weight sparsity')

# Newly added:
######## MU:
parser.add_argument('--min-mu', type=float, default=0.01, metavar='M',
                    help='starting point for mu')
parser.add_argument('--d-mu', type=float, default=0.001/600, metavar='M',
                    help='increase in mu after every mini-batch')
parser.add_argument('--mult-mu', type=float, default=1, metavar='M',
                    help='multiplies into mu after every mini-batch')
parser.add_argument('--mu-update-freq', type=int, default=1, metavar='M',
                    help='Number of minibatches between \mu updates (or just at end of epoch if larger than number of minibatches/epoch).')
parser.add_argument('--max-mu', type=float, default=10000, metavar='M',
                    help='cap for mu (can\'t go higher)')

######### MU-DEPENDENT SPARSITY:
parser.add_argument('--lambda_c_muFact', type=float, default=0.0, metavar='L',
                    help='Takes precedent over regular lambda_c. Sets lambda_c = L*mu.')
parser.add_argument('--lambda_w_muFact', type=float, default=0.0, metavar='L',
                    help='Takes precedent over regular lambda_w. Sets lambda_w = L*mu.')
####### MINI-EPOCHS
parser.add_argument('--mini-epochs', type=int, default=1, metavar='N',
                    help='Alternate between code and weight subproblem N times per minibatch')
######### LAYER-WISE DROPOUT!!

parser.add_argument('--LW-dropout-perc', type=float, default=0,
                    help='Percent chance of dropping out.')
parser.add_argument('--LW-dropout-delay', type=int, default=1,
                    help='Number of epochs to delay application of layer-wise dropout.')
##########################################################################################
##########################################################################################

# Continuing...
args_og = parser.parse_args()
if args_og.loadBestHyperparameters is None:
  args = args_og
else:
  # OR~~ load args from previously saved ddict(). SOME ARGS (below) 
  # WILL BE OVERWRITTEN (ie to load "best hyperparameters" then 
  # run diff number of epochs etc)
  args = Bunch(ddict()._load(args_og.loadBestHyperparameters).args)
  # Certain args will be changed from original:
  # all the "Logistics" arguments
  args.save_filename   = args_og.save_filename
  args.seed            = args_og.seed
  args.no_cuda         = args_og.no_cuda
  args.log_frequency   = args_og.log_frequency
  args.save_interval   = args_og.save_interval
  args.first_epoch_log = args_og.first_epoch_log

  # epochs
  args.epochs          = args_og.epochs

  # dataset (noisy test set)
  args.use_validation_size = args_og.use_validation_size

  # TODO Maybe change these as well at some point?...
  # L1 ?
  # n-iter-codes?






algName=args.opt_method
args.algorithm =algName + '_CNN'
# Train all layers with the same learning rate for controllability!
args.lr_out = args.lr_weights

# Check cuda
device = torch.device("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
if device.type != 'cpu':
    print('\033[93m'+'Using CUDA'+'\033[0m')
    torch.cuda.manual_seed(args.seed)

# Do not use multi-gpu!!
multi_gpu, num_workers = False, 1
if False:
  if device.type != 'cpu' and torch.cuda.device_count() > 1:
      print('\033[93m'+'Using ', torch.cuda.device_count(), 'GPUs'+'\033[0m')
      multi_gpu = True
      num_workers = torch.cuda.device_count()

# Load dataset
print('\n* Loading dataset {}'.format(args.dataset))
if   args.use_validation_size>0:
  dataName  = 'valid_'
  data_seed=23
  rand.seed(data_seed)
else:
  dataName  = ''
if args.data_augmentation:
    print('    data augmentation')
train_loader, test_loader, n_inputs = load_dataset(dataName+args.dataset, batch_size=args.batch_size, conv_net=True, data_augmentation=args.data_augmentation, num_workers=num_workers, valid_size=args.use_validation_size)

window_size = train_loader.dataset.train_data[0].shape[0]
if len(train_loader.dataset.train_data[0].shape) == 3:
    num_input_channels = train_loader.dataset.train_data[0].shape[2]
else:
    num_input_channels = 1

if hasattr(train_loader, 'numSamples'):
  numTrData = train_loader.numSamples
  numTeData = test_loader.numSamples
else:
  numTrData = len(train_loader.dataset)
  numTeData = len(test_loader.dataset)


# Load model
print('* Loading model {}'.format(args.model))
if args.model.lower() == 'lenet':
    from models import test, LeNet
    model = LeNet(num_input_channels=num_input_channels, window_size=window_size, bias=True).to(device)
elif args.model.lower() == 'vgg7':
    from models import test, VGG7
    model = VGG7(num_input_channels=num_input_channels, window_size=window_size, bias=True).to(device)
elif args.model.lower() in ['resnet', 'resnet18']:
    from models import test, resnet18
    model = resnet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()


# Main
if __name__ == "__main__":
    # Save everything in SHELF
    print('********************')
    print('Saving shelf to:')
    print(args.save_filename)
    print('********************')
    SH = ddict(args=args.__dict__)
    if args.save_filename:
        SH._save(args.save_filename, date=True)

    # Store training and test performance after each training epoch
    SH.perf = ddict(tr=[], te=[])

    # Store test performance after each iteration in first epoch
    SH.perf.first_epoch = []

    # Store test performance after each args.save_interval iterations
    SH.perf.te_vs_iterations = []

    # Create model, and expose modules that has_codes
    if algName=='altmin':
      model = get_mods(model)
      set_mod_optimizers_(model, optimizer = 'Adam',
                     optimizer_params = {'lr':args.lr_weights}, data_parallel=multi_gpu)
      model[-1].optimizer.param_groups[0]['lr'] = args.lr_weights

    elif algName == 'adam':
      optimizer = optim.Adam(model.parameters(), lr=args.lr_weights)
    elif algName == 'sgd':
      optimizer = optim.SGD(model.parameters(), lr=args.lr_weights)
      scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # Initial mu and increment after every mini-batch
    mu = args.min_mu
    mu_max = args.max_mu

    for epoch in range(1, args.epochs+1):

        # Configure dropout parameters.
        if args.LW_dropout_perc>0 and args.LW_dropout_delay<epoch:
          useDropout = args.LW_dropout_perc
        else:
          useDropout = 0

        batch_size = args.batch_size + args.delta_batch_size*(epoch-1)
        print('\nEpoch {} of {}. mu = {:.2f}, batch_size = {}, algorithm = {}'.format(epoch, args.epochs, mu, batch_size, algName))

        model.train()
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            if algName=='altmin':
            #----------------------------------------------------------
              # Set L1 weights according to \mu.
              if args.lambda_c_muFact>0:
                epoch_lam_c = args.lambda_c_muFact * mu
              else:
                epoch_lam_c = args.lambda_c
              if args.lambda_w_muFact>0:
                epoch_lam_w = args.lambda_w_muFact * mu
              else:
                epoch_lam_w = args.lambda_w
  
              # Now run alt-min algorithm.
              for mini_epoch in range(args.mini_epochs):
                ######## Forward: compute codes.
                model.train()
                with torch.no_grad():
                    outputs, codes = get_codes(model, data)
  
                # Update codes w/weights fixed.
                codes = update_codes(codes, model, targets, criterion, mu, lambda_c=epoch_lam_c, n_iter=args.n_iter_codes, lr=args.lr_codes)
  
                # Update weights w/codes fixed.
                # Manually apply dropout for last layer:
                if (torch.rand(1).item() > useDropout) or useDropout<=0.00001:
                  update_last_layer_(model[-1], codes[-1], targets, criterion, n_iter=args.n_iter_weights)
  
                update_hidden_weights_adam_(model, data, codes, lambda_w=epoch_lam_w, n_iter=args.n_iter_weights, dropout_perc=useDropout)
  
                # Increment mu. (batch_idx init to 0)
                if ((batch_idx+1) % args.mu_update_freq == 0) and (mu < mu_max):
                    mu += args.d_mu
                    mu *= args.mult_mu
            #----------------------------------------------------------
            elif algName in ['sgd','adam']:
              optimizer.zero_grad()
              outputs = model(data)
              loss = criterion(outputs, targets)
              loss.backward()
              optimizer.step()
            #----------------------------------------------------------

            # Store all iterations of first epoch
            if epoch == 1 and args.first_epoch_log:
                SH.perf.first_epoch += [test(model, data_loader=test_loader, label=" - Test")]

            # Outputs to terminal
            if batch_idx % int(len(train_loader)/args.log_frequency) == 0:
                loss = criterion(outputs, targets)
                print(' Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), numTrData,
                    100. * batch_idx / len(train_loader), loss.item()))

            # After every args.save_interval iterations compute and save test error
            if batch_idx % args.save_interval == 0 and batch_idx > 0:
                SH.perf.te_vs_iterations += [test(model, data_loader=test_loader, label=" - Test")]
                if args.save_filename:
                    SH._save()

        # If the mu-increment-frequency is larger than the number of batches,
        # here is a separate update for epoch-wise mu updating.
        if (args.mu_update_freq > len(train_loader)) and (mu < mu_max):
            mu += args.d_mu
            mu *= args.mult_mu

        # Print performances
        SH.perf.tr += [test(model, data_loader=train_loader, label="Training")]
        SH.perf.te += [test(model, data_loader=test_loader, label="Test")]

        # Save data after every epoch
        if args.save_filename:
            SH._save()

    # ----------------------------------------------------------------
    # Post-processing step from Carreira-Perpinan (fit last layer):
    # ----------------------------------------------------------------
    if args.postprocessing_steps > 0:

        print('\nPost-processing step:\n')

        for epoch in range(1, args.postprocessing_steps+1):
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(device), targets.to(device)
                post_processing_step(model, data, targets, criterion, args.lambda_w)

                # Outputs to terminal
                if batch_idx % args.log_interval == 0:
                    print(' Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

        # Print performances
        SH.perf.tr_final = test(model, data_loader=train_loader, label="  Training set after post-processing")
        SH.perf.te_final = test(model, data_loader=test_loader, label="  Test set after post-processing    ")

        # Save data after every epoch
        if args.save_filename:
            SH._save()

    if args.save_model:
      SH.perf.model = model
