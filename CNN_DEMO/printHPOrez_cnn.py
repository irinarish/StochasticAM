"""
Loads saved .pt's and prints best hyperparameters.
@author Benjamin Cowen
@date 19 January 2019
"""
from utils import ddict
import numpy as np
import glob
import torch
import argparse
parser = argparse.ArgumentParser(description='Load and print hyperparameters.')
parser.add_argument('--loadPath', default='./',
                    help='Path to directory w/data.')

args = parser.parse_args()

argList2Print={'all':['opt_method',
                      'batch_size',
                      'dataset',
                      'model',
                      'lr_weights'
                      ],
            'altmin':[
                     # Algorithm
                     'mini_epochs',
                     'n_iter_codes',
                     'n_iter_weights',
                     'lr_codes',
                     # Model Parameters
                     'lambda_w_muFact',
                     'lambda_c_muFact',
                     'min_mu',
                     'max_mu',
                     'd_mu',
                     'mu_update_freq',
                     'mult_mu'
                    ],
              'sgd':['lambda_w'],
             'adam':['lambda_w']
            }
for filepath in glob.iglob(args.loadPath + '*.pt'):
  D = ddict()._load(filepath).args

  print('============= {} ================='.format(D['algorithm']))
  for argName in argList2Print['all']:
    print('{} = {}'.format(argName, D[argName]))
  for argName in argList2Print[D['opt_method']]:
    print('{} = {}'.format(argName, D[argName]))

