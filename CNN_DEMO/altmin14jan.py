"""
@author Mattia Rigotti (augmented by Benjamin Cowen)
@date 15 Jan 2019
"""
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils.nn import Flatten


def compute_codes_loss(code, nmod, lin, loss_fn, codes_target, mu, lambda_c):
    ''' codes: outputs of the linear modules
        nmod: non-linear module downstream from linear module
        lin:  linear module (Conv2d or Linear)
    '''
    output = lin(nmod(code))
    loss = (1/mu)*loss_fn(output) + F.mse_loss(code, codes_target)
    if lambda_c>0.0:
        loss += (lambda_c/mu)*code.abs().mean()
    return loss


def update_memory(As, Bs, inputs, codes, model_mods, eta=0.0):
    "Updates the bookkeeping matrices using codes as in Mairal2009"
    if hasattr(model_mods, 'n_inputs'):
        x = inputs.view(-1, model_mods.n_inputs)
    else:
        x = inputs

    with torch.no_grad():
        id_codes = [i for i,m in enumerate(model_mods) if hasattr(m, 'has_codes') and getattr(m, 'has_codes')]
        for i, (idx, c_in, c_out) in enumerate(zip(id_codes, [x]+codes[:-1], codes)):
            try:
                nmod = model_mods[idx-1]
            except IndexError:
                nmod = lambda x: x

            a = nmod(c_in)
            if eta == 0.0:
                As[i] += a.t().mm(a)
                Bs[i] += c_out.t().mm(a)
            else:
                As[i] = (1-eta)*As[i] + eta*a.t().mm(a)
                Bs[i] = (1-eta)*Bs[i] + eta*c_out.t().mm(a)
    return As, Bs


def update_hidden_weights_bcd_(model_mods, As, Bs, lambda_w):
    # Use BCD to update intermediate weights
    id_codes = [i for i,m in enumerate(model_mods) if hasattr(m, 'has_codes') and getattr(m, 'has_codes')]
    for i, A, B in zip(id_codes, As, Bs):
        model_mods[i].weight.data = BCD(model_mods[i].weight.data, A, B, lambda_w)


def BCD(w, A, B, lambda_w, eps=1e-3, max_iter=20, return_errors=False):
    # lambda_w is referenced to A_jj in every column
    B = B.div(A.diag() + 1e-10)
    A = A.div(A.diag() + 1e-10)

    errors = []
    with torch.no_grad():
        for i in range(max_iter):
            w_pre = w.clone()
            error = 0
            for j in range(A.shape[1]):
                delta_j = (B[:,j] - w.mv(A[:,j]))
                w[:,j].add_(delta_j)
                #  u_j /= max(u_j.norm(), 1.0) # This was in Mairal2009, but assumes that B has spectral radius smaller than A
                # Shrinkage step (sparsity regularizer)
                if lambda_w > 0.0:
                    sign_w = w[:,j].sign()
                    w[:,j].abs_().add_(-lambda_w).clamp_(min=0.0).mul_(sign_w)
                error += delta_j.abs().mean().item()
            errors.append(error)
            # Converged is there is no change between steps
            if (w - w_pre).abs().max().item() < eps:
                break

    if return_errors:
        return w, errors
    else:
        return w


def post_processing_step(model, data, target, criterion, lambda_w, n_iter=1):
        with torch.no_grad():
            output, codes = get_codes(model, data)

        update_last_layer_(model, codes[-1], target, criterion, lambda_w=args.lambda_w, n_iter=n_iter)


def insert_mod(model_mods, mod, has_codes):
    "If a mod is not empty, close it, include it, and start a new mod"
    if len(mod) == 1:
        model_mods.add_module(str(len(model_mods)), mod[0])
        model_mods[-1].has_codes = has_codes
    elif len(mod) > 1:
        model_mods.add_module(str(len(model_mods)), mod)
        model_mods[-1].has_codes = has_codes
    mod = nn.Sequential()
    return mod

# EXTRACTED FROM get_mods 2 Jan 2019:
def set_mod_optimizers_(model_mods, optimizer=None, optimizer_params={}, data_parallel=False):
    '''
      Sets/resets model optimizer(s).
    '''
    if optimizer is not None:
        # Include an optimizer in modules with codes
        for m in model_mods:
            if m.has_codes:
                m.optimizer = getattr(optim, optimizer)(m.parameters(), **optimizer_params)

        # Add optimizer to the last layer
        model_mods[-1].optimizer = getattr(optim, optimizer)(model_mods[-1].parameters(), **optimizer_params)

    if data_parallel:
        data_parallel_mods_(model_mods)

    return model_mods

def get_mods(model, optimizer=None, optimizer_params={}, data_parallel=False):
    '''Returns all the modules in a nn.Sequential alternating linear and non-linear modules
        Arguments:
            optimizer: if not None, each module will be given an optimizer of the indicated type
            with parameters in the dictionary optimizer_params
      EDIT BY BEN: 2 Jan 2019
      --- if model is already a nn.Sequential, just resets the optimizers.
    '''
    if not isinstance(model, nn.Sequential):
      model_mods = nn.Sequential()
      if hasattr(model, 'n_inputs'):
          model_mods.n_inputs = model.n_inputs

      nmod, lmod = nn.Sequential(), nn.Sequential()
      for m in list(model.features) + [Flatten()] + list(model.classifier):
          if any([isinstance(m, t) for t in [nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d]]) or hasattr(m, 'has_codes'):
              nmod = insert_mod(model_mods, nmod, has_codes=False)
              lmod.add_module(str(len(lmod)), m)
          else:
              lmod = insert_mod(model_mods, lmod, has_codes=True)
              nmod.add_module(str(len(nmod)), m)

      insert_mod(model_mods, nmod, has_codes=False)
      insert_mod(model_mods, lmod, has_codes=True)

      # Last layer that generates codes is lumped together with adjacent modules to produce the last layer
      id_codes = [i for i,m in enumerate(model_mods) if hasattr(m, 'has_codes') and getattr(m, 'has_codes')]

      model_tmp = model_mods[:id_codes[-2]+1]
      model_tmp.add_module(str(len(model_tmp)), model_mods[id_codes[-2]+1:])
      model_tmp[-1].has_codes = False
      model_mods = model_tmp

      # Added by Ben.
      if optimizer is not None:
        set_mod_optimizers_(model, optimizer = optimizer,
                         optimizer_params = optimizer_params, data_parallel=data_parallel)
      return model_mods

def data_parallel_mods_(model_mods):
    for i,m in enumerate(model_mods):
        model_mods[i] = torch.nn.DataParallel(m)
        model_mods[i].has_codes = m.has_codes
        if hasattr(m, 'optimizer'):
            model_mods[i].optimizer = m.optimizer


# EDIT BY BEN 12/12/2018:
# all RNN stuff...
def get_codes(model_mods, inputs):
    '''Runs the architecture forward using `inputs` as inputs, and returns outputs and intermediate codes
    '''
    # First mess with the input shape like this.
    # Probably have to change for the RNN.
    if hasattr(model_mods, 'n_inputs'):
        x = inputs.view(-1, model_mods.n_inputs)
    else:
        x = inputs

    # If doing RNN, codes have to be computed a certain way.
    if hasattr(model_mods, 'isRNN') and getattr(model_mods, 'isRNN'):
      codes = model_mods.forward(x, get_codes=True)

    else:
      # As codes only return outputs of linear layers
      codes = []
      for m in model_mods:
          x = m(x)
          if hasattr(m, 'has_codes') and getattr(m, 'has_codes'):
              codes.append(x.clone())
    # Do not include output of very last linear layer (not counted among codes)
    return x, codes


def update_codes(codes, model_mods, targets, criterion, mu, lambda_c, n_iter, lr):
    id_codes = [i for i,m in enumerate(model_mods) if hasattr(m, 'has_codes') and getattr(m, 'has_codes')]
    for l in range(1, len(codes)+1):
        idx = id_codes[-l]

        codes[-l].requires_grad_(True)
        optimizer = optim.SGD([codes[-l]], lr=lr)
        codes_initial = codes[-l].clone()

        try:
            nmod = model_mods[idx+1]
        except IndexError:
            nmod = lambda x: x
        try:
            lin = model_mods[idx+2]
        except IndexError:
            lin = lambda x: x

        if l == 1:  # last layer
            loss_fn = lambda x: criterion(x, targets)
        else:       # intermediate layers
            loss_fn = lambda x: F.mse_loss(x, codes[-l+1].detach())

        # EDIT BY BEN 12/6:
        # This optionally exits if converged.
#        chgTol = 1e-4
#        chg    = chgTol+1
        # first iteration is out here.
        optimizer.zero_grad()
        loss = compute_codes_loss(codes[-l], nmod, lin, loss_fn, codes_initial, mu, lambda_c)
        loss.backward()
        optimizer.step() 
        it     = 1
        while ( (it < n_iter)):# and (chg>chgTol)):
            it +=1
            # For tracking convergence:
            prevLoss = loss.item()
            # Gradient Descent Step:
            optimizer.zero_grad()
            loss = compute_codes_loss(codes[-l], nmod, lin, loss_fn, codes_initial, mu, lambda_c)
            loss.backward()
            optimizer.step()
            # Check convergence:
#            chg = (np.abs(loss.item() - prevLoss)/np.abs(prevLoss))
    return codes


def update_last_layer_(mod_out, inputs, targets, criterion, n_iter):
    for it in range(n_iter):
        mod_out.optimizer.zero_grad()
        outputs = mod_out(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        mod_out.optimizer.step()


def update_hidden_weights_adam_(model_mods, inputs, codes, lambda_w, n_iter, dropout_perc=0):
    id_codes = [i for i,m in enumerate(model_mods) if hasattr(m, 'has_codes') and getattr(m, 'has_codes')]

    if hasattr(model_mods, 'n_inputs'):
        x = inputs.view(-1, model_mods.n_inputs)
    else:
        x = inputs

    for idx, c_in, c_out in zip(id_codes, [x]+codes[:-1], codes):
        # Randomly skip this layer according to dropout_perc:
        if (torch.rand(1).item() <= dropout_perc) and dropout_perc>0.00001:
          continue

        lin = model_mods[idx]
        if idx >= 1:
            nmod = model_mods[idx-1]
        else:
            nmod = lambda x: x

        # EDIT BY BEN 12/6:
        # This optionally exits if converged.
#        chgTol = 1e-4
#        chg    = chgTol+1
        # first iteration is out here.
        lin.optimizer.zero_grad()
        loss = F.mse_loss(lin(nmod(c_in)), c_out.detach())
        if lambda_w > 0.0:
            loss += lambda_w*lin.weight.abs().mean()
        loss.backward()
        lin.optimizer.step()
        it     = 1
        while ( (it < n_iter)):# and (chg>chgTol)):
            it +=1
            # For tracking convergence:
            prevLoss = loss.item()
            # Gradient Descent Step:
            lin.optimizer.zero_grad()
            loss = F.mse_loss(lin(nmod(c_in)), c_out.detach())
            if lambda_w > 0.0:
                loss += lambda_w*lin.weight.abs().mean()
            loss.backward()
            lin.optimizer.step()
            # Check convergence:
#            chg = (np.abs(loss.item() - prevLoss)/np.abs(prevLoss))



# ------------------------------------------------------------------------
# Misc
# ------------------------------------------------------------------------
def update_codes_TANH(model, codes, targets, mu, n_iter, lr):
    '''Update codes approximating sign non-linearity with the soft version tanh
    '''
    #  Last layer codes
    codes[-1].requires_grad_(True)
    codes_target = codes[-1].clone()
    optimizer = optim.SGD([codes[-1]], lr=lr)
    for it in range(n_iter):
        optimizer.zero_grad()
        loss = compute_codes_loss(codes[-1], F.nll_loss, targets, torch.tanh, model.w_out, codes_target, mu, 0)
        loss.backward()
        optimizer.step()

    # Hidden layers codes
    for l in range(2, len(codes)):
        codes[-l].requires_grad_(True)
        codes_target = codes[-l].clone()
        optimizer = optim.SGD([codes[-l]], lr=lr)
        for it in range(n_iter):
            optimizer.zero_grad()
            loss = compute_codes_loss(codes[-l], F.mse_loss, codes[-l+1].data, torch.tanh, model.w[-l+1], codes_target, mu, 0)
            loss.backward()
            optimizer.step()

    return codes


class OnlineCov(nn.Module):
    '''Online covariance class
        Online calcularion of covariance matrix between a stream x and y, updated as in the Online paragraph of https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Covariance"
    '''
    def __init__(self, in_features, out_features=None):
        super(OnlineCov, self).__init__()
        if out_features is None:
            out_features = in_features
        # Covariance
        self.register_buffer('cc', torch.zeros(out_features, in_features))
        # Means
        self.register_buffer('mX', torch.zeros(in_features))
        self.register_buffer('mY', torch.zeros(out_features))
        # Iteration number
        self.n_iter = 0

        self.in_features = in_features
        self.out_features = out_features

    def reset(self):
        self.cc.zero_()
        self.mX.zero_()
        self.mY.zero_()
        self.n_iter = 0

    def update_cov(self, x, y):
        # Checks
        if x.dim() < 2:
            x = x.view(1,-1)
        if y.dim() < 2:
            y = y.view(1,-1)
        assert x.size(0) == y.size(0)

        # Start update
        self.n_iter += 1

        self.mX  += (x - self.mX).mean(0)/self.n_iter
        dx = x - self.mX
        dy = y - self.mY # Note: it's correct that dy is computed with previous mean

        self.cc += dy.t().mm(dx)

        self.mY += (y - self.mY).mean(0)/self.n_iter

    def get_cc(self):
        return self.cc/self.n_iter

    def __len__(self):
        return self.n_iter

    def extra_repr(self):
        return '{in_features}, {out_features}, n_iter={n_iter}'.format(**self.__dict__)


def update_memory_cov(As, Bs, codes, nlin=F.relu):
    '''Updates the bookkeeping matrices using codes as in Mairal2009, but using centered variance and covariance
    Arguments:
        As, Bs: lists of OnlineCov objects
    '''
    a = codes[0].data
    for i in range(len(codes)-1):
        As[i].update_cov(a, a)
        Bs[i].update_cov(a, codes[i+1].data)
        a = nlin(codes[i+1].data)
    return As, Bs


def update_hidden_weights_cov_(Ws, As, Bs, lambda_w):
    # Use BCD to update intermediate weights
    for i, (A, B) in enumerate(zip(As, Bs)):
        Ws[i].weight.data = BCD(Ws[i].weight.data, A.get_cc(), B.get_cc(), lambda_w)
        Ws[i].bias.data = B.mY - Ws[i].weight.data.mv(B.mX)

