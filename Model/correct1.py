# -*- coding: utf-8 -*-
# The code is written for EnSRF
# by Seoleun Shin (KRISS)
#
# This version is saved at 15-APRIL-2025
# Publically added at https://github.com/seoleuns/MLforLorenz96

import torch
import numpy as np
from numpy.random import randn
from observation import observe

# Background error covariance inflation factor
rho = 1.0 # Blowup for physical model


def assimilate1(ensemble, obs, obs_covar):
   
    n_ens = len(ensemble[0][0])
    n_time = len(ensemble[0][0][0])
    nx_ens = len(ensemble[0][0][0][0])

    def observeop(state, row=None):

        if row is None:
            return np.copy(state[...,:nx_ens])
        else:
            return np.copy(state[...,row])


    flat_ensemble = []
    for sublist in ensemble:
        for item in sublist:
            for item2 in item:
                for item3 in item2:
                    for item4 in item3:
                        flat_ensemble.append(item4)
    numpylist = []
    for _tensor in flat_ensemble:
        if isinstance(_tensor, torch.Tensor):
           numpylist.append(_tensor.detach().cpu().numpy())
        else:
        # Handle the case when _tensor is not a PyTorch tensor
           pass
    flat_ensemble = [_tensor for _tensor in flat_ensemble if isinstance(_tensor, torch.Tensor)]

    def flatten_to_nested2(input_list, shape):

        size = shape[0]
        result = torch.tensor(input_list).view(shape).tolist()
        if size == 1:
    #    # If the first dimension has size 1, append the first element directly
            result = [result]
        return result


    def flatten_to_nested(input_list, shape):
        if not shape:
            return input_list

        result = []
        size = shape[0]

        for _ in range(size):
            if len(shape) == 1:
               result.append(input_list.pop(0))
            else:
               sub_shape = shape[1:]
               sublist = flatten_to_nested(input_list, sub_shape)
               result.append(sublist)

        return result
    desired_shape=[1,n_ens,n_time,nx_ens]

    #flat_ensemble.clear()
    newens = np.array(numpylist, dtype=np.float32)
    newens = newens.reshape(1,n_ens,n_time,nx_ens)
    newensf = newens[0,:,1,:]
    ensemble = newensf.reshape(n_ens,nx_ens)

    newobs = obs.reshape(1,n_ens,n_time,nx_ens)
    newobsf = newobs[0,0,1,:]
    obsarray = newobsf.reshape(nx_ens)
    obslist = obsarray.tolist()


    # Form ensemble mean
    ens_mean = np.mean(ensemble, axis=0)

    # Form the background ensemble perturbation matrix
    X_f = rho*(ensemble - ens_mean)
    X_f_trans = X_f.T

    # Sequentially process observations
    for i, ob in enumerate(obslist):
        # Ensemble covariance times transpose of observation matrix
        P_f_H_T = np.matmul(X_f_trans, observeop(X_f, i))/(n_ens - 1)

        HP_f_H_T = observeop(P_f_H_T, i)

        P_f_H_T_comp = P_f_H_T
        HP_f_H_T_comp = HP_f_H_T

        # Kalman gain
        first = P_f_H_T_comp
        second = HP_f_H_T_comp + obs_covar[i,i].detach().cpu().numpy()
        gain = first/second
        ens_mean += gain*(ob - observeop(ens_mean, i))

        np.random.seed(i+50000)

        # Update perturbations
        alphap = 1.0/(1.0+np.sqrt(obs_covar[i,i].detach().cpu().numpy()/(HP_f_H_T_comp + obs_covar[i,i].detach().cpu().numpy())))
        for j in range(n_ens):
            X_f[j,:] += -alphap*gain*(observeop(X_f[j,:],i))
        
    finalv = ens_mean + X_f
    # Form final ensemble
    modifinalv = np.repeat(finalv[np.newaxis,:,:],1,axis=0)
    modifinalv = np.repeat(modifinalv[:,:,np.newaxis,:],2,axis=2)


    return modifinalv 
