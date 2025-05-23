# -*- coding: utf-8 -*-
# The initial version of this code is obtained from https://github.com/wenhaomin/DiffSTG
#
# The code is customized to apply for the Lorenz '96 system
# by Seoleun Shin (KRISS) 
# 
# This version is saved at 15-APRIL-2025
# Publically added at https://github.com/seoleuns/MLforLorenz96

import os, sys
import torch
import argparse
import numpy as np
import torch.utils.data
from easydict import EasyDict as edict
from timeit import default_timer as timer
import random

from utils.eval import Metric
from utils.gpu_dispatch import GPU
from utils.pdfutilsgrid40 import draw_predicted_distribution, dir_check, to_device, ws, unfold_dict, dict_merge, GpuId2CudaId, Logger

from algorithm.dataset import FullDataset, CleanDataset, TrafficDataset
from algorithm.diffstg.model import DiffSTG, save2file
#RTPS
#from rtps import assimilate1
from correct1 import assimilate1
from math import sqrt

#from numodels import step_one_layer as fcst_step
from numodels import step_two_layer as fcst_step
from params import params as nuparams

def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


# for tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard'))
except:
    pass

def get_params():
    parser = argparse.ArgumentParser(description='Entry point of the code')

    # model
    parser.add_argument("--epsilon_theta", type=str, default='UGnet')
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--beta_schedule", type=str, default='quad')  # uniform, quad
    parser.add_argument("--beta_start", type=float, default=0.97)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--sample_steps", type=int, default=200)  # sample_steps
    parser.add_argument("--ss", type=str, default='ddpm') #help='sample strategy', ddpm, multi_diffusion, one_diffusion
    parser.add_argument("--T_h", type=int, default=1)
    parser.add_argument("--T_p", type=int, default=1)

    # eval
    parser.add_argument('--n_samples', type=int, default=20)

    # train
    parser.add_argument("--is_train", type=bool, default=False) # train or evaluate
    parser.add_argument("--data", type=str, default='LoM')
    parser.add_argument("--mask_ratio", type=float, default=0.0) # mask of history data
    parser.add_argument("--is_test", type=bool, default=True)
    parser.add_argument("--is_da", type=bool, default=True)
    parser.add_argument("--nni", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch_size", type=int, default=128)

    args, _ = parser.parse_known_args()
    return args

def default_config(data='AIR_BJ'):
    config = edict()
    config.PATH_MOD = ws + '/output/model/'
    config.PATH_LOG = ws + '/output/log/'
    config.PATH_FORECAST = ws + '/output/forecast/'

    # Data Config
    config.data = edict()
    config.data.name = data
    config.data.path = ws + '/data/dataset/'
    config.data.feature_file = config.data.path + config.data.name + '/40everylongmydat.npy'
    config.data.spatial = config.data.path + config.data.name + '/40everylonggriddat.npy'

    config.data.num_recent = 1

    if config.data.name == 'LoM':
        config.data.num_features = 1
        config.data.num_vertices = 40
        config.data.num_vertices_nu = 40
        config.data.points_per_hour = 1
        config.data.val_start_idx = int(2010000 * 0.7)
        config.data.test_start_idx = int(2010000 * 0.995)


    gpu_id = GPU().get_usefuel_gpu(max_memory=6000, condidate_gpu_id=[0,1,2,3,4,6,7,8])
    config.gpu_id = gpu_id
    if gpu_id != None:
        cuda_id = GpuId2CudaId(gpu_id)
        torch.cuda.set_device(f"cuda:{cuda_id}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model config
    config.model = edict()

    config.assim_freq = 6
    config.model.T_p = 1
    config.model.T_h = 1
    config.model.V = config.data.num_vertices
    config.model.F = config.data.num_features
    config.model.week_len = 7
    config.model.day_len = config.data.points_per_hour * 24
    config.model.device = device
    config.model.d_h = 32

    # config for diffusion model
    config.model.N = 200
    config.model.sample_steps = 200
    config.model.epsilon_theta = 'UGnet'
    config.model.is_label_condition = True
    config.model.beta_start = 0.97
    config.model.beta_end = 0.2
    config.model.beta_schedule = 'quad'
    config.model.sample_strategy = 'ddpm'

    # if 995  
    config.dataind = 10000
    config.dataindf = 0

    config.dataint = config.dataind - config.dataindf

    config.n_samples = 20
    config.n_ddim = 10

    # config for UGnet
    config.model.channel_multipliers = [1, 2]  # The list of channel numbers at each resolution.
    config.model.supports_len = 2

    # training config
    config.model_name = 'DiffSTG'
    config.is_test = True  # Whether run the code in the test mode
    config.is_da = True  # Whether run the code in the test mode
    config.epoch = 1  # Number of max training epoch
    config.optimizer = "adam"
    config.lr = 1e-4
    config.batch_size = 1
    config.obs_err_var = 0.1
    config.wd = 1e-5
    config.early_stop = 10
    config.start_epoch = 0
    config.device = device
    config.logger = Logger()

    # data assimilation config

    if not os.path.exists(config.PATH_MOD):
        os.makedirs(config.PATH_MOD)
    if not os.path.exists(config.PATH_LOG):
        os.makedirs(config.PATH_LOG)
    if not os.path.exists(config.PATH_FORECAST):
        os.makedirs(config.PATH_FORECAST)
    return config

def nu_config(data='AIR_BJ'):
    confignu = edict()
    confignu.PATH_MOD = ws + '/output/model/'
    confignu.PATH_LOG = ws + '/output/log/'
    confignu.PATH_FORECAST = ws + '/output/forecast/'

    # Data Config
    confignu.data = edict()
    confignu.data.name = data
    confignu.data.path = ws + '/data/dataset/'

    confignu.data.feature_file = confignu.data.path + confignu.data.name + '/40everylongmydat.npy'
    confignu.data.spatial = confignu.data.path + confignu.data.name + '/40everylonggriddat.npy'
    confignu.data.num_recent = 1

    if confignu.data.name == 'LoM':
        confignu.data.num_features = 1
        confignu.data.num_vertices = 40
        confignu.data.num_vertices_nu = 40
        confignu.data.points_per_hour = 1
        confignu.data.val_start_idx = int(2010000 * 0.95)
        #confignu.data.test_start_idx = int(2010000 * 0.993) #For Final_Best
        confignu.data.test_start_idx = int(2010000 * 0.995)

    gpu_id = GPU().get_usefuel_gpu(max_memory=6000, condidate_gpu_id=[0,1,2,3,4,6,7,8])
    confignu.gpu_id = gpu_id
    if gpu_id != None:
        cuda_id = GpuId2CudaId(gpu_id)
        torch.cuda.set_device(f"cuda:{cuda_id}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model config
    confignu.model = edict()

    confignu.assim_freq = 6
    confignu.model.T_p = 1
    confignu.model.T_h = 1
    confignu.model.V = confignu.data.num_vertices
    confignu.model.V_nu = confignu.data.num_vertices_nu
    confignu.model.F = confignu.data.num_features
    confignu.model.week_len = 7
    confignu.model.day_len = confignu.data.points_per_hour * 24
    confignu.model.device = device
    confignu.model.d_h = 32

    # config for diffusion model
    confignu.model.N = 200
    confignu.model.sample_steps = 200
    confignu.model.epsilon_theta = 'UGnet'
    confignu.model.is_label_condition = True
    confignu.model.beta_schedule = 'quad'
    confignu.model.sample_strategy = 'ddpm'

    confignu.n_samples = 20

    # config for UGnet
    confignu.model.channel_multipliers = [1, 2]  # The list of channel numbers at each resolution.
    confignu.model.supports_len = 2

    # training config
    confignu.model_name = 'DiffSTG'
    confignu.is_test = True  # Whether run the code in the test mode
    confignu.is_da = True  # Whether run the code in the test mode
    confignu.epoch = 1  # Number of max training epoch
    confignu.optimizer = "adam"
    confignu.lr = 1e-4
    confignu.batch_size = 32
    confignu.obs_err_var = 0.1
    confignu.wd = 1e-5
    confignu.early_stop = 10
    confignu.start_epoch = 0
    confignu.device = device
    confignu.logger = Logger()

    # data assimilation config

    if not os.path.exists(confignu.PATH_MOD):
        os.makedirs(confignu.PATH_MOD)
    if not os.path.exists(confignu.PATH_LOG):
        os.makedirs(confignu.PATH_LOG)
    if not os.path.exists(confignu.PATH_FORECAST):
        os.makedirs(confignu.PATH_FORECAST)
    return confignu


def cycles(model, ini_loader, data_loader, nu_loader, epoch, metric, config, confignu, clean_data, clean_data_nu, mode='Test'):

    setup_seed(2022)

    y_pred, y_true, time_lst = [], [], []
    metrics_future = Metric(T_p=config.model.T_p)
    metrics_history = Metric(T_p=config.model.T_h)
    model.eval()
    inix = torch.zeros(1, config.n_samples, 1, config.data.num_vertices,2) 

    nx = nuparams.nx
    ny = nuparams.ny
    print(nx,ny)


    # Initialise state vector array
    numinix = torch.zeros(1, config.n_samples, 1, nx+nx*ny, 2)
    numinixshort = torch.zeros(1, config.n_samples, 1, config.data.num_vertices, 2) 

    j = 0
    random.seed(2048)
    randomn = [random.randint(1, 500000) for _ in range(config.n_samples)]
    for i, batch in enumerate(ini_loader):
        future, history, pos_w, pos_d = to_device(batch, config.device) # target:(B,T,V,1), history:(B,T,V,1), pos_w: (B,1), pos_d:(B,T,1)
        x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device)  # (B, T, V, F)
        x_masked = x_masked.transpose(1, 3)  # (B, F, V, T)
        if i == 0:
           x0dat = x_masked 

        #if i%10001 == 0 and j < config.n_samples:
        if i in randomn and j < config.n_samples:
            print("WRITE=", i)
            inix[0, j, 0, :, :] = x_masked[0, 0, :, :]  
            numinix[0, j, 0, :nx, :] = x_masked[0, 0, :, :] 
            numinixshort[0, j, 0, :, :] = x_masked[0, 0, :, :] 
            j = j + 1


    samples, targets, ensmem = [], [], []
    nuensmem, nusamples = [], []

    i = 0
    for i, batch in enumerate(nu_loader):
        #print("CHECKSPEED=", i)
        j = 0

        future, history, pos_w, pos_d = to_device(batch, config.device) # target:(B,T,V,1), history:(B,T,V,1), pos_w: (B,1), pos_d:(B,T,1)

        x = torch.cat((history, future), dim=1).to(config.device)  # in cpu (B, T, V, F), T =  T_h + T_p
        x_masked = torch.cat((history, future), dim=1).to(config.device)  # (B, T, V, F)
        targets.append(x_masked.cpu().clone())
        x = x.transpose(1, 3)  # (B, F, V, T)
        x_masked = x_masked.transpose(1, 3)  # (B, F, V, T)

        n_samples = config.n_samples
        n_ddim = config.n_ddim
        timestepnum = i
        ## Make Ensemble Prediction and cycling !!
        if i==0:
           x_masked = x_masked.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)

           obsdata = x_masked 
           obsarray = obsdata.transpose(2,4).cpu().numpy()

           obs_covar = config.obs_err_var * torch.eye(40)
           obs_covar_ext0 = np.repeat(obs_covar[np.newaxis,:,:],1,axis=0)
           obs_covar_ext1 = np.repeat(obs_covar_ext0[:,np.newaxis,:,:],n_samples,axis=1)

           ini_x_masked = inix.transpose(2,4).cpu().numpy()
           ini_x_masked = torch.from_numpy(ini_x_masked)
           ensmem.append(ini_x_masked)

           numini_x_masked = numinixshort.transpose(2,4).cpu().numpy()
           numini_x_masked = torch.from_numpy(numini_x_masked)
           nuensmem.append(numini_x_masked)
           #exit()

           x_hat = model((inix.to(config.device), pos_w, pos_d), timestepnum, n_ddim, n_samples) # (B, n_samples, F, V, T)
           samples.append(x_hat.transpose(2,4).cpu().clone())
           analysis = assimilate1(samples, obsarray, obs_covar)
           extsamples = np.repeat(analysis[:,:,np.newaxis,:,:],1,axis=2)
           torchsamples = torch.from_numpy(extsamples)
           x_hat = torchsamples.permute(0,1,2,4,3)
           x_hat[:,:,:,:,0] = x_hat[:,:,:,:,1]
           x_hat[:,:,:,:,1] = 0. 
           updatehat = x_hat
           samples = extsamples.tolist()
           samples = [_tensor for _tensor in samples if isinstance(_tensor, torch.Tensor)]


           x_ini = numinix.cpu().numpy()

           for j in range(n_samples):
               x_ini[0, j, 0, :, 1] = fcst_step(x_ini[0, j, 0, :, 0])

           nux_hat_to_assimilate = torch.from_numpy(x_ini)
           numshort = nux_hat_to_assimilate[:, :, :, :nx, :] 
           nusamples.append(numshort.transpose(2,4).cpu().clone())

           nuanalysis = assimilate1(nusamples, obsarray, obs_covar)
           nuextsamples = np.repeat(nuanalysis[:,:,np.newaxis,:,:],1,axis=2)
           nutorchsamples = torch.from_numpy(nuextsamples)
           nux_hat = nutorchsamples.permute(0,1,2,4,3)
           nux_hat_to_assimilate[:, :, :, :nx, :]=nux_hat

           nuupdatehat = nux_hat_to_assimilate.cpu().numpy()
           nuupdatehat[:,:,:,:,0] = nuupdatehat[:,:,:,:,1]
           #print('CHECK=', nuupdatehat[0,0,0,0,1])
           nusamples = nuextsamples.tolist()
           nusamples = [_tensor for _tensor in nusamples if isinstance(_tensor, torch.Tensor)]
           #exit()

        elif i > 0 and i%50== 0:

           x_masked = torch.cat((history, future), dim=1).to(config.device)  # (B, T, V, F)
           x_masked = x_masked.transpose(1, 3)  # (B, F, V, T)
           x_masked = x_masked.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)

           obsdata = x_masked 
           obsarray = obsdata.transpose(2,4).cpu().numpy()

           obs_covar = config.obs_err_var * torch.eye(40)
           obs_covar_ext0 = np.repeat(obs_covar[np.newaxis,:,:],1,axis=0)
           obs_covar_ext1 = np.repeat(obs_covar_ext0[:,np.newaxis,:,:],n_samples,axis=1)

           x_hat = model((updatehat.to(config.device), pos_w, pos_d), timestepnum, n_ddim, n_samples) # (B, n_samples, F, V, T)

           samples.append(x_hat.transpose(2,4).cpu().clone())
           ensmem.append(x_hat.transpose(2,4).cpu().clone())

           analysis = assimilate1(samples, obsarray, obs_covar)
           extsamples = np.repeat(analysis[:,:,np.newaxis,:,:],1,axis=2)
           torchsamples = torch.from_numpy(extsamples)
           x_hat = torchsamples.permute(0,1,2,4,3)
           x_hat[:,:,:,:,0] = x_hat[:,:,:,:,1]
           x_hat[:,:,:,:,1] = 0. 
           updatehat = x_hat

           samples = extsamples.tolist()
           samples = [_tensor for _tensor in samples if isinstance(_tensor, torch.Tensor)]


           x_ini = nuupdatehat
           for j in range(n_samples):
               x_ini[0, j, 0, :, 1] = fcst_step(x_ini[0, j, 0, :, 0])

           nux_hat_to_assimilate = torch.from_numpy(x_ini) 
           numshort = nux_hat_to_assimilate[:, :, :, :nx, :] 

           nusamples.append(numshort.transpose(2,4).cpu().clone())
           nuensmem.append(numshort.transpose(2,4).cpu().clone())

           nuanalysis = assimilate1(nusamples, obsarray, obs_covar)
           nuextsamples = np.repeat(nuanalysis[:,:,np.newaxis,:,:],1,axis=2)
           nutorchsamples = torch.from_numpy(nuextsamples)
           nux_hat = nutorchsamples.permute(0,1,2,4,3)
           nux_hat_to_assimilate[:,:,:,:nx,:] = nux_hat

           nuupdatehat = nux_hat_to_assimilate.cpu().numpy()
           nuupdatehat[:,:,:,:,0] = nuupdatehat[:,:,:,:,1]

           nusamples = nuextsamples.tolist()
           nusamples = [_tensor for _tensor in nusamples if isinstance(_tensor, torch.Tensor)]

        else:

           x_hat = model((updatehat.to(config.device), pos_w, pos_d), timestepnum, n_ddim, n_samples) # (B, n_samples, F, V, T)
           ensmem.append(x_hat.transpose(2,4).cpu().clone())

           x_hat[:,:,:,:,0] = x_hat[:,:,:,:,1]
           x_hat[:,:,:,:,1] = 0.
           updatehat = x_hat


           x_ini = nuupdatehat

           for j in range(n_samples):
               x_ini[0, j, 0, :, 1] = fcst_step(x_ini[0, j, 0, :, 0])

           nux_hat_to_assimilate = torch.from_numpy(x_ini)

           numshort = nux_hat_to_assimilate[:, :, :, :nx, :] 
           nuensmem.append(numshort.transpose(2,4).cpu().clone())

           nuupdatehat = nux_hat_to_assimilate.cpu().numpy()
           nuupdatehat[:,:,:,:,0] = nuupdatehat[:,:,:,:,1]
           #print('CHECK=', nuupdatehat[0,0,0,0,1])
           #exit()


    if mode == 'test': # save the prediction result to file
        targets = torch.cat(targets, dim=0)[config.dataindf:config.dataind]

        checksamples = torch.cat(ensmem, dim=0)[config.dataindf:config.dataind]
        checknoda = torch.cat(nuensmem, dim=0)[config.dataindf:config.dataind]

        observed_flag = torch.ones_like(targets) #(B, T, V, F)
        evaluate_flag = torch.zeros_like(targets) #(B, T, V, F)
        observed_flag[:, :config.model.T_h, :, :] = 1
        evaluate_flag[:, -config.model.T_p:, :, :] = 1

        import pickle
        ### Now Visualization
        #with open (config.forecast_path, 'wb') as f:
        #    pickle.dump([samples, targets, observed_flag, evaluate_flag], f)

        #message = f"predict_path = '{config.forecast_path}'"
        #config.logger.message_buffer += f"{message}\n"
        #config.logger.write_message_buffer()

        # Save data
        with open('ens20_twolayer_data_x40_ddpm.pkl', 'wb') as f:
            # Save both the arrays and the config to the pickle file
            #pickle.dump(config, f)
            pickle.dump(targets, f)
            pickle.dump(checksamples, f)
            pickle.dump(checknoda, f)

        draw_predicted_distribution(checksamples, targets, checknoda, observed_flag, evaluate_flag, config)


    torch.cuda.empty_cache()
    return nuupdatehat 



from pprint import  pprint
def main(params: dict):
    # torch.manual_seed(2022)
    setup_seed(2022)
    torch.set_num_threads(2)
    config = default_config(params['data'])
    confignu = nu_config(params['data'])

    config.is_test = params['is_test']
    config.is_da = params['is_da']
    config.nni = params['nni']
    config.lr = params['lr']
    config.batch_size = params['batch_size']
    config.mask_ratio = params['mask_ratio']

    # model
    config.model.N = params['N']
    config.T_h = config.model.T_h = params['T_h']
    config.T_p = config.model.T_p =  params['T_p']
    config.model.epsilon_theta =  params['epsilon_theta']
    config.model.sample_steps = params['sample_steps']
    config.model.d_h = params['hidden_size']
    config.model.C = params['hidden_size']
    config.model.n_channels = params['hidden_size']
    config.model.beta_end = params['beta_end']
    config.model.beta_start = params['beta_start']
    config.model.beta_schedule = params["beta_schedule"]
    config.model.sample_strategy = params["ss"]
    config.n_samples = params['n_samples']

    if config.model.sample_steps > config.model.N:
        #print('sample steps large than N, exit')
        # nni.report_intermediate_result(50)
        nni.report_final_result(50)
        return 0


    config.trial_name = '+'.join([f"{v}" for k, v in params.items()])
    config.log_path = f"{config.PATH_LOG}/{config.trial_name}.log"

    pprint(config)
    dir_check(config.log_path)
    config.logger.open(config.log_path, mode="w")
    #log parameters
    config.logger.write(config.__str__()+'\n', is_terminal=False)

    #  data pre-processing
    clean_data = FullDataset(config)
    clean_data_nu = FullDataset(config)
    config.model.A = clean_data.adj

    model = DiffSTG(config.model)
    model = model.to(config.device)

    # Load training dataset
    train_dataset = TrafficDataset(clean_data, (0 + config.model.T_p, config.data.val_start_idx - config.model.T_p + 1), config)
    train_loader = torch.utils.data.DataLoader(train_dataset, config.batch_size, shuffle=True, pin_memory=True)

    val_dataset = TrafficDataset(clean_data, (config.data.val_start_idx + config.model.T_p, config.data.test_start_idx - config.model.T_p + 1), config)
    val_loader = torch.utils.data.DataLoader(val_dataset, 1, shuffle=False)

    test_dataset = TrafficDataset(clean_data, (config.data.test_start_idx + config.model.T_p, -1), config)
    test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False)
    nu_dataset = TrafficDataset(clean_data_nu, (confignu.data.test_start_idx + config.model.T_p, -1), config)
    nu_loader = torch.utils.data.DataLoader(nu_dataset, 1, shuffle=False)
    ini_dataset = TrafficDataset(clean_data, (config.data.val_start_idx, -1), config)
    ini_loader = torch.utils.data.DataLoader(ini_dataset, 1, shuffle=True)


    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # metrics in val, and test dataset, note that we cannot evaluate the performance in the train dataset
    metrics_val = Metric(T_p=config.model.T_h + config.model.T_p)

    model_path = './nodedrop.dmstg4'
    #*****************************************************************


    config.model_path = model_path
    config.logger.write(f"model path:{model_path}\n", is_terminal=False)
    print('model_path:', model_path)
    dir_check(model_path)

    config.forecast_path = forecast_path = config.PATH_FORECAST + config.trial_name + '.pkl'
    config.logger.write(f"forecast_path:{model_path}\n", is_terminal=False)
    print('forecast_path:', forecast_path)
    dir_check(forecast_path)


    # log model architecture
    #print(model)
    config.logger.write(model.__str__())

    # log training process
    config.logger.write(f'Num_of_parameters:{sum([p.numel() for p in model.parameters()])}\n', is_terminal=True)
    message = "      |---Train--- |---Val Future-- -|-----Val History----|\n"
    config.logger.write(message, is_terminal=True)

    message = "Epoch | Loss  Time | MAE     RMSE    |  MAE    RMSE   Time|\n" #f"{'Type':^7}{'Epoch':^7} | {'MAE':^7}{'RMSE':^7}{'MAPE':^7}
    config.logger.write(message, is_terminal=True)


    train_start_t = timer()
    # Train and sample the data
    for epoch in range(config.epoch):
        if not params['is_train']: break
        if epoch > 1 and config.is_da: break

        n, avg_loss, time_lst = 0, 0, []
        # train diffusion model
        for i, batch in enumerate(train_loader):
            if i > 3 and config.is_test:break
            time_start =  timer()
            future, history, pos_w, pos_d = batch # future:(B, T_p, V, F), history: (B, T_h, V, F)

            # get x0
            x = torch.cat((history, future), dim=1).to(config.device) #  (B, T, V, F)

            # get x0_masked
            mask =  torch.randint_like(history, low=0, high=100) < int(config.mask_ratio * 100)# mask the history in a ratio with mask_ratio
            history[mask] = 0
            x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device) # (B, T, V, F)

            # reshape
            x = x.transpose(1,3) # (B, F, V, T)
            x_masked = x_masked.transpose(1,3) # (B, F, V, T)

            # loss calculate
            loss = 10 * model.loss(x, (x_masked, pos_w, pos_d))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate the moving average training loss
            n += 1
            avg_loss = avg_loss * (n - 1) / n + loss.item() / n

            time_lst.append((timer() - time_start))
            message = f"{i / len(train_loader) + epoch:6.1f}| {avg_loss:0.3f} {np.sum(time_lst):.1f}s"
            print('\r' + message, end='', flush=True)

        config.logger.message_buffer += message

        try:
            writer.add_scalar('train/loss', avg_loss, epoch)
        except:
            pass


        if metrics_val.best_metrics['epoch'] == epoch:
            #print('[save model]>> ', model_path)
            torch.save(model, model_path)

        if epoch - metrics_val.best_metrics['epoch'] > config.early_stop: break  # Early_stop


    try:
        model = torch.load(model_path, map_location=config.device)
        print('best model loaded from: <<', model_path)
    except Exception as err:
        print(err)
        print('load best model failed')

    # conduct multiple-samples, then report the best
    metric_lst = []
    #for sample_strategy, sample_steps in [('ddim_multi', config.n_ddim)]:
    for sample_strategy, sample_steps in [('ddpm', 50)]:
        if sample_steps > config.model.N: break

        config.model.sample_strategy = sample_strategy
        config.model.sample_steps = sample_steps

        model.set_ddim_sample_steps(sample_steps)
        model.set_sample_strategy(sample_strategy)

        metrics_test = Metric(T_p=config.model.T_h + config.model.T_p)
        cycles(model, ini_loader, test_loader, nu_loader, epoch, metrics_test, config, confignu, clean_data, clean_data_nu, mode='test')


# data.name	model	model.N	model.epsilon_theta	model.d_h	model.T_h	model.T_p	model.sample_strategy
# PEMS08	UGnet	100	    UGnet	            32	        12	        12	        ddpm

if __name__ == '__main__':

    import nni
    import logging

    logger = logging.getLogger('training')

    #print('GPU:', torch.cuda.current_device())
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
