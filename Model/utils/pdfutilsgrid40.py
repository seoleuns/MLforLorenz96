# -*- coding: utf-8 -*-
# The initial version of this code is obtained from https://github.com/wenhaomin/DiffSTG
#
# The code is customized to apply for the Lorenz '96 system
# by Seoleun Shin (KRISS): Diverse Evaluation Metrics and Plots
#
# This version is saved at 15-APRIL-2025
# Publically added at https://github.com/seoleuns/MLforLorenz96

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

import os
import torch
def get_workspace():
    """
    get the workspace path, i.e., the root directory of the project
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file
ws =  get_workspace()


def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)
    return path

def to_device(batch, device):
    batch = [x.to(device) for x in batch]
    return batch


# merge all the dict in the list
def dict_merge(dict_list = []):
    dict_ =  {}
    for dic in dict_list:
        assert isinstance(dic, dict), "object is not a dict!"
        dict_ = {**dict_, **dic}
    return dict_

def unfold_dict(in_dict):
    from easydict import EasyDict as edict
    # convert 2 level easydict to 1 level, mainly for record the results.
    out_dict = {}
    for k1, v1 in in_dict.items():
        if isinstance(v1, edict) or isinstance(v1, dict):
            for k2, v2 in v1.items():
                out_dict[f'{k1}.{k2}'] = v2
        else:
            out_dict[k1] = v1
    return out_dict


def save2file_meta(params, file_name, head):
    def timestamp2str(stamp):
        utc_t = int(stamp)
        utc_h = utc_t // 3600
        utc_m = (utc_t // 60) - utc_h * 60
        utc_s = utc_t % 60
        hour = (utc_h + 8) % 24
        t = f'{hour}:{utc_m}:{utc_s}'
        return t
    import csv, time, os
    dir_check(file_name)
    if not os.path.exists(file_name):
        f = open(file_name, "w", newline='\n', encoding='utf-8')
        csv_file = csv.writer(f)
        csv_file.writerow(head)
        f.close()

    df = pd.read_csv(file_name, encoding='utf-8')
    old_head = df.columns
    if len(set(head)) > len(set(old_head)):
        f = open(file_name, "w", newline='\n')
        csv_file = csv.writer(f)
        csv_file.writerow(head) # write new head
        for idx, data_df in df.iterrows():
            data = [data_df[k] if k in old_head else -1 for k in head]
            csv_file.writerow(data)
        f.close()

    with open(file_name, "a", newline='\n', encoding='utf-8') as file:
        csv_file = csv.writer(file)
        params['log_time'] = timestamp2str(time.time())
        data = [params[k] for k in head]
        csv_file.writerow(data)


def GpuId2CudaId(gpu_id):

    # return {0:8, 1:4, 2:0, 3:5, 4:1, 5:6, 6:9, 7:7, 8:2, 9:3}[gpu_id]# for server in NUS, where the gpu id does not equal the cuda id in the server.
    return {i:i for i in range(8)}.get(gpu_id, 0)



# print logger
class Logger(object):
    def __init__(self):
        import sys
        self.terminal = sys.stdout  #stdout
        self.file = None
        self.message_buffer = ''

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=True, is_file=True):
        if '\r' in message: is_file=False

        if is_terminal:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file:
            self.file.write(message)
            self.file.flush()

    def write_message_buffer(self):
        self.write(self.message_buffer, is_terminal=False)
        self.message_buffer = ''


def shape_correct(x: torch.Tensor, expected_shape: list):
    # automatic convert a tensor to the expected_shape
    # e.g., x: (B, T, V, F) --> (B, V, T, F)
    dim4idx = {d: i for i, d in enumerate(x.shape)}
    assert len(expected_shape) == len(dim4idx), "length of expected shape does not equal to the input shape"
    permute_idx = [dim4idx[d] for d in expected_shape]
    x = x.permute(tuple(permute_idx))
    return x



from multiprocessing import Pool
def multi_thread_work(parameter_queue,function_name,thread_number=5):
    pool = Pool(thread_number)
    result = pool.map(function_name, parameter_queue)
    pool.close()
    pool.join()
    return  result

import matplotlib.pyplot as plt
def draw_predicted_distribution(samples, target, noda, observed_flag, evaluate_flag, config={}):

    """
    All should be torch.Tensor
    :param samples: (B, n_samples, T, V, F)
    :param label: (B, T, V, F)
    :param observed_flag: (B, T, V, F), equals 1 if the data is observed in the data
    :param evaluate_flag: (B, T, V, F), equals 1 if the data if we want to draw the distribution of this data
    :return:
    """

    def get_quantile(samples, q, dim=1):
        distribution =  torch.quantile(samples, q, dim=dim).cpu().numpy()
        result = np.repeat(distribution[:,:,:,np.newaxis],1,axis=3)
        return result
    dataind = config.get('dataint', 1)  # change to visualize a different sample
    B, n_samples, T, V, F = samples.shape
    Btest, n_test, Ttest, Vtest, Ftest = noda.shape
    #print('CHECKTNUM:', Ftest)
    # take out the last feature
    all_sample_np_all = samples[:, :, :, :, 0].cpu().numpy()
    all_sample_np_all =  all_sample_np_all.reshape(dataind, n_samples, 2, 40)
    all_sample_np = np.mean(all_sample_np_all, axis=1)
    all_sample_np_time = np.mean(all_sample_np_all, axis=0)
    all_noda_np_all = noda[:, :, :, :, 0].cpu().numpy()
    all_noda_np_all =  all_noda_np_all.reshape(dataind, n_samples, 2, 40)
    all_noda_np = np.mean(all_noda_np_all, axis=1)
    all_noda_np_time = np.mean(all_noda_np_all, axis=0)
    # Indi mem check
    all_target_np_all = target[:, :, :,:].cpu().numpy()
    all_target_np = all_target_np_all[:,:,:,0]
    all_observed_np = observed_flag[:, :, :].cpu().numpy()
    all_evalpoint_np = evaluate_flag[:, :, :].cpu().numpy()

    all_sample_np_mean = np.repeat(all_sample_np[:,np.newaxis,:,:],n_samples,axis=1)
    all_deviation_ml = all_sample_np_all[:,:,0,:] - all_sample_np_mean[:,:,0,:]
    sprdml = np.std(all_deviation_ml, axis=1)
    sprdml = np.mean(sprdml, axis=1)
    sprdmlall = np.mean(sprdml, axis=0)
    sprdml = sprdml.reshape(dataind, 1)

    print('SPREADCHECK1:', sprdmlall)

    all_noda_np_mean = np.repeat(all_noda_np[:,np.newaxis,:,:],n_samples,axis=1)
    all_deviation_phys = all_noda_np_all[:,:,0,:]- all_noda_np_mean[:,:,0,:]
    sprdphys = np.std(all_deviation_phys, axis=1)
    sprdphys = np.mean(sprdphys, axis=1)
    sprdphysall = np.mean(sprdphys, axis=0)
    sprdphys = sprdphys.reshape(dataind, 1)

    print('SPREADCHECK2:', sprdphysall)

    all_target_np_all_ext = np.repeat(all_target_np[:,np.newaxis,:,:],n_samples,axis=1)
    all_diff_ml = all_sample_np[:,0,:] - all_target_np[:,0, :]
    rmseml = np.sqrt(np.mean(all_diff_ml**2, axis=1))
    rmsemlall = np.mean(rmseml, axis=0)
    rmseml = rmseml.reshape(dataind, 1)

    all_diff_phys = all_noda_np[:,0,:] - all_target_np[:,0,:]
    rmsephys = np.sqrt(np.mean(all_diff_phys**2, axis=1))
    rmsephysall = np.mean(rmsephys, axis=0)
    rmsephys = rmsephys.reshape(dataind, 1)

    samplesq = samples[:,:,:,:,0]
    nodaq = noda[:,:,:,:,0]

    qlist = [0.025, 0.25, 0.5, 0.75, 0.975]
    q_ml = []
    q_phys = []
    for q in qlist:
        q_ml.append(get_quantile(samplesq, q, dim=1))  
        q_phys.append(get_quantile(nodaq, q, dim=1)) 

    v = 1
    evalpoint_slice_phase = all_target_np[:, 0, v]

    v = 20
    sample_slice = all_sample_np[:, 0, v]
    evalpoint_slice = all_target_np[:, 0, v]
    noda_slice = all_noda_np[:, 0, v]

    print('CHECKMEAN number:', rmsemlall)
    print('CHECKMEAN number2:', rmsephysall)
    dataset = all_target_np
    datasetmean = np.mean(dataset, axis=2)
    flattened_data = datasetmean[:,0]
    data_min, data_max = np.min(flattened_data), np.max(flattened_data)
    data_normalized = (flattened_data - data_min) / (data_max - data_min)
    plt.boxplot(data_normalized, vert=False)
    plt.title("Boxplot of Data")
    plt.savefig('hisgram.png')
    plt.close()

    t1 = np.mean(data_normalized)
    t2 = (np.percentile(data_normalized,25))
    t3 = (np.percentile(data_normalized,75))
    t4 = (np.percentile(data_normalized,90))
    print("MEAN=", t1, t2, t3, t4)

    rmse_slice1 = rmseml[:, 0]
    rmse_slice2 = rmsephys[:, 0]
    sprd_slice1 = sprdml[:, 0]
    sprd_slice2 = sprdphys[:, 0]
    mlquantiles_upper = q_ml[4][0:10000, 0, v].flatten()
    mlquantiles_lower = q_ml[0][0:10000, 0, v].flatten()
    physquantiles_upper = q_phys[4][0:10000, 0, v].flatten()
    physquantiles_lower = q_phys[0][0:10000, 0, v].flatten()

    upper2 = q_phys[4][:, 0, v].flatten()
    low2 = q_phys[0][:, 0, v].flatten()

# Create time array (assuming 200 time grids)
    time = np.arange(dataind)
    cinterval = 100
    cend = 5000

    plt.figure(figsize=(12, 10))
    plt.plot(evalpoint_slice_phase[:cend], evalpoint_slice[:cend], label='Phase_Diagram', color='black')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)

# Customize plot if needed
    plt.ylim(-20,25.)
    plt.xlim(-20,25.)
    plt.xlabel('$X_2$', fontsize=30)
    plt.ylabel('$X_3$', fontsize=30)
    plt.title('Phase Diagram of $X_2$ and $X_3$', fontsize=30)
    plt.legend()
    plt.savefig('phase_diagram_x2_x3.png')
    plt.close()

# Plot the sample and evalpoint slices
    plt.figure(figsize=(15, 10))
    plt.plot(time[0:10000:200], evalpoint_slice[0:10000:200], label='Ground Truth', linewidth=3.0, linestyle='solid', color='black')
    plt.plot(time[0:10000:200], sample_slice[0:10000:200], label='ML Prediction', linewidth=3.5, linestyle='solid', color='gray')
    plt.plot(time[0:10000:200], noda_slice[0:10000:200], label='PhysM Prediction', linewidth=3.5, linestyle='dashed', color='gray')
    plt.fill_between(range(0,10000,200), physquantiles_lower[::200], physquantiles_upper[::200], color='cyan', alpha=0.4, label='95% Quantile Phys')
    plt.fill_between(range(0,10000,200), mlquantiles_lower[::200], mlquantiles_upper[::200], color='magenta', alpha=0.4, label='95% Quantile ML')
    plt.xticks(size = 24)
    plt.yticks(size = 24)

# Customize plot if needed
    #plt.ylim(-5.0,5.0)
    plt.ylim(-15,25.)
    plt.xlabel('Time Steps', fontsize=30)
    plt.ylabel('Value', fontsize=30)
    plt.title('Time Series of $X_{20}$', fontsize=30)
    plt.legend(fontsize=20)
    plt.savefig('40grid_1tp_10thda_ml_ddpm.png')
    plt.close()

    plt.figure(figsize=(15, 10))
    plt.plot(time[0:10000:200], rmse_slice1[0:10000:200], label='ML Prediction RMSE', linewidth=3.5, linestyle='solid', color='black')
    plt.plot(time[0:10000:200], rmse_slice2[0:10000:200], label='PhysM Prediction RMSE', linewidth=3.5, linestyle='solid', color='gray')
    plt.plot(time[0:10000:200], sprd_slice1[0:10000:200], label='ML Prediction SPRD', linewidth=3.5, linestyle='dashed', color='black')
    plt.plot(time[0:10000:200], sprd_slice2[0:10000:200], label='PhysM Prediction SPRD', linewidth=3.5, linestyle='dashed', color='gray')
    plt.xticks(size = 24)
    plt.yticks(size = 24)

# Customize plot if needed
    plt.ylim(0, 10)
    plt.xlabel('Time Steps', fontsize=30)
    plt.ylabel('Value', fontsize=30)
    plt.title('RMSD/SPRD Time Series', fontsize=30)
    plt.legend(fontsize=20)
    plt.savefig('40grid_1tp_10thda_rmse_sprd_ddpm.png')
    plt.close()

    def get_crps(observation, ensemble):
        #print("Computing CRPS")

        # Get number of ensemble members
        num_mem = n_samples

        # Extract raw arrays
        ens_arr = ensemble
        obs_arr = observation

        # Compute mean innovations
        #print("Computing mean innovations")
        mean_innov = np.mean(np.abs(ens_arr - obs_arr), axis=0)

        # Compute mean ensemble member difference across all member pairs
        #print("Computing mean pair differences")
        mean_pair_difference = np.zeros(obs_arr.shape)
        for i, mem in enumerate(ensemble):
            mem_arr = mem
            #print("MEMMEM=", mem_arr.shape)
            mean_pair_difference += np.sum(np.abs(mem_arr[:] - ens_arr[i,:]), axis=0)

        if small_ensemble_correction:
            mean_pair_difference /= 2*num_mem*(num_mem-1)
        else:
            mean_pair_difference /= 2*num_mem**2

        #crps = mean_innov - mean_pair_difference
        crps = np.mean(mean_innov - mean_pair_difference)

        return crps

    small_ensemble_correction = True
    mlcrps = np.zeros(noda_slice.shape)
    physcrps = np.zeros(noda_slice.shape)

    # Compute space-mean time-mean CRPS
    for i in range (dataind):
        mlcrps[i] = get_crps(all_target_np[i,0,:], all_sample_np_all[i,:,0,:])
        physcrps[i] = get_crps(all_target_np[i,0,:], all_noda_np_all[i,:,0,:])

    #print(f"CRPS = {crps.data}")
    plt.figure(figsize=(15, 10))
    plt.plot(time[0:10000:200], mlcrps[0:10000:200], label='ML Prediction', linewidth=3.5, linestyle='solid', color='gray')
    plt.plot(time[0:10000:200], physcrps[0:10000:200], label='PhysM Prediction', linewidth=3.5, linestyle='dashed', color='gray')
    plt.xticks(size = 24)
    plt.yticks(size = 24)

# Customize plot if needed
    #plt.ylim(-5.0,5.0)
    plt.ylim(0,10.)
    plt.xlabel('Time Steps', fontsize=30)
    plt.ylabel('Value', fontsize=30)
    plt.title('Time Series of CRPS', fontsize=30)
    plt.legend(fontsize=20)
    plt.savefig('crps.png')
    plt.close()

    x_train = all_sample_np_all[:,:,0,20] 
    x_train1 = all_noda_np_all[:,:,0,20] 
    x_train0 = all_target_np[:,0,20]

    x_trainfull = all_sample_np_all[:,:,0,:] 
    x_train1full = all_noda_np_all[:,:,0,:] 
    x_train0full = all_target_np[:,0,:]

    # Renormalise input data
    max_train = 35.0
    min_train = -20.0
    x_train = 2.0*(x_train - min_train)/(max_train - min_train) - 1.0
    x_train0 = 2.0*(x_train0 - min_train)/(max_train - min_train) - 1.0
    x_train1 = 2.0*(x_train1 - min_train)/(max_train - min_train) - 1.0
    x_trainfull = 2.0*(x_trainfull - min_train)/(max_train - min_train) - 1.0
    x_train0full = 2.0*(x_train0full - min_train)/(max_train - min_train) - 1.0
    x_train1full = 2.0*(x_train1full - min_train)/(max_train - min_train) - 1.0

    count0, bins_count0 = np.histogram(x_train0, bins=20)
    count, bins_count = np.histogram(x_train, bins=20)
    count1, bins_count1 = np.histogram(x_train1, bins=20)
    count0full, bins_count0full = np.histogram(x_train0full, bins=20)
    countfull, bins_countfull = np.histogram(x_trainfull, bins=20)
    count1full, bins_count1full = np.histogram(x_train1full, bins=20)

    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
    pdf0 = count0 / sum(count0)
    pdf1 = count1 / sum(count1)

    pdffull = countfull / sum(countfull)
    pdf0full = count0full / sum(count0full)
    pdf1full = count1full / sum(count1full)

    plt.figure(figsize=(12, 8))
    plt.plot(bins_count[1:], pdf, color="red", label="PDF_ML", linewidth=3.0)
    plt.legend(loc='upper left')
    plt.plot(bins_count1[1:], pdf1, color="green", label="PDF_PHYS", linewidth=3.0)
    plt.legend(loc='upper left')
    plt.plot(bins_count0[1:], pdf0, 'r--', label="PDF_truth", linewidth=3.0)
    plt.legend(loc='upper left')

    plt.ylim((0.0,0.2))
    plt.xlim((-1.2,1.2))
    plt.xlabel("Normalised $x_{20}$", fontsize=20)
    plt.ylabel(" PDF", fontsize=20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.savefig('pdfplotx20.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(bins_countfull[1:], pdffull, color="red", label="PDF_ML", linewidth=3.0)
    plt.legend(loc='upper left')
    plt.plot(bins_count1full[1:], pdf1full, color="green", label="PDF_PHYS", linewidth=3.0)
    plt.legend(loc='upper left')
    plt.plot(bins_count0full[1:], pdf0full, 'r--', label="PDF_truth", linewidth=3.0)
    plt.legend(loc='upper left')

    plt.ylim((0.0,0.2))
    plt.xlim((-1.2,1.2))
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylabel(" PDF", fontsize=20)
    plt.xlabel("Normalised $\mathbf{x}$", fontsize=20)
    plt.savefig('pdfplot.png')
    plt.close()
    #exit()

# Plot histogram of signal values to visualize the distribution
    from scipy.stats import norm
    time_mean_sample = np.mean(all_target_np, axis=0)
    mean_signal = time_mean_sample[0,:]
    min_signal = np.min(mean_signal)
    max_signal = np.max(mean_signal)
    normalized_signal = (mean_signal - min_signal) / (max_signal - min_signal)
    plt.figure(figsize=(10, 6))
    plt.hist(normalized_signal, bins=20, density=True, alpha=0.6, color='g', label='Signal Distribution')

# Fit a Gaussian distribution to the signal data
    mu, std = norm.fit(normalized_signal)

    # Plot the Gaussian fit
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label=f'Gaussian Fit: $\mu={mu:.2f}$, $\sigma={std:.2f}$')

# Adding labels and title
    plt.title(f"Probability Distribution of Signal Rate")
    plt.xlabel("Signal Value")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.savefig('signalratedist.png')
    plt.close()

# Return the fit parameters
    print("ResultedSignal", mu, std)

# Plot histogram of ensemble values to visualize the distribution
    var_mean_truth = np.mean(all_target_np[:,0,:], axis=0)
    var_mean_sample = all_sample_np_time[:,0,:]
    var_mean_noda = all_noda_np_time[:,0,:]

    # Compute ranks of the truth within ensemble members
    ranks1 = np.sum(var_mean_sample < var_mean_truth[None,:], axis=0)
    mean_rank = np.mean(ranks1)
    median_rank = np.median(ranks1)
    std_rank = np.std(ranks1)

    plt.figure(figsize=(8, 5))
    # Add quantitative analysis text
    plt.text(0.7, 25, f"Mean Rank: {mean_rank:.2f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.8), transform=plt.gca().transAxes)

    ranks2 = np.sum(var_mean_noda < var_mean_truth[None,:], axis=0)
    # Plot rank histogram
    plt.hist(ranks1, bins=np.arange(n_samples + 2) - 0.5, edgecolor='black', facecolor='pink', alpha=0.9)

    mean_rank2 = np.mean(ranks2)
    median_rank2 = np.median(ranks2)
    std_rank2 = np.std(ranks2)

    # Add quantitative analysis text
    plt.text(0.7, 25, f"Mean Rank: {mean_rank2:.2f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.8), transform=plt.gca().transAxes)

    plt.hist(ranks2, bins=np.arange(n_samples + 2) - 0.5, edgecolor='black', facecolor='cyan', alpha=0.4)
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Rank Histogram of ML and Phys Forecast")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(['ML', 'Phys'])
    #plt.grid(True)
    plt.xticks(range(n_samples + 1))
    plt.savefig('ensemblehist.png')
    plt.close()

    # Plot CDF
    plt.figure(figsize=(8, 5))

    # Compute CDF for Ensemble 1
    sorted_ranks1 = np.sort(ranks1)
    cdf_ensemble1 = np.arange(1, len(sorted_ranks1) + 1) / len(sorted_ranks1)
    plt.plot(sorted_ranks1, cdf_ensemble1, label="ML", color='red')

    # Compute CDF for Ensemble 2
    sorted_ranks2 = np.sort(ranks2)
    cdf_ensemble2 = np.arange(1, len(sorted_ranks2) + 1) / len(sorted_ranks2)
    plt.plot(sorted_ranks2, cdf_ensemble2, label="Phys", color='blue')

    # Quantitative Analysis
    mean_rank = np.mean(ranks1)
    median_rank = np.median(ranks1)
    std_rank = np.std(ranks1)

    mean_rank2 = np.mean(ranks2)
    median_rank2 = np.median(ranks2)
    std_rank2 = np.std(ranks2)

    # Add quantitative analysis text
    plt.text(0.7, 0.4, f"ML \nMean Rank: {mean_rank:.2f}\nMedian Rank: {median_rank}\nStd Rank: {std_rank:.2f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.8), transform=plt.gca().transAxes)
    plt.text(0.7, 0.2, f"Phys \nMean Rank: {mean_rank2:.2f}\nMedian Rank: {median_rank2}\nStd Rank: {std_rank2:.2f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.8), transform=plt.gca().transAxes)

    # Customize plot
    plt.xlabel("Rank")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF Comparison of Two Ensembles with Truth")
    plt.legend()
    plt.grid(True)
    plt.savefig('cdfplot.png')
    plt.close()


# Function to compute cross-correlation
    from scipy.signal import correlate
    def cross_correlation(truth, series, label):
        lags = np.arange(-len(truth) + 1, len(truth))
        corr = correlate(series - np.mean(series), truth - np.mean(truth), mode='full') / (np.std(series) * np.std(truth) * len(truth))
        return lags, corr

    lags1, corr1 = cross_correlation(evalpoint_slice, sample_slice, "ML")
    lags2, corr2 = cross_correlation(evalpoint_slice, noda_slice, "Phys")

    # Plotting cross-correlation
    plt.figure(figsize=(12, 8))
    plt.plot(lags1, corr1, label='ML  vs Truth', color='red', linewidth=3.0)
    plt.plot(lags2, corr2, label='Phys  vs Truth', color='blue', linewidth=3.0)
    plt.axvline(0, color='black', linestyle='--', label='Zero Lag', linewidth=2.0)
    plt.xlabel('Lag', fontsize=20)
    plt.ylabel('Cross-Correlation', fontsize=20)
    plt.title('Cross-Correlation of Time Series', fontsize=20)
    plt.ylim(-0.6,1.0)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.legend()
    plt.grid()
    plt.savefig('crosscorr.png')
    plt.close()

    from scipy.fftpack import fft

   # Compute FFT
    def compute_phase(series):
        fft_vals = fft(series)
        return np.angle(fft_vals)  # Extract phase in radians

    # Get phase differences
    freqs = np.fft.fftfreq(len(evalpoint_slice))
    phase_truth = compute_phase(evalpoint_slice)
    phase_series1 = compute_phase(sample_slice)
    phase_series2 = compute_phase(noda_slice)

    # Compute absolute phase difference
    phase_diff1 = np.abs(phase_series1 - phase_truth)
    phase_diff2 = np.abs(phase_series2 - phase_truth)

    # Normalize phase differences to be between 0 and π
    phase_diff1 = np.minimum(phase_diff1, 2 * np.pi - phase_diff1)
    phase_diff2 = np.minimum(phase_diff2, 2 * np.pi - phase_diff2)
    mean_phase_diff1 = np.mean(phase_diff1)
    mean_phase_diff2 = np.mean(phase_diff2)

    print(f"Mean Phase Deviation of ML: {mean_phase_diff1:.4f} radians")
    print(f"Mean Phase Deviation of Phys: {mean_phase_diff2:.4f} radians")

    # Compare which is closer to the truth
    if mean_phase_diff1 < mean_phase_diff2:
            print("ML is more in phase with the truth.")
    else:
            print("Phys is more in phase with the truth.")

# Plot the power spectrum (log-log scale)

    grid_points = 40
    sample_slice = all_target_np[5000:10000, 0, :]
    time_mean_sample_slice = np.mean(sample_slice, axis=0)
    state_values = time_mean_sample_slice 

# Perform Fast Fourier Transform (FFT) on spatial data
    fourier_image = np.fft.fft(state_values)
    fourier_amplitudes = np.abs(fourier_image) ** 2
    #fs = 92e-07/grid_points
    #fs = (1/grid_points)*np.pi*2*grid_points
    fs = grid_points/2 
    freq = fs/2 * np.linspace(0,1,int(grid_points/2))

    kfreq = np.fft.fftfreq(grid_points)*grid_points
    knrm = np.abs(kfreq)
    kbins = np.arange(0.5, grid_points//2+1, 1.)
    kvals = 0.5*(kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins = kbins)
    Abins *= np.pi*(kbins[1:]**2 - kbins[:-1]**2)
    power_spectrum = Abins

    
# ML
    sample_slice_ml = all_sample_np[5000:10000, 0, :]
    time_mean_sample_slice = np.mean(sample_slice_ml, axis=0)
    state_values = time_mean_sample_slice 

    fourier_image = np.fft.fft(state_values)
    fourier_amplitudes = np.abs(fourier_image) ** 2
    fs = grid_points/2 
    freq = fs/2 * np.linspace(0,1,int(grid_points/2))

    kfreq = np.fft.fftfreq(grid_points)*grid_points
    knrm = np.abs(kfreq)
    kbins = np.arange(0.5, grid_points//2+1, 1.)
    kvals = 0.5*(kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins = kbins)
    Abins *= np.pi*(kbins[1:]**2 - kbins[:-1]**2)
    power_spectrum_ml = Abins


# Physical
    sample_slice_phys = all_noda_np[5000:10000, 0, :]
    time_mean_sample_slice = np.mean(sample_slice_phys, axis=0)
    state_values = time_mean_sample_slice 

    fourier_image = np.fft.fft(state_values)
    fourier_amplitudes = np.abs(fourier_image) ** 2
    fs = grid_points/2 
    freq = fs/2 * np.linspace(0,1,int(grid_points/2))

    kfreq = np.fft.fftfreq(grid_points)*grid_points
    knrm = np.abs(kfreq)
    kbins = np.arange(0.5, grid_points//2+1, 1.)
    kvals = 0.5*(kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins = kbins)
    Abins *= np.pi*(kbins[1:]**2 - kbins[:-1]**2)
    power_spectrum_phys = Abins


# Plot the power spectrum (log-log scale)

    plt.figure(figsize=(12, 8))
    plt.loglog(fs/kvals, power_spectrum, label='Truth', linewidth=3.0, linestyle='solid', color='black')
    plt.loglog(fs/kvals, power_spectrum_ml, label='ML', linewidth=3.5, linestyle='solid', color='gray')
    plt.loglog(fs/kvals, power_spectrum_phys, label='Phys', linewidth=3.5, linestyle='dashed', color='gray')

    plt.xlabel('Spatial Frequency', fontsize=20)
    plt.ylabel('Power', fontsize=20)
    plt.title('Spatial Power Spectrum wrt. Frequency', fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize = 20)
    plt.ylim(1.0e01, 1.0e05)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize=20)
    plt.savefig('spectrum.png')
    plt.close()

    # Create the Hovmöller diagram
    xval = np.linspace(0, 40, 40) 
    alltargetplot = np.reshape(all_target_np[:,0,:], (dataind, grid_points))
    X, Y = np.meshgrid(xval, time)
    allsampleplot = np.reshape(all_sample_np[:,0,:], (dataind, grid_points))
    allphysplot = np.reshape(all_noda_np[:,0,:], (dataind, grid_points))

    rows=1
    cols=3
    fig, ax = plt.subplots(1,3,figsize=(15,9), facecolor='white')
    fig.set_facecolor("white")
    im0=ax[0].imshow(alltargetplot, aspect='auto', extent=[xval.min(), xval.max(), time.max(), time.min()], cmap='viridis')
    ax[0].set_ylabel('Time Steps', fontsize=18)
    ax[0].set_xlabel('Space (x)', fontsize=18)
    ax[0].set_title("Ground Truth", fontsize=18)
    im1=ax[1].imshow(allsampleplot, aspect='auto', extent=[xval.min(), xval.max(), time.max(), time.min()], cmap='viridis')
    ax[1].set_xlabel('Space (x)', fontsize=18)
    ax[1].set_title("ML prediction", fontsize=18)
    im2 = ax[2].imshow(allphysplot, aspect='auto', extent=[xval.min(), xval.max(), time.max(), time.min()], cmap='viridis')
    ax[2].set_xlabel('Space (x)', fontsize=18)
    ax[2].set_title("Phys prediction", fontsize=18)
    fig.colorbar(im0, ax=ax, location='right', shrink=0.8, label='Data Value')
    fig.savefig('hovmoellerdiagram.png')
    fig.savefig('hovmoellerdiagram.eps', format ='eps')
    plt.close(fig)

    plt.figure(figsize=(6, 12), dpi=100)
    plt.subplots_adjust(left=0., right=1., top=1., bottom=0., wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(0,0)
    plt.contourf(xval, time, alltargetplot, levels=100, vmin=-15.0, vmax=20.0, cmap='gray')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('truehov.png')
    plt.close()

    plt.figure(figsize=(6, 12), dpi=100)
    plt.subplots_adjust(left=0., right=1., top=1., bottom=0., wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(0,0)
    plt.contourf(xval, time, allsampleplot, levels=100, vmin=-15.0, vmax=20.0, cmap='gray')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('mlhov.png')
    plt.close()

    plt.figure(figsize=(6, 12), dpi=100)
    plt.subplots_adjust(left=0., right=1., top=1., bottom=0., wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(0,0)
    plt.contourf(xval, time, allphysplot, levels=100, vmin=-15.0, vmax=20.0, cmap='gray')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('physhov.png')
    plt.close()


    exit()


















