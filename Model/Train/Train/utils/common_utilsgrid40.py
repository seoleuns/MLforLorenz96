# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.sparse as sp

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
    #print('SAMPLECHECK:', all_sample_np_all)
    all_sample_np_all =  all_sample_np_all.reshape(dataind, n_samples, 2, 40)
    all_sample_np = np.mean(all_sample_np_all, axis=1)
    all_noda_np_all = noda[:, :, :, :, 0].cpu().numpy()
    #print('NODACHECK:', all_noda_np_all)
    all_noda_np_all =  all_noda_np_all.reshape(dataind, n_samples, 2, 40)
    all_noda_np = np.mean(all_noda_np_all, axis=1)
    # Indi mem check
    #all_noda_np = all_noda_np_all[:,1,:,:]
    #print('NODAMEANCHECK:', all_noda_np[:, 0, 2])
    #import sys
    #np.set_printoptions(threshold=sys.maxsize)
    all_target_np_all = target[:, :, :,:].cpu().numpy()
    all_target_np = all_target_np_all[:,:,:,0]
    all_observed_np = observed_flag[:, :, :].cpu().numpy()
    all_evalpoint_np = evaluate_flag[:, :, :].cpu().numpy()

    all_sample_np_mean = np.repeat(all_sample_np[:,np.newaxis,:,:],n_samples,axis=1)
    all_deviation_ml = all_sample_np_all[:,:,0,:] - all_sample_np_mean[:,:,0,:]
    sprdml = np.std(all_deviation_ml, axis=1)
    sprdmlall = np.mean(sprdml)
    sprdml = sprdml.reshape(dataind, 1, 40)

    print('SPREADCHECK1:', sprdmlall)

    all_noda_np_mean = np.repeat(all_noda_np[:,np.newaxis,:,:],n_samples,axis=1)
    all_deviation_phys = all_noda_np_all[:,:,0,:]- all_noda_np_mean[:,:,0,:]
    sprdphys = np.std(all_deviation_phys, axis=1)
    sprdphysall = np.mean(sprdphys)
    sprdphys = sprdphys.reshape(dataind, 1, 40)

    print('SPREADCHECK2:', sprdphysall)

    all_target_np_all_ext = np.repeat(all_target_np[:,np.newaxis,:,:],n_samples,axis=1)
    all_diff_ml = all_sample_np_all[:,:,0,:] - all_target_np_all_ext[:,:,0, :]
    rmseml = np.std(all_diff_ml, axis=2)
    print('GivenCHECK1:', rmseml.shape)
    rmsemlall = np.mean(rmseml, axis=1)
    rmseml = rmsemlall.reshape(dataind, 1)
    rmsemlall = np.mean(rmsemlall, axis=0)

    all_diff_phys = all_noda_np_all[:,:, 0,:] - all_target_np_all_ext[:,:, 0,:]
    rmsephys = np.std(all_diff_phys, axis=2)
    rmsephysall = np.mean(rmsephys, axis=1)
    rmsephys = rmsephysall.reshape(dataind, 1)
    rmsephysall = np.mean(rmsephysall, axis=0)


    samples = samples[:,:,:,:,0]

    qlist = [0.025, 0.25, 0.5, 0.75, 0.975]
    q_ml = []
    q_phys = []
    for q in qlist:
        q_ml.append(get_quantile(samples, q, dim=1))  
        q_phys.append(get_quantile(noda, q, dim=1)) 

    #print("CHECKSAMPLEsample=", samples[:,1,1,0])
    #print("CHECKSAMPLEQ=", quantiles_imp[4])

    v = 2
    sample_slice = all_sample_np[:, 0, v]
    evalpoint_slice = all_target_np[:, 0, v]
    noda_slice = all_noda_np[:, 0, v]

    v = 1
    evalpoint_slice_phase = all_target_np[:, 0, v]
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
    t1 = np.mean(data_normalized)
    t2 = (np.percentile(data_normalized,25))
    t3 = (np.percentile(data_normalized,75))
    t4 = (np.percentile(data_normalized,90))
    print("MEAN=", t1, t2, t3, t4)
    rmse_slice1 = rmseml[:, 0]
    rmse_slice2 = rmsephys[:, 0]
    sprd_slice1 = sprdml[:, 0, v]
    sprd_slice2 = sprdphys[:, 0, v]
    quantiles_upper2 = q_ml[3][:, 0, v].flatten()
    quantiles_lower2 = q_ml[1][:, 0, v].flatten()
    quantiles_upper4 = q_ml[4][:, 0, v].flatten()
    quantiles_lower4 = q_ml[0][:, 0, v].flatten()
    #quantiles_mean = np.mean(quantiles_imp[:][:,0,v], axis=0)
    upper1 = q_ml[4][:, 0, v].flatten()
    low1 = q_ml[0][:, 0, v].flatten()
    upper2 = q_phys[4][:, 0, v].flatten()
    low2 = q_phys[0][:, 0, v].flatten()

# Create time array (assuming 200 time grids)
    time = np.arange(dataind)
    cinterval = 100
    cend = 1800

    plt.figure(figsize=(15, 10))
    plt.plot(evalpoint_slice_phase[:cend], evalpoint_slice[:cend], label='Phase_Diagram', color='black')
    plt.xticks(size = 24)
    plt.yticks(size = 24)

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
    #plt.plot(time, sample_slice, label='ML Prediction', color='red')
    #plt.plot(time, evalpoint_slice, label='Ground Truth', color='green')
    #plt.plot(time, noda_slice, label='PhysM Prediction', color='blue')
    plt.plot(time[::cinterval], evalpoint_slice[::cinterval], label='Ground Truth', linewidth=3.0, linestyle='solid', color='black')
    plt.plot(time[::cinterval], sample_slice[::cinterval], label='ML Prediction', linewidth=2.5, linestyle='solid', color='gray')
    plt.plot(time[::cinterval], noda_slice[::cinterval], label='PhysM Prediction', linewidth=2.5, linestyle='dashed', color='gray')
    #plt.fill_between(range(0,dataind), quantiles_lower, quantiles_upper, color='pink', alpha=0.3, label='95% Quantile')
    #plt.fill_between(range(0,dataind), quantiles_lower2, quantiles_upper2, color='pink', alpha=0.3, label='50% Quantile ML')
    #plt.fill_between(range(0,dataind), quantiles_lower4, quantiles_upper4, color='pink', alpha=0.3, label='97.5% Quantile')
    #plt.fill_between(range(0,dataind), low2, upper2, color='cyan', alpha=0.3, label='97.5% Quantile Phys')
    plt.xticks(size = 24)
    plt.yticks(size = 24)

# Customize plot if needed
    #plt.ylim(-5.0,5.0)
    plt.ylim(-10,25.)
    plt.xlabel('Time', fontsize=30)
    plt.ylabel('Value', fontsize=30)
    plt.title('Time Series (Truth: Green, Phys: Blue, ML: Red)', fontsize=30)
    plt.legend()
    plt.savefig('40grid_1tp_10thda_ml_ddpm.png')
    plt.close()

    plt.figure(figsize=(15, 10))
    plt.plot(time[::cinterval], rmse_slice1[::cinterval], label='ML Prediction RMSE', linewidth=2.5, linestyle='solid', color='black')
    plt.plot(time[::cinterval], rmse_slice2[::cinterval], label='PhysM Prediction RMSE', linewidth=2.5, linestyle='solid', color='gray')
    plt.plot(time[::cinterval], sprd_slice1[::cinterval], label='ML Prediction SPRD', linewidth=2.5, linestyle='dashed', color='black')
    plt.plot(time[::cinterval], sprd_slice2[::cinterval], label='PhysM Prediction SPRD', linewidth=2.5, linestyle='dashed', color='gray')
    #plt.fill_between(range(0,dataind), quantiles_lower, quantiles_upper, color='pink', alpha=0.3, label='95% Quantile')
    plt.xticks(size = 24)
    plt.yticks(size = 24)

# Customize plot if needed
    plt.ylim(0, 10)
    plt.xlabel('Time', fontsize=30)
    plt.ylabel('Value', fontsize=30)
    plt.title('RMSD/SPRD Series (Phys: Blue, ML: Red)', fontsize=30)
    plt.legend()
    #plt.savefig('NonormalizedDA_Ens16_phys_ml_x7_daresult_10points.png')
    #plt.savefig('rmse_sprd_ddpm_testda.png')
    #plt.savefig('nunormalized_rmse_sprd_ddpm_testda.png')
    #plt.savefig('onelayerml_rmse_sprd_ddpm_testda.png')
    plt.savefig('40grid_1tp_10thda_rmse_sprd_ddpm.png')
    plt.close()

    grid_points = 40
    #sample_slice = all_sample_np[500:, 0, :]
    sample_slice = all_target_np[:, 0, :]
    time_mean_sample_slice = np.mean(sample_slice, axis=0)
    state_values = time_mean_sample_slice 

# Perform Fast Fourier Transform (FFT) on spatial data
    window = np.hanning(len(state_values))
    state_values_windowed = state_values * window
    #fft_values = np.fft.fft(state_values_windowed)
    fft_values = np.fft.fft(state_values)

# Compute the power spectrum (magnitude squared of the FFT)
    power_spectrum = np.abs(fft_values) ** 2
    power_spectrum_dB = 10 * np.log10(power_spectrum)

# Compute the corresponding spatial frequencies
# Spatial frequency: inverse of the spatial grid spacing

    spatial_frequencies = np.fft.fftfreq(grid_points, d=1.0/grid_points)
    #print("CHECKFREQ=", spatial_frequencies)
    hz_frequencies = spatial_frequencies
    hz_frequencies = 2*np.pi*spatial_frequencies

# Only consider positive spatial frequencies (since the spectrum is symmetric)
    positive_hz_frequencies = hz_frequencies[:grid_points // 2 ]
    positive_power_spectrum_dB = power_spectrum_dB[:grid_points // 2 ]
    positive_power_spectrum = power_spectrum[:grid_points // 2 ]
    
# ML
    #sample_slice_ml = all_target_np[500:, 0, :]
    sample_slice_ml = all_sample_np[:, 0, :]
    time_mean_sample_slice = np.mean(sample_slice_ml, axis=0)
    state_values = time_mean_sample_slice 

    window = np.hanning(len(state_values))
    state_values_windowed = state_values * window
    #fft_values = np.fft.fft(state_values_windowed)
    fft_values = np.fft.fft(state_values)

    power_spectrum = np.abs(fft_values) ** 2
    power_spectrum_dB = 10 * np.log10(power_spectrum)

    spatial_frequencies = np.fft.fftfreq(grid_points, d=1.0/grid_points)
    hz_frequencies = spatial_frequencies
    hz_frequencies = 2*np.pi*spatial_frequencies

# Only consider positive spatial frequencies (since the spectrum is symmetric)
    positive_hz_frequencies_ml = hz_frequencies[:grid_points // 2 ]
    positive_power_spectrum_dB_ml = power_spectrum_dB[:grid_points // 2 ]
    positive_power_spectrum_ml = power_spectrum[:grid_points // 2 ]

# Physical
    sample_slice_phys = all_noda_np[:, 0, :]
    time_mean_sample_slice = np.mean(sample_slice_phys, axis=0)
    state_values = time_mean_sample_slice 

    window = np.hanning(len(state_values))
    state_values_windowed = state_values * window
    #fft_values = np.fft.fft(state_values_windowed)
    fft_values = np.fft.fft(state_values)

    power_spectrum = np.abs(fft_values) ** 2
    power_spectrum_dB = 10 * np.log10(power_spectrum)

    spatial_frequencies = np.fft.fftfreq(grid_points, d=1.0/grid_points)
    hz_frequencies = spatial_frequencies
    hz_frequencies = 2*np.pi*spatial_frequencies

# Only consider positive spatial frequencies (since the spectrum is symmetric)
    positive_hz_frequencies_phys = hz_frequencies[:grid_points // 2 ]
    positive_power_spectrum_dB_phys = power_spectrum_dB[:grid_points // 2 ]
    positive_power_spectrum_phys = power_spectrum[:grid_points // 2 ]

# Plot the power spectrum (log-log scale)

    plt.figure(figsize=(12, 8))
    plt.loglog(positive_hz_frequencies, positive_power_spectrum_dB, label='Truth', linewidth=3.0, linestyle='solid', color='black')
    plt.loglog(positive_hz_frequencies_ml, positive_power_spectrum_dB_ml, label='ML', linewidth=2.5, linestyle='solid', color='gray')
    plt.loglog(positive_hz_frequencies_phys, positive_power_spectrum_dB_phys, label='Phys', linewidth=2.5, linestyle='dashed', color='gray')
    #plt.loglog(positive_hz_frequencies, positive_power_spectrum, label='Truth', linewidth=3.0, linestyle='solid', color='black')
    #plt.loglog(positive_hz_frequencies_ml, positive_power_spectrum_ml, label='ML', linewidth=2.5, linestyle='solid', color='gray')
    #plt.loglog(positive_hz_frequencies_phys, positive_power_spectrum_phys, label='Phys', linewidth=2.5, linestyle='dashed', color='gray')
    #plt.plot(positive_hz_frequencies, positive_power_spectrum_dB, label='Power Spectrum', color='green')
    #plt.plot(positive_hz_frequencies_ml, positive_power_spectrum_dB, label='Power Spectrum', color='red')
    #plt.plot(positive_hz_frequencies_phys, positive_power_spectrum_dB, label='Power Spectrum', color='blue')

    plt.xlabel('Spatial wavenumber', fontsize=18)
    #plt.ylabel('Power (dB)', fontsize=18)
    plt.ylabel('Power', fontsize=18)
    plt.title('Spatial Power Spectrum wrt. Wavenumber (k)', fontsize=18)
    plt.grid(True)
    plt.xlim(5, 100)
    plt.xticks([5, 10, 50, 75, 100], labels=[r'0.5x$10^1$', r'1x$10^1$', r'5x$10^1$', r'7.5x$10^1$', r'$10^2$'])
    #plt.xlim(5, 20)
    #plt.xticks([1, 2, 3])
    plt.ylim(1, 30)
    #plt.yticks([10, 20, 30], labels=[r'1x$10^1$', r'2x$10^1$', r'3x$10^1$'])
    plt.legend()
    #plt.show()
    plt.savefig('spectrum.png')
    plt.close()
    exit()

    # Create the Hovmöller diagram
    xval = np.linspace(0, 2*np.pi, 40) 
    plt.figure(figsize=(12, 8))
    plt.contourf(time, xval, all_target_np, 20, cmap='viridis')  # Use contour plot for the color map

    # Add labels and title
    plt.xlabel('Time (t)', fontsize=18)
    plt.ylabel('Space (x)', fontsize=18)
    plt.title('Hovmöller Diagram', fontsize=18)

    # Add a colorbar to show the scale
    plt.colorbar(label='Data Value')

    # Show the plot
    plt.legend()
    #plt.savefig('NonormalizedDA_Ens16_phys_ml_x7_daresult_10points.png')
    plt.savefig('diagramhovm.png')
    plt.tight_layout()
    #plt.savefig('DDPM_RMSE_NormalizedDA_Ens32_phys_ml_x2_daresult_10points.png')
    # fig.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=1)


















