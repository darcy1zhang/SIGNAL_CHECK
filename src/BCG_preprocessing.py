#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:25:07 2022

@author: Yingjian
"""

import pytz
from datetime import datetime
from influxdb import InfluxDBClient
import operator
from scipy import signal, stats
from statsmodels.tsa.stattools import acf
import pywt
import numpy as np
from scipy.signal import hilbert, savgol_filter, periodogram, welch, resample
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, detrend
from scipy.interpolate import interp1d
import seaborn as sns
import re
# from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import subprocess

# import torch.nn as nn

# def segment_signal(signal, segment_length, overlap):
#     step = segment_length - overlap
#     num_segments = (len(signal) - segment_length) // step + 1
#     segments = np.array([signal[i * step : i * step + segment_length] for i in range(num_segments)])
#     return segments

# def ideal_filter(x, fs, low = None, high = None, method = 'bandpass'):
#     N = len(x)
#     fft_result = np.fft.fft(x)
#     fft_freqs = np.fft.fftfreq(N, 1/fs)
#     denoised_fft_res = np.copy(fft_result)
    
#     if method == 'bandpass':
#         selected_ids = np.argwhere((abs(fft_freqs) > high)).flatten()
#         denoised_fft_res[selected_ids] = 0

#         selected_ids = np.argwhere((abs(fft_freqs) < low)).flatten()
#         denoised_fft_res[selected_ids] = 0
#     elif method == 'highpass':
#         selected_ids = np.argwhere((abs(fft_freqs) <= high)).flatten()
#         denoised_fft_res[selected_ids] = 0
#     elif method == 'lowpass':
#          selected_ids = np.argwhere((abs(fft_freqs) > low)).flatten()
#          denoised_fft_res[selected_ids] = 0
        
#     ##### Perform IFFT
#     ifft_result = np.fft.ifft(denoised_fft_res)
#     filtered_x = ifft_result.real
    
#     return filtered_x
    
# def my_resample(x, fs, target_fs, target_len = None):
#     if not target_len:
#         duration = len(x)/fs
#         target_len = int(target_fs * duration)
#     if target_fs < fs:
#         N = len(x)
#         fft_result = np.fft.fft(x)
#         fft_freqs = np.fft.fftfreq(N, 1/fs)
#         valid_fs = target_fs/2
        
#         denoised_fft_res = np.copy(fft_result)
#         selected_ids = np.argwhere((abs(fft_freqs) > valid_fs)).flatten()
#         denoised_fft_res[selected_ids] = 0
        
#         ##### Perform IFFT
#         ifft_result = np.fft.ifft(denoised_fft_res)
#         x = ifft_result.real
#     resampled_signal = resample(x, target_len)
#     return resampled_signal

# class Regressor(nn.Module):
#     def __init__(self, final_out_channels, features_len, n_target):
#         super(Regressor, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear(features_len, features_len)
#         self.batchnorm = nn.BatchNorm1d(final_out_channels)
#         self.relu = nn.ReLU()
#         # self.regressor = nn.Linear(final_out_channels , n_target)
#         self.regressor = nn.Linear(features_len * final_out_channels, n_target)

#     def forward(self, x):
#         # x = self.linear(x)
#         # x = self.batchnorm(x)
#         # features = self.relu(x)
#         x = self.flatten(x)
#         features = x
#         x = self.flatten(x)
#         x = self.regressor(x)
#         return x, features

# class Wav2Vec2FeatureExtractor(nn.Module):
#     def __init__(self):
#         super(Wav2Vec2FeatureExtractor, self).__init__()
#         conv_layers = [
#             (512, 10, 5),
#             (512, 3, 2),
#             (512, 3, 2),
#             (512, 3, 2),
#             (512, 3, 2),
#             (512, 2, 2),
#             (512, 2, 2),
#         ]
#         self.conv_layers = nn.ModuleList()
#         for out_channels, kernel_size, stride in conv_layers:
#             self.conv_layers.append(
#                 nn.Sequential(
#                     nn.Conv1d(in_channels=1 if len(self.conv_layers) == 0 else 512,
#                               out_channels=out_channels,
#                               kernel_size=kernel_size,
#                               stride=stride),
#                     nn.GroupNorm(num_groups=16, num_channels=out_channels),
#                     nn.GELU()
#                 )
#             )
#         # Convolutional layer for relative positional embeddings
#         self.pos_conv = nn.Conv1d(512, 512, kernel_size=128, groups=16, padding=64)

#     def forward(self, x):
#         for conv in self.conv_layers:
#             x = conv(x)
#         x = self.pos_conv(x)
#         return x

# def breath_signal_extraction(x, fs, low_threshold, high_threshold, 
#                              dis = 1.5,
#                              extract_envelop = True, resample = False,
#                              target_fs = 100, Integral = False,
#                              show = False):
#     x = (x - np.mean(x))/np.std(x)
#     if not Integral:
#         raw_data = np.copy(x)
#     if show:
#         fig, ax = plt.subplots(5, 1, figsize=(16, 6))
#         ax[0].plot(x)
        
#     if extract_envelop:
#         # x = ideal_filter(x, fs, low = 20, high = None, 
#         #                   method = 'lowpass')
#         z= hilbert(x) #form the analytical signal
#         x = np.abs(z)
        
#         if show:
#             ax[1].plot(x)
#     if Integral:
#         x = x - np.mean(x)
#         x = np.cumsum(x)
        
#         if show:
#             ax[2].plot(x)
    
#     N = len(x)
#     fft_result = np.fft.fft(x)
#     fft_freqs = np.fft.fftfreq(N, 1/fs)
    
#     ##### trim energy below threshold
#     denoised_fft_res = np.copy(fft_result)
#     selected_ids = np.argwhere((abs(fft_freqs) > 1/low_threshold)).flatten()
#     # selected_ids = np.argwhere((fft_freqs > 1/low_threshold)).flatten()
#     denoised_fft_res[selected_ids] = 0
#     selected_ids = np.argwhere((abs(fft_freqs) < 1/high_threshold)).flatten()
#     # selected_ids = np.argwhere((fft_freqs < 1/high_threshold)).flatten()
#     denoised_fft_res[selected_ids] = 0
    
#     ##### Perform IFFT
#     ifft_result = np.fft.ifft(denoised_fft_res)
#     filtered_x = ifft_result.real
#     filtered_x = (filtered_x - np.mean(filtered_x))/np.std(filtered_x)
    
#     if show:
#         if not Integral:
#             ax[2].plot(np.cumsum(raw_data))
#         ax[3].plot(filtered_x)
#         rr_peak_ids, _ = signal.find_peaks(filtered_x, distance = int(dis * fs))
#         ax[3].scatter(rr_peak_ids, filtered_x[rr_peak_ids], c = 'r')
#         power_freqs, power_spec = periodogram(filtered_x, fs=fs, 
#                                               nfft=len(filtered_x))
#         ax[4].plot(power_freqs, power_spec)
    
#     if resample:
#         filtered_x = my_resample(filtered_x, fs, target_fs, target_len = None)
    
#     return filtered_x

# def fft_rr_cal(x, fs, low_threshold, high_threshold, dis = 1.5,
#                extract_envelop = True, resample = True, target_fs = 100, 
#                method = 'CountPeaks', Integral = False, show = False):
#     filtered_x = breath_signal_extraction(x, fs, low_threshold, high_threshold, dis,
#                                           extract_envelop, resample, 
#                                           target_fs, Integral, show)
    
#     # filtered_x2 = breath_signal_extraction(x, fs, 
#     #                                        low_threshold, 
#     #                                        high_threshold,
#     #                                        method = None)
    
#     if method == 'Spectral':
#         # ###################### Frequency domain #############################################
#         # Compute the power spectrum
#         ma_acf = acf(filtered_x, nlags = len(filtered_x))
#         ma_acf /= ma_acf[0]
#         N = len(ma_acf)
#         # fft_freqs, acf_fft_mag = welch(ma_acf, fs, window='hann', 
#         #                                 nperseg=int(10 * fs), noverlap=int(8 * fs), 
#         #                                 nfft=N, average='mean')
#         fft_freqs, acf_fft_mag = periodogram(filtered_x, fs=fs, nfft=N)
        
#         # acf_fft_mag = (acf_fft_power + acf_fft_mag)/2
        
#         selected_ids = np.argwhere((fft_freqs > 1/low_threshold)).flatten()
#         acf_fft_mag[selected_ids] = 0
#         selected_ids = np.argwhere((fft_freqs < 1/high_threshold)).flatten()
#         acf_fft_mag[selected_ids] = 0
        
#         fft_peak_ids, _ = signal.find_peaks(acf_fft_mag, height = 0)
#         # print(len(fft_peak_ids))
#         max_freq_id = fft_peak_ids[np.argmax(acf_fft_mag[fft_peak_ids])]
#         rr = fft_freqs[max_freq_id]
#         rr *= 60
        
#         ###################################################################
#     elif method == 'Time':
    
#     # # ######################## Time Domain #######################
#         # ma_sig = moving_window_integration(signal = filtered_x, fs = fs, 
#         #                           win_duration = 1)
#         # ma_sig2 = moving_window_integration(signal = filtered_x, fs = fs, 
#         #                           win_duration = 0.75)
#         # ma_sig3 = moving_window_integration(signal = filtered_x, fs = fs, 
#         #                           win_duration = 2)
        
#         # ma_acf1 = acf(ma_sig, nlags = len(ma_sig))
#         # ma_acf2 = acf(ma_sig2, nlags = len(ma_sig2))
#         # ma_acf3 = acf(ma_sig3, nlags = len(ma_sig3))
        
        
#         # ma_acf = ma_acf1[:] * ma_acf2[:] * ma_acf3[:]
#         # ma_acf =(ma_acf1[:] + ma_acf2[:] + ma_acf3[:])/3
#         ma_acf = acf(filtered_x, nlags = len(filtered_x))
#         ma_acf /= ma_acf[0]
        
#         # ma_acf2 = acf(filtered_x2, nlags = len(filtered_x2))
#         # ma_acf2 /= ma_acf2[0]
#         # ma_acf = ma_acf2
        
#         # ma_acf = ma_acf * ma_acf2
#         # ma_acf /= ma_acf[0]
        
#         rr_peak_ids, _ = signal.find_peaks(ma_acf[:len(ma_acf)], 
#                                     height = 0.1)
    
#         candidates_rr_ids = []
#         for rr_peak_id in rr_peak_ids:
#             if rr_peak_id > low_threshold * fs and rr_peak_id < high_threshold * fs:
#                 candidates_rr_ids.append(rr_peak_id)
                
#         if len(candidates_rr_ids) > 1:
#             rr = candidates_rr_ids[np.argmax(ma_acf[candidates_rr_ids])]
#             # rr = np.median(np.diff(candidates_rr_ids))
            
#         elif len(candidates_rr_ids) == 1:
#             rr = candidates_rr_ids[0]
#         else:
#             rr = -1
#         if rr != -1:
#             rr = 1/(rr/fs)
#             rr *= 60
#     # ############## ####################################################
#     elif method == 'CountPeaks':

#         if resample:
#             fs = target_fs
#         rr_peak_ids, _ = signal.find_peaks(filtered_x, height = 0, 
#                                             distance = int(dis * fs))
#         # rr_peak_ids, _ = signal.find_peaks(filtered_x, distance = int(2 * fs))
#         duration = len(filtered_x)/fs
#         rr = len(rr_peak_ids) * (60/duration)
        
#         # power_freqs, power_spec = periodogram(filtered_x, fs=fs, nfft=len(filtered_x))
#         # fft_peak_ids, _ = signal.find_peaks(power_spec, height = 0)
#         # max_id = fft_peak_ids[np.argmax(power_spec[fft_peak_ids])]
#         # if power_spec[max_id] > 0.1:
#         #     duration = len(filtered_x)/fs
#         #     rr = len(rr_peak_ids) * (60/duration)
#         #     # rr2 = len(rr_peak_ids2) * (60/duration)
#         #     # rr = (rr + rr2)/2
#         # else:
#         #     rr = -1
        
#     return rr

# def my_write_influx(influx, unit, table_name, data_name, data, 
#                     start_timestamp, time_stamp_list):
#     # print("epoch time:", timestamp)
#     max_size = 100
#     count = 0
#     cnt = 0
#     total = len(data)
#     prefix_post  = "curl -s -POST \'"+ influx['ip']+":8086/write?db="+influx['db']+"\' -u "+ influx['user']+":"+ influx['passw']+" --data-binary \' "
#     http_post = prefix_post
#     max_len = len(data)
#     for value in tqdm(data):
#         count += 1
#         cnt += 1
#         http_post += "\n" + table_name +",location=" + unit + " "
#         # st()
#         http_post += data_name + "=" + str(int(round(value))) + " " + str(int(start_timestamp*10e8))
#         if cnt < max_len:
#             start_timestamp +=  (time_stamp_list[cnt] - time_stamp_list[cnt-1])
#         # start_timestamp = time_stamp_list[cnt]
#         # print((time_stamp_list[count] - time_stamp_list[count-1])/1000)
#         if(count >= max_size):
#             http_post += "\'  &"
#             # print(http_post)
#             # print("Write to influx: ", table_name, data_name, count)
#             subprocess.call(http_post, shell=True)
#             total = total - count
#             count = 0
#             http_post = prefix_post
#     if count != 0:
#         http_post += "\'  &"
#         # print(http_post)
#         # print("Write to influx: ", table_name, data_name, count, data)
#         subprocess.call(http_post, shell=True)
        
# def local_time_epoch(time, zone):
#     local_tz = pytz.timezone(zone)
#     try:
#         localTime = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f")
#     except:
#         localTime = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")

#     local_dt = local_tz.localize(localTime, is_dst=None)
#     epoch = local_dt.timestamp()
#     return epoch

# def read_influx(influx, unit, table_name, data_name, start_timestamp, end_timestamp):
#     if influx['ip'] == '127.0.0.1' or influx['ip'] == 'localhost':
#         client = InfluxDBClient(influx['ip'], '8086', influx['user'], influx['passw'], influx['db'],  ssl=influx['ssl'])
#     else:
#         client = InfluxDBClient(influx['ip'].split('//')[1], '8086', influx['user'], influx['passw'], influx['db'],  ssl=influx['ssl'])

#     # client = InfluxDBClient(influx['ip'].split('//')[1], '8086', influx['user'], influx['passw'], influx['db'],  ssl=True)
#     query = 'SELECT "' + data_name + '" FROM "' + table_name + '" WHERE "location" = \''+unit+'\' AND time >= '+ str(int(start_timestamp*10e8))+' AND time < '+str(int(end_timestamp*10e8))

#     result = client.query(query)

#     points = list(result.get_points())
#     values =  list(map(operator.itemgetter(data_name), points))
#     times  =  list(map(operator.itemgetter('time'),  points))
    
#     data = values
#     return data, times

# def non_outlier_rate(x):
#     x = (x - np.mean(x))/np.std(x)
    
#     valid_ids = np.where((x < 2) & (x > -2))[0]
#     purity = len(valid_ids)/len(x)
#     return purity

# def epoch_time_local(epoch, zone):
#     local_tz = pytz.timezone(zone)
#     time = datetime.fromtimestamp(epoch).astimezone(local_tz).strftime("%Y-%m-%dT%H:%M:%S.%f")
#     return time

# def mac_to_int(mac):
#     res = re.match('^((?:(?:[0-9a-f]{2}):){5}[0-9a-f]{2})$', mac.lower())
#     if res is None:
#         raise ValueError('invalid mac address')
#     return int(res.group(0).replace(':', ''), 16)

# def make_confusion_matrix(cf_matrix, title):
#     group_names = ["True Positive","False Negitive","False Positive","True Negitive"]
#     group_counts = ["{0:0.0f}".format(value) for value in
#                     cf_matrix.flatten()]
#     # group_percentages = ["{0:.2%}".format(value) for value in
#     #                      cf_matrix.flatten()/np.sum(cf_matrix)]
#     labels = [f"{v1}\n{v2}" for v1, v2 in
#               zip(group_names,group_counts)]
#     labels = np.asarray(labels).reshape(2,2)
#     plt.figure(figsize = (8,6))
#     sns.set(font_scale=2)
#     sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues', 
#                 xticklabels = ['on bed', 'off bed'],
#                 yticklabels = ['on bed', 'off bed'])
#     plt.title(title)
#     plt.xlabel('true label')
#     plt.ylabel('prediction')

# def segmentation(x, win_len):
#     """
#     win_len should be odd
#     """
#     segments = []
#     start = 0
#     end = start + win_len
#     while end < len(x):
#         segments.append(x[start:end])
#         start += 1
#         end = start + win_len
#     return segments

# def M_ACF(x, start_lag, end_lag):
#     x1 = x[:len(x)//2+1]
#     x2 = x[len(x)//2:]
#     acf_x = []
#     for N in range(start_lag, end_lag):
#         r = (x1[-N:] @ x2[:N])/N
#         acf_x.append(r)
#     acf_x = np.array(acf_x)
#     acf_x = (acf_x - min(acf_x))/(max(acf_x) - min(acf_x))
#     acf_x = acf_x/sum(acf_x)
#     return acf_x

# def AMDF(x, start_lag, end_lag):
#     x1 = x[:len(x)//2+1]
#     x2 = x[len(x)//2:]
#     amdf = []
#     for N in range(start_lag, end_lag):
#         d = np.mean(abs(x1[-N:] - x2[:N]))
#         amdf.append(1/d)
#     amdf = np.array(amdf)
#     amdf = amdf/sum(amdf)
#     return amdf

# def MAP(x, start_lag, end_lag):
#     x1 = x[:len(x)//2+1]
#     x2 = x[len(x)//2:]
#     map_x = []
#     for N in range(start_lag, end_lag):
#         max_a = max(x1[-N:] + x2[:N])
#         map_x.append(max_a)
#     map_x = np.array(map_x)
#     map_x = map_x/sum(map_x)
#     return map_x

# def joint_estimator(x, start_lag, end_lag):
#     score_corr = M_ACF(x, start_lag, end_lag)
#     score_amdf = AMDF(x, start_lag, end_lag)
#     score_map = MAP(x, start_lag, end_lag)
    
#     joint_dis = score_corr * score_amdf * score_map
#     joint_dis = joint_dis/sum(joint_dis)
#     return joint_dis

# def KL(x1, x2):
#     u1, u2, sigma1, sigma2 = np.mean(x1), np.mean(x2), np.std(x1), np.std(x2)
#     kld = np.log(sigma2/sigma1) + (sigma1 ** 2 + (u1-u2) ** 2)/(2 * (sigma2 ** 2)) -0.5
#     return kld

# def if_valley(i, x):
#     if (x[i] - x[i-1]) < 0 and (x[i + 1] - x[i]) > 0:
#         return True

# def get_first_last_valley(x):
#     p1 = 1
#     p2 = -2
#     first_valley = 0
#     last_valley = -1
#     while(p1 < (len(x) + p2)):
#         if if_valley(p1, x) and first_valley == 0:
#             first_valley = p1
#         if if_valley(p2, x) and last_valley == -1:
#             last_valley = p2
#         if first_valley != 0 and last_valley != - 1:
#             return first_valley, last_valley + len(x)
#         p1 += 1
#         p2 -= 1
#     return None, None

# def find_similar_pattern(x, start_len, max_len):
#     p_mean = np.mean(x)
#     p_sigma = np.std(x)
    
#     start = 0
#     end = start_len + start
    
#     segment_ids = []
    
#     while(end < len(x)):
#         while(end < min(len(x), start + max_len)):
#             # if end == start + start_len:
#             #     q_sum = np.sum(x[start:end])
#             #     q_mean = q_sum/(end - start)
#             #     q_square_sum = sum(x[start:end] - q_mean) ** 2
#             #     q_std = np.sqrt(np.mean(q_square_sum))
#             # else:
#             #     q_sum += x[end - 1]
#             #     q_mean = q_sum/(end - start)
#             #     q_square_sum += sum(x[start:end] - q_mean) ** 2
#             #     q_std = np.sqrt(q_square_sum/(end - start))
                
#             q_mean = np.mean(x[start:end])
#             q_std = np.std(x[start:end])
                
                
#             kld = KL(u1 = p_mean, u2 = q_mean, 
#                      sigma1 = p_sigma, sigma2 = q_std)
            
#             if (end == start_len + start) or (kld < min_kld):
#                 min_kld = kld
#                 seg_id = end
#             end += 1
#         segment_ids.append(seg_id)
#         start = seg_id
#         end = start_len + start
#     return segment_ids

# def get_covariance_matrix(x, step, window_size):
#     N = (len(x) - window_size)//step + 1

#     segments = []
#     covariance = np.zeros((N, N))
#     start = 0
#     for i in range(N):
#         start = int(i * step)
#         end = start + window_size
#         segment = x[start:end]
#         if len(segment) > 1:
#             segment = (segment - np.mean(segment))/np.std(segment)
#         segments.append(segment)
#     segments = np.array(segments)
#     # print(segments.shape)
#     covariance1 = np.cov(segments)
#     covariance2 = np.cov(segments.T)
#     return covariance1, covariance2
        
# def peak_correction(x, peak_ids):
#     corrected_peak_ids = []
#     continue_flag = 0
#     diff = peak_ids[1:] - peak_ids[:-1]
#     for j in range(len(diff)):
#         if diff[j] < np.median(diff) - 12 * (100/31.25): 
#             if continue_flag:
#                 continue_flag = 0
#                 continue
            
#             if np.array(x)[peak_ids[j]] > np.array(x)[peak_ids[j + 1]]:
#                 max_id = peak_ids[j]
#                 continue_flag = 1
#             elif np.array(x)[peak_ids[j]] < np.array(x)[peak_ids[j + 1]]:
#                 max_id = peak_ids[j+1]
#             elif abs(diff[j-1] - np.median(diff)) < abs(diff[j+1] - np.median(diff)):
#                 max_id = peak_ids[j]
#                 continue_flag = 1
#             elif abs(diff[j-1] - np.median(diff)) > abs(diff[j+1] - np.median(diff)):
#                 max_id = peak_ids[j+1]
#             else:
#                 max_id = round((peak_ids[j+1] + peak_ids[j])/2)
#                 continue_flag = 1
            
#             if len(corrected_peak_ids) == 0 or max_id != corrected_peak_ids[-1]:
#                 corrected_peak_ids.append(max_id)
                
#         elif len(corrected_peak_ids) == 0 or peak_ids[j] != corrected_peak_ids[-1]:
#             if not continue_flag:
#                 corrected_peak_ids.append(peak_ids[j])
#             else:
#                 continue_flag = 0
#             if j == len(diff) - 1:
#                 corrected_peak_ids.append(peak_ids[j+1])
#     corrected_peak_ids = np.array(corrected_peak_ids)
#     return corrected_peak_ids

# def get_S_peak(segment, start_index):
#     valley_ids = signal.find_peaks(-segment)[0]
#     peak_ids = signal.find_peaks(segment)[0]
#     segment = np.array(segment)
#     i = 0
#     j = 0
#     max_diff = 0
#     if len(peak_ids) == 0:
#         return -1, -1
#     else:
#         for i in range(len(peak_ids)):
#             if i == 0 or segment[peak_ids[i]] > max_amp:
#                 max_amp = segment[peak_ids[i]]
#                 S_peak = start_index + peak_ids[i]
#     return S_peak, -1

# def get_SS_peak(segment, start_index):
#     valley_ids = signal.find_peaks(-segment)[0]
#     peak_ids = signal.find_peaks(segment)[0]
#     segment = np.array(segment)
#     if len(valley_ids) == 0:
#         return -1, -1
#     else:
#         for i in range(len(valley_ids)):
#             if i == 0 or segment[valley_ids[i]] < min_amp:
#                 min_amp = segment[valley_ids[i]]
#                 pre_SS_peak = start_index + valley_ids[i]
#                 if i < len(valley_ids) - 1:
#                     SS_peak = start_index + valley_ids[i + 1]
#                 else:
#                     SS_peak = -1
#         return pre_SS_peak, SS_peak


# def S_peak_detection(x, HR, target = 'S'):
#     segment_lines = []
#     last_seg = 0
#     n_heartbeat = round(HR//6)
#     n_win_len = round(len(x)/n_heartbeat)
#     most_win_len = round(len(x)/(n_heartbeat - 1))
#     S_peaks_id = []
#     if target == 'S':
#         peak_finder = get_S_peak
#     elif target == 'SS':
#         peak_finder = get_SS_peak
#     for i in range(n_heartbeat):
#         #update start id
#         start_index = i * n_win_len
#         if i == 0:
#             start_index = 0
#         else:
#             start_index = S_peak + int(3/4 * n_win_len)
#         #update end id
#         if i == 0:
#             end_index = start_index + most_win_len #- 12
#         else:
#             end_index = S_peak + most_win_len + 12
        
#         if end_index > len(x):
#             end_index = len(x)
#         #update segment
#         if i < n_heartbeat - 1:
#             segment = x[start_index:end_index]
#         elif end_index < len(x) - 1:
#             segment = x[start_index:end_index]
#             last_seg = 1
#         else:
#             segment = x[start_index:]
        
#         S_peak, SS_peak = peak_finder(segment, start_index)
#         if i == 0:
#             if S_peak != -1 and segment[S_peak] < np.mean(x):
#                 S_peak = np.argmax(segment)
#                 continue
#             while(S_peak == -1):
#                 end_index += 12
#                 segment = x[start_index:end_index]
#                 S_peak, SS_peak = peak_finder(segment, start_index)
                
#         elif S_peak == -1:
#             continue
#         segment_lines.append([start_index, end_index])
#         if target == 'S': 
#             S_peaks_id.append(S_peak)
#         elif target == 'SS':
#             if SS_peak == -1:
#                 start_index = S_peak - 2
#                 end_index = start_index + n_win_len
#                 segment = x[start_index:end_index]
#                 valley_ids = signal.find_peaks(-segment)[0]
#                 if len(valley_ids) > 1:
#                     for j in range(len(valley_ids)):
#                         if valley_ids[j] + start_index == S_peak:
#                             print(S_peak)
#                             SS_peak = valley_ids[j + 1]
#                 else:
#                     continue
#             S_peaks_id.append(SS_peak)
#     return S_peaks_id, segment_lines

# def get_J_peak(segment, start_index):
#     peaks_id = signal.find_peaks(segment)[0]
#     segment = np.array(segment)
#     # if len(peaks_id) == 0:
#     if len(peaks_id) > 2:
#         max_id = np.argmax(segment[peaks_id])
#         R_peak = peaks_id[max_id] + start_index
#     elif len(peaks_id) == 2:
#         if abs(segment[peaks_id[0]] - segment[peaks_id[1]]) >= 0.1:
#             max_id = np.argmax(segment[peaks_id])
#         else:
#             max_id = 0
#         R_peak = peaks_id[max_id] + start_index
#     elif len(peaks_id) == 1:
#         R_peak = peaks_id[0] + start_index
#     else:
#         return None
#     return R_peak
    
# def J_peak_detection(x, HR):
#     segment_lines = []
#     last_seg = 0
#     n_heartbeat = round(HR//6)
#     n_win_len = round(len(x)/n_heartbeat)
#     most_win_len = round(len(x)/(n_heartbeat - 1))
#     R_peaks_id = []
#     for i in range(n_heartbeat):
#         start_index = i * n_win_len
#         if i == 0:
#             start_index = 0
#             R_peak = start_index
#         else:
#             start_index = R_peak + int(3/4 * n_win_len)
        
#         if i == 0:
#             end_index = R_peak + n_win_len #- 12
#         else:
#             end_index = R_peak + most_win_len + 12
#         if i < n_heartbeat - 1:
#             segment = x[start_index:end_index]
#         elif end_index < len(x) - 1:
#             segment = x[start_index:end_index]
#             last_seg = 1
#         else:
#             segment = x[start_index:]
#         R_peak = get_J_peak(segment, start_index)
#         # print('=============================')
#         # print(start_index, end_index, R_peak)
#         if i == 0:
#             if R_peak and segment[R_peak] < np.mean(x):
#                 R_peak = np.argmax(segment)
#                 continue
#             while(not R_peak):
#                 end_index += 12
#                 segment = x[start_index:end_index]
#                 R_peak = get_J_peak(segment, start_index)
                
#         elif not R_peak:
#             middle_index = (start_index + end_index)//2
#             start_index = middle_index -  most_win_len//2
#             end_index = middle_index +  most_win_len//2
#             R_peak = get_J_peak(segment, start_index)
#             if not R_peak:
#                 R_peak = middle_index
#         # print(start_index, end_index, R_peak)
#         # elif not R_peak:
#         #     continue
#         segment_lines.append([start_index, end_index])
#         R_peaks_id.append(R_peak)

#     if last_seg:
#         start_index = R_peak + int(3/4 * n_win_len)
#         end_index = R_peak + most_win_len + 12
#         if start_index < len(x):
#             segment = x[start_index:]
#             R_peak = get_J_peak(segment, start_index)
#             if R_peak:
#                 R_peaks_id.append(R_peak)
#     return R_peaks_id, segment_lines

# def ACF(x, lag):
#     acf = []
#     mean_x = np.mean(x)
#     var = sum((x - mean_x) ** 2)
#     for i in range(lag):
#         if i == 0:
#             lag_x = x
#             original_x = x
#         else:
#             lag_x = x[i:]
#             original_x = x[:-i]
#         new_x = sum((lag_x - mean_x) * (original_x - mean_x))/var
#         new_x = new_x/len(lag_x)
#         acf.append(new_x)
#     return np.array(acf)

# def annotate_R_peaks(x, HR):
#     if HR/6 - (HR//6) > 0.5:
#         n_heartbeat = int(HR//6) + 1
#     else:
#         n_heartbeat = int(HR//6)
#     n_win_len = round(len(x)/n_heartbeat)
#     a_win_len = round(len(x)/(n_heartbeat-1))
#     side_win_len = (a_win_len - n_win_len)//2
#     R_peaks_id = []
#     for i in range(n_heartbeat):
#         start_index = i * n_win_len
#         if i > 0:
#             start_index -= side_win_len
        
#         if i < n_heartbeat - 1:
#             end_index = (i + 1) * n_win_len
#             end_index += side_win_len
#             segment = x[start_index:end_index]
#             # print(start_index, end_index)
#         else:
#             segment = x[start_index:]
#         peaks_id = signal.find_peaks(segment)[0]
#         segment = np.array(segment)
#         if len(peaks_id) == 0:
#             # R_peak = np.argwhere(segment == np.max(segment))[0][0] + start_index
#             continue
#         elif len(peaks_id) > 1:
#             max_id = np.argmax(segment[peaks_id])
#             R_peak = peaks_id[max_id] + start_index
#         else:
#             R_peak = peaks_id[0] + start_index
#         if x[R_peak] < (np.mean(x) + (0.5 * np.std(x))):
#             continue
#         R_peaks_id.append(R_peak)
#     return R_peaks_id

# def kurtosis(x):
#     x = x - np.mean(x)
#     a = 0
#     b = 0
#     a = sum(x ** 4)
#     b = sum(x ** 2)
#     # for i in range(len(x)):
#     #     a += x[i] ** 4
#     #     b += x[i] ** 2
#     a = a/len(x)
#     b = b/len(x)
#     k = a/(b**2)
#     return k

# #define high-order butterworth low-pass filter
# def low_pass_filter(data, Fs, low, order, causal = False):
#     b, a = signal.butter(order, low/(Fs * 0.5), 'low')
#     if causal:
#         filtered_data = signal.lfilter(b, a, data)
#     else:
#         filtered_data = signal.filtfilt(b, a, data)
#     # filtered_data = signal.filtfilt(b, a, data, method = 'gust')
#     return filtered_data

# def high_pass_filter(data, Fs, high, order, causal = False):
#     b, a = signal.butter(order, high/(Fs * 0.5), 'high')
#     if causal:
#         filtered_data = signal.lfilter(b, a, data)
#     else:
#         filtered_data = signal.filtfilt(b, a, data)
#     # filtered_data = signal.filtfilt(b, a, data, method = 'gust')
#     return filtered_data

def band_pass_filter(data, Fs, low, high, order, causal = False):
    b, a = signal.butter(order, [low/(Fs * 0.5), high/(Fs * 0.5)], 'bandpass')
    # perform band pass filter
    if causal:
        filtered_data = signal.lfilter(b, a, data)
    else:
        filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def wavelet_denoise(data, wave, Fs = None, n_decomposition = None):
    a = data
    w = wave
    ca = []
    cd = []
    rec_a = []
    rec_d = []
    freq_range = []
    for i in range(n_decomposition):
        if i == 0:
            freq_range.append(Fs/2)
        freq_range.append(Fs/2/(2** (i+1)))
        (a, d) = pywt.dwt(a, w)
        ca.append(a)
        cd.append(d)

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec = pywt.waverec(coeff_list, w)
        rec_a.append(rec)
        # ax3[i].plot(Fre, FFT_y1)
        # print(max_freq)

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))
    
    print(freq_range)
    return rec_a, rec_d

# def zero_cross_rate(x):
#     x = x - np.mean(x)
#     temp_x = x[1:] * x[:-1]
#     temp_ids = np.argwhere(temp_x < 0).flatten()
#     cnt = len(temp_ids)/len(x)
#     # for i in range(1,len(x)):
#     #     if x[i] * x[i-1] <= 0:
#     #         cnt += 1
#     return cnt

# def get_rr_evnvelope(x):
#     max_ids = argrelextrema(x, np.greater)[0]
#     peaks_val = x[max_ids]
    
#     min_ids = argrelextrema(x, np.less)[0]
#     valley_val = x[min_ids]
    
#     if len(max_ids) > 3 and len(min_ids) > 3:
#         if max_ids[0] < min_ids[0]:
#             valley_val = np.hstack((x[0], valley_val))
#             peaks_val = np.hstack((x[max_ids[0]], peaks_val))
    
#         elif max_ids[0] > min_ids[0]:
#             valley_val = np.hstack((x[max_ids[0]], valley_val))
#             peaks_val = np.hstack((x[0], peaks_val))
        
#         if max_ids[-1] > min_ids[-1]:
#             valley_val = np.hstack((valley_val, x[-1]))
#             peaks_val = np.hstack((peaks_val, x[max_ids[-1]]))
#         elif max_ids[-1] < min_ids[-1]:
#             valley_val = np.hstack((valley_val, x[max_ids[-1]]))
#             peaks_val = np.hstack((peaks_val, x[-1]))
        
#         max_ids = np.hstack((0, max_ids))
#         max_ids = np.hstack((max_ids, len(x) - 1))
        
#         min_ids = np.hstack((0, min_ids))
#         min_ids = np.hstack((min_ids, len(x) - 1))
    
#         upper_envelope = interp1d(max_ids, peaks_val, kind = 'cubic', bounds_error = False)(list(range(len(x)))) # (x[max_ids[0]], x[max_ids[-1]])
#         lower_envelope = interp1d(min_ids, valley_val, kind = 'cubic', bounds_error = False)(list(range(len(x)))) #(x[min_ids[0]], x[min_ids[-1]])
#         rr_envelope = (upper_envelope + lower_envelope)/2
#         return upper_envelope, lower_envelope, rr_envelope
#     else:
#         return [], [], []

# def rr_estimation(x, Fs, show = False):
#     x = (x - np.mean(x))/np.std(x)
#     cum_x = np.cumsum(x)
#     cum_detrended = detrend(cum_x)
#     rec_a, rec_d = wavelet_denoise(data = np.copy(cum_detrended), 
#                                    wave = 'db12', Fs = 100, 
#                                    n_decomposition = 9)
    
#     for j in range(1,4):
#         cut_len = (len(rec_d[-j]) - len(x))//2
#         if j == 1:
#             filtered_x = rec_d[-j][cut_len :- cut_len]
#         else:
#             filtered_x += rec_d[-j][cut_len :- cut_len]

#     N = 201
#     sg_filtered_x = savgol_filter(cum_detrended, 201, 3, mode='nearest')
#     sg_filtered_x = savgol_filter(sg_filtered_x, 101, 3, mode='nearest')
#     # MA_x = expectation(cum_detrended, 201)
#     MA_x = np.convolve(cum_detrended, np.ones(N)/N, mode='same')
#     MA_x = savgol_filter(MA_x, 201, 3, mode='nearest')
    
#     ##########################################
#     acf1 = acf(MA_x, nlags = len(MA_x))
    
#     acf1 = (acf1 - min(acf1))#/(max(acf1) - min(acf1))
#     acf1 = acf1/sum(acf1)
#     acf2 = acf(filtered_x, nlags = len(filtered_x))
#     acf2 = (acf2 - min(acf2))#/(max(acf2) - min(acf2))
#     acf2 = acf2/sum(acf2)
#     acf3 = acf(sg_filtered_x, nlags = len(sg_filtered_x))
#     acf3 = (acf3 - min(acf3))#/(max(acf3) - min(acf3))
#     acf3 = acf3/sum(acf3)

#     min_len = min(len(acf1), len(acf2), len(acf3))
#     acf_x = acf1[:min_len] * acf2[:min_len] * acf3[:min_len] #* amdf[:min_len]
#     acf_x /= sum(acf_x)
#     ##########################################
    
#     # acf_x = acf(MA_x, nlags = len(MA_x))
#     rr_peak_ids, _ = signal.find_peaks(acf_x[:len(acf_x)], 
#                                 height = np.mean(acf_x[:len(acf_x)]))
    
#     candidates_rr_ids = []
#     for rr_peak_id in rr_peak_ids:
#         if rr_peak_id > 220 and rr_peak_id < 750:
#             candidates_rr_ids.append(rr_peak_id)
    
#     if show:
#         fig, ax = plt.subplots(7, 1, figsize=(12,14), 
#                                sharex=True)
#         time = np.arange(0,len(x))/Fs
#         ax[0].plot(time, x)
        
#         ax[1].plot(time, cum_x)
        
#         ax[2].plot(time, cum_detrended)
        
#         time = np.arange(0,len(filtered_x))/Fs
#         ax[3].plot(time, filtered_x)
#         # ax[3].plot(time, acf2)
        
#         time = np.arange(0,len(MA_x))/Fs
#         ax[4].plot(time, MA_x)
#         # ax[4].plot(time, acf1)
        
#         time = np.arange(0,len(sg_filtered_x))/Fs
#         ax[5].plot(time, sg_filtered_x)
#         # ax[5].plot(time, acf3)
        
#         time = np.arange(0,len(acf_x))/Fs
#         ax[6].plot(time, acf_x)
#         ax[6].scatter(time[candidates_rr_ids], acf_x[candidates_rr_ids])
#         ax[6].set_ylabel('amplitude')
#         ax[6].xaxis.set_major_formatter('{x}s')
#         ax[6].set_xlabel('time (s)')
    
#     if len(candidates_rr_ids) > 1:
#         print(1111111111, len(candidates_rr_ids))
#         sorted_ids = sorted(range(len(acf_x[candidates_rr_ids])), 
#                             key=lambda k: acf_x[candidates_rr_ids][k])
#         height = acf_x[candidates_rr_ids][sorted_ids[-1]]
#         if (height - acf_x[candidates_rr_ids][sorted_ids[-2]] < 0.05):
#             intv= 0.5 * (candidates_rr_ids[sorted_ids[-2]] + candidates_rr_ids[sorted_ids[-1]])
#         else:
#             intv = candidates_rr_ids[np.argmax(acf_x[candidates_rr_ids])]
#         intv = candidates_rr_ids[np.argmax(acf_x[candidates_rr_ids])]
        
#     elif len(candidates_rr_ids) == 1:
#         intv = candidates_rr_ids[0]
#     else:
#         intv = -1
    
#     if intv != -1:
#         rr = 1/(intv/100)
#         rr *= 60
#         return rr
#     else:
#         return intv

# def BCG_IBI_estimation(BCG, Fs, hr):
#     BCG = (BCG - np.min(BCG))/(np.max(BCG) - np.min(BCG))
#     envelope = get_envelope(x = BCG, Fs = Fs, 
#                             low = 2, high = 10, m_wave = 'db12',
#                             denoised_method = 'bandpass')
#     envelope = envelope[:len(BCG)]
#     envelope = (envelope - np.min(envelope))/(np.max(envelope) - np.min(envelope))
#     acf_env = acf(envelope, nlags = len(envelope))
#     acf_env = band_pass_filter(data = acf_env, Fs = Fs, low = 0.6, high = 3, order = 3)
#     acf_env = (acf_env - np.min(acf_env))/(np.max(acf_env) - np.min(acf_env))
#     acf_env = acf_env[round(Fs * 2):-round(Fs * 2)]
#     J_peaks_id, segment_lines = J_peak_detection(acf_env, hr)
#     return J_peaks_id, segment_lines

# def false_J_peak_detection(x, J_peaks_index, median_diff):
#     time_diff = np.array(J_peaks_index[1:]) - np.array(J_peaks_index[:-1])
#     index_to_delete = []
#     for i in range(len(time_diff)):
#         if time_diff[i] < median_diff * 0.7:
#             diff1 = abs(J_peaks_index[i] - J_peaks_index[i-1] - median_diff)
#             diff2 = abs(J_peaks_index[i + 1] - J_peaks_index[i-1] - median_diff)
            
#             if diff1 > diff2:
#                 index_to_delete.append(i)
#             else:
#                 index_to_delete.append(i + 1)
            
#             # if x[J_peaks_index[i]] > x[J_peaks_index[i + 1]]:
#             #     index_to_delete.append(i+1)
#             # elif x[J_peaks_index[i]] < x[J_peaks_index[i + 1]]:
#             #     index_to_delete.append(i)
#             # else:
#             #     if diff1 > diff2:
#             #         index_to_delete.append(i)
#             #     else:
#             #         index_to_delete.append(i + 1)
    
#     if len(index_to_delete) > 0:
#         for i in range(len(index_to_delete)):
#             if i == 0:
#                 new_J_peaks_index = J_peaks_index[:index_to_delete[i]]
#             else:
#                 new_J_peaks_index += J_peaks_index[index_to_delete[i-1]+1 : index_to_delete[i]]
#         new_J_peaks_index += J_peaks_index[index_to_delete[-1] + 1:]
#         J_peaks_index = new_J_peaks_index
    
#     return J_peaks_index

# def missing_J_peak_detection(x, J_peaks_index, median_diff):
#     time_diff = np.array(J_peaks_index[1:]) - np.array(J_peaks_index[:-1])
#     new_J_peaks = []
#     for i in range(len(time_diff)):
#         if time_diff[i] - median_diff > median_diff//3:
#             segmentation = x[J_peaks_index[i] + median_diff//3 : J_peaks_index[i + 1] - median_diff//3]
#             new_index = peak_selection(segmentation, J_peaks_index[i] + median_diff//3, median_diff)
#             if new_index:
#                 new_J_peaks.append(new_index)
#     if len(new_J_peaks):
#         J_peaks_index = J_peaks_index + new_J_peaks
#         J_peaks_index = sorted(J_peaks_index)
#     return J_peaks_index

# def peak_selection(segmentation, start, median_diff):
#     max_ids = argrelextrema(segmentation, np.greater)[0]
#     if len(max_ids) > 0:
#         for index in max_ids:
#             if index == max_ids[0]:
#                 diff = abs(median_diff - index)
#                 J_peak_index = index + start
#             elif diff > abs(median_diff - index):
#                 diff = abs(median_diff - index)
#                 J_peak_index = index + start
#     else:
#         J_peak_index = None
#     return J_peak_index

# def J_peaks_detection(x, f):
#     window_len = round(1/f * 100)
#     start = 0
#     J_peaks_index = []
    
#     #get J peaks
#     while start < len(x):
#         end = start + window_len
#         if start > 0:
#             segmentation = x[start -1 : end]
#         else:
#             segmentation = x[start : end]
#         # J_peak_index = np.argmax(segmentation) + start
#         max_ids = argrelextrema(segmentation, np.greater)[0]
#         for index in max_ids:
#             if index == max_ids[0]:
#                 max_val = segmentation[index]
#                 J_peak_index = index + start
#             elif max_val < segmentation[index]:
#                 max_val = segmentation[index]
#                 J_peak_index = index + start
        
#         if len(max_ids) > 0 and x[J_peak_index] > 0:
#             if len(J_peaks_index) == 0 or J_peak_index != J_peaks_index[-1]:
#                 J_peaks_index.append(J_peak_index)
#         start = start + window_len//2
    
#     #correct J peak false detection
#     # median_diff = window_len
#     # median_diff = np.median(time_diff)
#     ##false J peak detection
#     # J_peaks_index = false_J_peak_detection(x, J_peaks_index, median_diff)
#     # # ## missing J peak detection
#     # J_peaks_index = missing_J_peak_detection(x, J_peaks_index, median_diff)
#     # ##false J peak detection
#     # J_peaks_index = false_J_peak_detection(x, J_peaks_index, median_diff)
#     # print(J_peaks_index)
    
#     return J_peaks_index

# def bed_status_detection(x):
#     x = (x - np.mean(x))/np.std(x)
#     zcr = zero_cross_rate(x)
    
#     rec_a, rec_d = wavelet_denoise(data = x, wave = 'db4', Fs = 100, n_decomposition = 6)
#     min_len = min(len(rec_d[-4]), len(rec_d[-3])) #len(rec_d[-5])
#     denoised_sig = rec_d[-4][:min_len] + rec_d[-3][:min_len]
    
#     denoise_zcr = zero_cross_rate(denoised_sig)

#     if denoise_zcr > 165:
#         if zcr < 180:
#             bed_status = 1
#         else:
#             bed_status = -1
#     elif denoise_zcr < 150:
#         bed_status = 1
#     elif zcr > 340:
#         bed_status = -1
#     else:
#         bed_status = 1
#     return bed_status, denoise_zcr, zcr

# def get_envelope(x, Fs, low, high, m_wave = 'db12',
#                  denoised_method = 'DWT'):
#     x = (x - np.mean(x))/np.std(x)
#     if denoised_method == 'DWT':
#         n_decomposition, max_n, min_n = freq_com_select(Fs = 100, low = 0.8, high = 12)
#         rec_a, rec_d = wavelet_denoise(data = x, wave = m_wave, Fs = Fs, 
#                                        n_decomposition = n_decomposition)
#         min_len = len(rec_d[max_n])
#         for n in range(max_n, min_n):
#             if n == max_n:
#                 denoised_sig = rec_d[n][:min_len]
#             else:
#                 denoised_sig += rec_d[n][:min_len]
#         # min_len = min(len(rec_d[-1]), len(rec_d[-2]), len(rec_d[-3]), len(rec_d[-4])) #len(rec_a[-1]) len(rec_d[-5])
#         # denoised_sig = rec_d[-1][:min_len] + rec_d[-2][:min_len] + rec_d[-4][:min_len] \
#         #                 + rec_d[-3][:min_len]
#         cut_len = (len(denoised_sig) - len(x))//2
#         denoised_sig = denoised_sig[cut_len:-cut_len]
        
#     elif denoised_method == 'bandpass':
#         denoised_sig = band_pass_filter(data = x, Fs = 100, 
#                                         low = low, high = high, order = 5)
#     z= hilbert(denoised_sig) #form the analytical signal
#     envelope = np.abs(z)
    
#     smoothed_envelope = savgol_filter(envelope, int(0.41 * Fs), 3, mode='nearest')
#     smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope))/np.std(smoothed_envelope)
    
#     trend = savgol_filter(smoothed_envelope, int(2.01 * Fs), 3, mode='nearest')
#     smoothed_envelope = smoothed_envelope - trend
#     # # smoothed_envelope = detrend(smoothed_envelope)
#     # smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope))/np.std(smoothed_envelope)
   
#     return smoothed_envelope

# def energy_envelope(x, N):
#     energy = x ** 2
#     envelope = np.convolve(energy, np.ones(N), mode='valid')
#     return envelope

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

# def get_peaks_valleys(x):
#     smoothed_envelope =x
#     peaks = []
#     valleys = []
#     features = []
#     features_x = []
#     features_y = []
#     for i in range(1, len(smoothed_envelope)-1):
#         pre_diff = smoothed_envelope[i] - smoothed_envelope[i-1]
#         post_diff = smoothed_envelope[i + 1] - smoothed_envelope[i]
#         if pre_diff > 0 and post_diff < 0 and smoothed_envelope[i] > 0:
#             if len(peaks) == 0:
#                 peaks.append(i)
#             elif len(peaks) == len(valleys):
#                 peaks.append(i)
#             elif smoothed_envelope[peaks[-1]] < smoothed_envelope[i]:
#                 peaks.pop()
#                 peaks.append(i)
#         elif pre_diff < 0 and post_diff > 0 and smoothed_envelope[i] < 0:
#             if len(valleys) == 0 and len(peaks) == 1:
#                 valleys.append(i)
#             elif len(peaks) - len(valleys) == 1:
#                 valleys.append(i)
#             elif len(valleys) > 0 and smoothed_envelope[valleys[-1]] > smoothed_envelope[i]:
#                 valleys.pop()
#                 valleys.append(i)

#         if len(peaks) > 1:
#             features.append([smoothed_envelope[peaks[-2]], valleys[-1] - peaks[-2], 
#                             smoothed_envelope[valleys[-1]], peaks[-1] - valleys[-1]])
#             features_x.append(smoothed_envelope[peaks[-2]] - smoothed_envelope[valleys[-1]])
#             features_y.append(peaks[-1] - valleys[-1] - valleys[-1] + peaks[-2])
#     return peaks, valleys, features_x, features_y

# def freq_com_select(Fs, low, high):
#     n = 0
#     valid_freq = Fs/2
#     temp_f = valid_freq
#     min_diff_high = abs(temp_f - high)
#     min_diff_low = abs(temp_f - low)
    
#     while(temp_f > low):
#         temp_f = temp_f / 2
#         n += 1
#         diff_high = abs(temp_f - high)
#         diff_low = abs(temp_f - low)
#         if diff_high < min_diff_high:
#             max_n = n
#             min_diff_high = diff_high
#         if diff_low < min_diff_low:
#             min_n = n
#             min_diff_low = diff_low
#     return n, max_n, min_n

# def signal_quality_assessment(x, n_decomposition, Fs, n_lag,
#                               high, low,
#                               denoised_method = 'DWT', acf_window = False,
#                               show = False):
#     x = (x - np.mean(x))#/np.std(x)
#     if denoised_method == 'DWT':
#         n_decomposition, max_n, min_n = freq_com_select(Fs = 100, low = 0.8, high = 10)
#         rec_a, rec_d = wavelet_denoise(data = x, wave = 'db12', 
#                                        Fs = Fs, n_decomposition = n_decomposition)
#         min_len = len(rec_d[max_n])
#         for n in range(max_n, min_n):
#             if n == max_n:
#                 denoised_sig = rec_d[n][:min_len]
#             else:
#                 denoised_sig += rec_d[n][:min_len]
#         cut_len = (len(denoised_sig) - len(x))//2
#         denoised_sig = denoised_sig[cut_len:-cut_len]
#         # rec_a, rec_d = wavelet_denoise(data = x, wave = 'db12', Fs = Fs, n_decomposition = n_decomposition)
#         # min_len = min(len(rec_d[-1]), len(rec_d[-2]), len(rec_d[-3]), len(rec_d[-4])) #len(rec_a[-1]) len(rec_d[-5])
#         # denoised_sig = rec_d[-1][:min_len] + rec_d[-2][:min_len] + rec_d[-4][:min_len] \
#         #                 + rec_d[-3][:min_len]
#         # cut_len = (len(denoised_sig) - len(x))//2
#         # denoised_sig = denoised_sig[cut_len:-cut_len]
#         # for i in range(1,5):
#         #     cut_len = (len(rec_d[-i]) - len(x))//2
#         #     if i == 1:
#         #         denoised_sig = rec_d[-i][cut_len :- cut_len]
#         #         # denoised_sig = rec_d[-i]
#         #     else:
#         #         # min_len = min(len(denoised_sig), len(rec_d[-i]))
#         #         # denoised_sig[:min_len] += rec_d[-i][:min_len]
#         #         if cut_len != 0:
#         #             denoised_sig += rec_d[-i][cut_len :- cut_len] #+ rec_a[-1][:min_len]
#         #         else:
#         #             denoised_sig += rec_d[-i]
        
#     elif denoised_method == 'bandpass':
#         # denoised_sig = band_pass_filter(data = x, Fs = 100, low = 0.8, high = 15, order = 5)
#         denoised_sig = band_pass_filter(data = x, Fs = 100, 
#                                         low = low, high = high, order = 5)
#     index = 0
#     window_size = int(Fs)
#     z= hilbert(denoised_sig) #form the analytical signal
#     envelope = np.abs(z)
    
#     smoothed_envelope = savgol_filter(envelope, round(0.41 * Fs), 3, mode='nearest')
#     smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope))/np.std(smoothed_envelope)
    
#     trend = savgol_filter(smoothed_envelope, round(2.01 * Fs), 3, mode='nearest')
#     smoothed_envelope1 = smoothed_envelope - trend
    
#     smoothed_envelope0 = band_pass_filter(data = envelope, Fs = 100, 
#                                           low = 0.6, high = 3, order = 5)
#     # smoothed_envelope1 = detrend(smoothed_envelope)
#     # z= hilbert(smoothed_envelope) #form the analytical signal
#     # envelope = np.abs(z)
#     smoothed_envelope2 = smoothed_envelope/envelope
#     smoothed_envelope1 = (smoothed_envelope1 - np.mean(smoothed_envelope1))/np.std(smoothed_envelope1)
#     smoothed_envelope2 = (smoothed_envelope2 - np.mean(smoothed_envelope2))/np.std(smoothed_envelope2)
#     # pdf_x = stats.norm.pdf(smoothed_envelope)
#     # log_pdf = stats.norm.logpdf(smoothed_envelope)
#     # smoothed_envelope = pdf_x * log_pdf
#     # smoothed_envelope = low_pass_filter(data = smoothed_envelope, Fs = 100, low = 3, order = 5)
#     # peaks,_ = signal.find_peaks(smoothed_envelope, height = 0)
#     # valleys,_ = signal.find_peaks(-smoothed_envelope, height = 0)
#     # smoothed_envelope2 = band_pass_filter(data = smoothed_envelope1, Fs = 100, 
#     #                                 low = 0.8, high = 5, order = 5)
#     if acf_window:
#         n_per_seg = int(2.5 * Fs)
#         n_step = int(0.5 * Fs)
#         start = 0
#         end = start + n_per_seg
#         acf_x = np.ones(n_per_seg)
#         n_seg = 0
#         while(end < len(smoothed_envelope)):
#             seg = smoothed_envelope[start:end]
#             n_seg += 1
#             temp = acf(seg, nlags = len(seg))
#             temp = (temp - min(temp))/(max(temp) - min(temp))
#             acf_x *= temp
#             start += n_step
#             end = start + n_per_seg
#         # acf_x = acf_x/n_seg
#     else:
#         acf_x1 = acf(smoothed_envelope1, nlags = n_lag)
#         acf_x1_1 = acf_x1/acf_x1[0]
#         acf_x1 = (acf_x1 - min(acf_x1))/(max(acf_x1) - min(acf_x1))
#         acf_x1 = acf_x1 / sum(acf_x1)
        
#         acf_x2 = acf(acf_x1, nlags = n_lag)
#         acf_x2 = (acf_x2 - min(acf_x2))/(max(acf_x2) - min(acf_x2))
#         acf_x2 = acf_x2 / sum(acf_x2)
        
#         acf_x = acf_x1 * acf_x2
#         acf_x = acf_x/sum(acf_x)
#         acf_x = acf_x1
        
#     acf_x = acf_x/acf_x[0]
#     if acf_window:
#         acf_x[:n_step] = 0
#         acf_x = acf_x/max(acf_x)
    
#     # for i in range(2):
#     #     acf_x = acf(acf_x, nlags = n_lag * 2)
#     #     acf_x = acf_x/acf_x[0]
    
#     # acf_x = acf_x - np.mean(acf_x)
#     nfft = next_power_of_2(x = len(x) * 2)
#     f, Pxx_den = periodogram(acf_x1_1, fs = Fs, nfft = nfft)
    
#     sig_means = []
#     index = 0
#     frequency = f[np.argmax(Pxx_den)]
#     power = max(Pxx_den)
#     if show:
#         fig, ax = plt.subplots(5, 1, figsize=(12,20), sharex = True)
#         time = np.arange(0, len(x))/Fs
#         ax[0].plot(time, x, label = 'raw data')
#         # ax[0].set_xlabel('time (s)')
#         # ax[0].set_ylabel('amplitude')
#         ax[0].xaxis.set_major_formatter('{x}s')
        
#         time = np.arange(0, len(denoised_sig))/Fs
#         ax[1].plot(time, denoised_sig, label = 'wavelet denoised data')
#         # ax[1].set_xlabel('time (s)')
#         # ax[1].set_ylabel('amplitude')
#         ax[1].xaxis.set_major_formatter('{x}s')
        
#         ax[2].plot(time, envelope, label = 'envelope extraction by Hilbert transform')
#         # ax[2].set_xlabel('time (s)')
#         # ax[2].set_ylabel('amplitude')
#         ax[2].xaxis.set_major_formatter('{x}s')
        
#         ax[3].plot(time, smoothed_envelope1, label = 'smoothed envelope')
#         # ax[3].set_xlabel('time (s)')
#         # ax[3].set_ylabel('amplitude')
#         ax[3].xaxis.set_major_formatter('{x}s')
#         # ax[3].scatter(peaks, smoothed_envelope[peaks], c = 'r')
#         # ax[3].scatter(valleys, smoothed_envelope[valleys], c = 'g')
#         time = np.arange(0, len(acf_x))/Fs
#         ax[4].plot(time, acf_x, label = 'ACF of smoothed envelope')
#         # ax[4].set_xlabel('time (s)')
#         # ax[4].set_ylabel('amplitude')
#         ax[4].xaxis.set_major_formatter('{x}s')
        
#         # ax[5].plot(f, Pxx_den, label = 'spectrum of ACF')
#         # ax[5].set_xlabel('freuqency (Hz)')
#         # # ax[5].set_ylabel('amplitude')
#         # ax[5].xaxis.set_major_formatter('{x}Hz')
#         # ax[5].plot(acf_x2, label = 'ACF of ACF')
#         # ax[5].scatter(features_x, features_y)
#         fig.supylabel("amplitude", fontsize = 20)
#         fig.supxlabel("time(s)", fontsize = 20)
#         for i in range(len(ax)):
#             ax[i].legend()
#             ax[i].xaxis.set_tick_params(labelsize=20)
#             ax[i].yaxis.set_tick_params(labelsize=20)

#     while (index + window_size < len(acf_x)):
#         sig_means.append(np.mean(acf_x[index:index + window_size]))
#         index = index + window_size
        
#     peak_ids, _ = signal.find_peaks(acf_x, height = np.mean(acf_x))
#     r = max(acf_x[peak_ids])
#     # if power > 0.7 and 0.6< frequency < 2.5:
#     # if np.std(sig_means) < 0.1 and 0.6< frequency < 3 \
#     #     and power > 0.1 and r > 0.8:
#     # if np.std(sig_means) < 0.1 and 0.6< frequency < 3 \
#     #     and (power > 0.25 or r > 0.6):
#     if np.std(sig_means) < 0.1 and power > 0.1 and r > 0.6:# and 0.6< frequency < 3:
#         time_diff = peak_ids[1:] - peak_ids[:-1]
        
#         # check if the peaks is periodic
#         cadidates = []
#         for peak_id in peak_ids:
#             if (peak_id > int(0.33 * Fs) and peak_id < int(1.5 * Fs)):
#                 cadidates.append(peak_id)
        
#         # print(acf_x[cadidates])
#         if len(cadidates) > 1:
#             sorted_ids = sorted(range(len(acf_x[cadidates])), key=lambda k: acf_x[cadidates][k])
#             height = acf_x[cadidates][sorted_ids[-1]]
#             if (height - acf_x[cadidates][sorted_ids[-2]] < 0.05) and \
#                 (peak_ids[0] in cadidates):
#                 median_hr = np.median(time_diff)
#             else:
#                 median_hr = cadidates[np.argmax(acf_x[cadidates])]
#         elif len(cadidates) == 1:
#             median_hr = cadidates[0]
#         else:
#             for peak_id in peak_ids:
#                 if (peak_id > int(1.25 * Fs) and peak_id < int(1.51 * Fs)):
#                     cadidates.append(peak_id)
#                     median_hr = cadidates[np.argmax(acf_x[cadidates])]
#             if len(cadidates) == 0:
#                 median_hr = np.median(time_diff)
#         if len(cadidates) == 0:
#             res = ['bad data', np.std(sig_means), frequency, power, []]
#             return res
#         median_hr = cadidates[np.argmax(acf_x[cadidates])] #+ int(0.5 * Fs)
#         # median_hr = np.median(time_diff)
#         frequency = 1/(median_hr/Fs)
#         res = ['good data', np.std(sig_means), frequency, r, acf_x]
#         if show:
#             ax[4].scatter(time[peak_ids], acf_x[peak_ids])
#             fig.suptitle('good data')
#     else:
#         res = ['bad data', np.std(sig_means), frequency, r, []]
#         if show:
#             fig.suptitle('bad data')
#     return res

# def wavelet_decomposition(data, wave, Fs = None, n_decomposition = None):
#     a = data
#     w = wave
#     ca = []
#     cd = []
#     rec_a = []
#     rec_d = []
#     freq_range = []
#     for i in range(n_decomposition):
#         if i == 0:
#             freq_range.append(Fs/2)
#         freq_range.append(Fs/2/(2** (i+1)))
#         (a, d) = pywt.dwt(a, w)
#         ca.append(a)
#         cd.append(d)

#     for i, coeff in enumerate(ca):
#         coeff_list = [coeff, None] + [None] * i
#         rec = pywt.waverec(coeff_list, w)
#         rec_a.append(rec)

#     for i, coeff in enumerate(cd):
#         coeff_list = [None, coeff] + [None] * i
#         # print(coeff_list)
#         rec_d.append(pywt.waverec(coeff_list, w))

#     return rec_a, rec_d


# def wavelet_reconstruction(x, fs, low, high):
#     n_decomposition, max_n, min_n = freq_com_select(Fs = fs, low = low, high = high)
#     rec_a, rec_d = wavelet_decomposition(data = x, wave = 'db12', Fs = fs, n_decomposition = n_decomposition)
#     min_len = len(rec_d[max_n])
#     for n in range(max_n, min_n):
#         if n == max_n:
#             denoised_sig = rec_d[n][:min_len]
#         else:
#             denoised_sig += rec_d[n][:min_len]
#     cut_len = (len(denoised_sig) - len(x))//2
#     denoised_sig = denoised_sig[cut_len:-cut_len]
#     return denoised_sig

def moving_window_integration(signal, fs, win_duration):

    # Initialize result and window size for integration
    result = signal.copy()
    win_size = round(win_duration * fs)
    sum = 0

    # Calculate the sum for the first N terms
    for j in range(win_size):
      sum += signal[j]/win_size
      result[j] = sum
    
    # Apply the moving window integration using the equation given
    for index in range(win_size,len(signal)):  
      sum += signal[index]/win_size
      sum -= signal[index-win_size]/win_size
      result[index] = sum

    return result

def periodogram_acf_fusion(Pxx_den, f, acf_x, Fs,
                           target_f_range, target_acf_range):
    power_valid_ids = np.where((f > target_f_range[0]) & (f < target_f_range[1]))[0]
    power_spectrum_pdf = Pxx_den[power_valid_ids]/sum(Pxx_den[power_valid_ids])
    
    new_f = Fs/np.arange(target_acf_range[1], target_acf_range[0], -1)
    
    valid_acf = acf_x[np.arange(target_acf_range[1], target_acf_range[0], -1)]
    valid_acf = (valid_acf - np.min(valid_acf))/(np.max(valid_acf) - np.min(valid_acf))
    acf_pdf = valid_acf/sum(valid_acf)

    cs = CubicSpline(f[power_valid_ids], power_spectrum_pdf)
    new_power_spectrum_pdf = cs(new_f)
    new_power_spectrum_pdf = new_power_spectrum_pdf/np.sum(new_power_spectrum_pdf)
    # print(len(power_spectrum_pdf), len(acf_pdf))
    fusion_pdf = acf_pdf * new_power_spectrum_pdf
    fusion_pdf = fusion_pdf/sum(fusion_pdf)
    
    return fusion_pdf, acf_pdf, new_power_spectrum_pdf, new_f

def estimate_vital_fusion(fusion_pdf, acf_pdf, new_f, target = 'hr'):
    max_fusion_id = np.argmax(fusion_pdf)
    fused_vital = 60 * new_f[max_fusion_id]
    peak_ids, _ = signal.find_peaks(acf_pdf, height = 0.1)
    
    if len(peak_ids) == 0:
        max_id = np.argmax(acf_pdf)
    else:
        max_id = peak_ids[np.argmax(acf_pdf[peak_ids])]
    
    acf_vital = 60 * new_f[max_id]
    
    if target == 'hr':
        ## searh closest peak compare to fusion pdf
        if len(peak_ids) != 0:
            closest_id = peak_ids[np.argmin(abs(peak_ids - max_fusion_id))]
        else:
            closest_id = max_fusion_id
        r1 = acf_pdf[max_id]
        r2 = acf_pdf[closest_id]
        ## identify real peak location by gradient
        acf_pdf_d = np.diff(acf_pdf)
        acf_pdf_d = (acf_pdf_d - np.min(acf_pdf_d))/(np.max(acf_pdf_d) - np.min(acf_pdf_d))
        acf_pdf_d = acf_pdf_d/sum(acf_pdf_d)
        peak_ids2, _ = signal.find_peaks(acf_pdf_d, height = 0)
        intv_vital = np.median(60/((peak_ids[1:] - peak_ids[:-1])/100))
        if len(peak_ids2) != 0:
            largest_gd_id = peak_ids2[np.argmax(acf_pdf_d[peak_ids2])]
        else:
            largest_gd_id = max_id
        if (r2 != 0 and r1/r2 > 1.3) \
            or (abs(max_id - largest_gd_id) <= abs(closest_id - largest_gd_id) \
                and len(peak_ids2) < 3) \
            or (2.1 > fused_vital/acf_vital > 1.9 and abs(intv_vital - fused_vital) < 8):
            vital = acf_vital
        else:
            vital = fused_vital
        # vital = fused_vital
        if abs(intv_vital - vital) <= 8:
            vital = intv_vital
    else:
        max_fusion_peak_ids, _ = signal.find_peaks(fusion_pdf[:len(fusion_pdf)], 
                                    height = 0.02)
        if len(max_fusion_peak_ids) > 0:
            max_val_id = np.argmax(fusion_pdf[max_fusion_peak_ids])
            vital = 60 * new_f[max_fusion_peak_ids[max_val_id]]
        else:
            vital = -1
    
    return vital

def rr_estimation_v2(x, low, high, Fs, quality_threshold = 0.1):
    low_threshold = int(2 * Fs)
    high_threshold = int(7.5 * Fs)
    
    x = (x - np.mean(x))/np.std(x)
    filtered_x = band_pass_filter(data = x, Fs = Fs, 
                                  low = low, high = high, order = 3, 
                                  causal = True)
    
    ma_sig = moving_window_integration(signal = filtered_x, fs = 100, 
                                       win_duration = 1)
    ma_sig2 = moving_window_integration(signal = filtered_x, fs = 100, 
                                        win_duration = 0.75)
    ma_sig3 = moving_window_integration(signal = filtered_x, fs = 100, 
                                        win_duration = 2)
    
    ma_acf1 = acf(ma_sig, nlags = len(ma_sig))
    ma_acf2 = acf(ma_sig2, nlags = len(ma_sig2))
    ma_acf3 = acf(ma_sig3, nlags = len(ma_sig3))
    
    ma_acf = ma_acf1[:] * ma_acf2[:] * ma_acf3[:]
    ma_acf /= ma_acf[0]
    
    rr_peak_ids, _ = signal.find_peaks(ma_acf[:len(ma_acf)], 
                                height = quality_threshold)
    candidates_rr_ids = []
    for rr_peak_id in rr_peak_ids:
        if rr_peak_id > low_threshold and rr_peak_id < high_threshold:
            candidates_rr_ids.append(rr_peak_id)
            
    if len(candidates_rr_ids) > 1:
        rr = candidates_rr_ids[np.argmax(ma_acf[candidates_rr_ids])]
        
    elif len(candidates_rr_ids) == 1:
        rr = candidates_rr_ids[0]
    else:
        rr = -1

    if rr != -1:
        rr = 1/(rr/Fs)
        rr *= 60
    return rr

# def signal_quality_assessment_v4(x, Fs, n_lag, low, high, 
#                                  acf_low, acf_high,
#                                  f_low, f_high, fusion = 0, diff3 = 1,
#                                  smooth_filter = 'sg', denoised_method = 'DWT', 
#                                  show = False):
#     x = (x - np.mean(x))/np.std(x)
#     if denoised_method == 'DWT':
#         denoised_sig = wavelet_reconstruction(x = x, fs = Fs, 
#                                               low = low, high = high)

#     elif denoised_method == 'bandpass':
#         denoised_sig = band_pass_filter(data=x, Fs=Fs, 
#                                         low=low, high=high, order=3,
#                                         causal = True)
#         if diff3:
#             # denoised_sig = np.diff(np.diff(denoised_sig))
#             denoised_sig = np.diff(denoised_sig)
#     index = 0
#     window_size = int(Fs)
#     z= hilbert(denoised_sig) #form the analytical signal
#     envelope = np.abs(z)
    
#     if smooth_filter == 'sg': 
#         sg_win_len = round(0.41 * Fs)
#         if sg_win_len%2 == 0:
#             sg_win_len -= 1
#         smoothed_envelope = savgol_filter(envelope, sg_win_len, 3, mode='nearest')
#         smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope))/np.std(smoothed_envelope)
        
#         sg_win_len = round(2.01 * Fs)
#         if sg_win_len%2 == 0:
#             sg_win_len -= 1
#         trend = savgol_filter(smoothed_envelope, sg_win_len, 3, mode='nearest')
#         smoothed_envelope = smoothed_envelope - trend
#     elif smooth_filter == 'bandpass': 
#         # envelope, trend = hpfilter(envelope)
#         smoothed_envelope = band_pass_filter(data=envelope, Fs=Fs, 
#                                         low=0.6, high=6, order=3)
#     elif smooth_filter == 'MA':
#         smoothed_envelope = moving_window_integration(signal = envelope, 
#                                                       fs = Fs,
#                                                       win_duration = 0.040)
    
#     smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope))/np.std(smoothed_envelope)
#     acf_x = acf(smoothed_envelope, nlags=n_lag)
#     acf_x = acf_x / acf_x[0]
    
#     nfft = next_power_of_2(x = len(x) * 2)
#     f, Pxx_den = periodogram(acf_x, fs = Fs, nfft = nfft)
    
#     ####### compute entropy #####################
#     pos_fft = np.copy(Pxx_den)
#     pos_fft = pos_fft/np.sum(pos_fft)
#     # interest_ids = np.argwhere((0.7 < f) & (f < 3.7)).flatten()
#     interest_ids = np.argwhere(f<6).flatten()
#     power_interest = pos_fft[interest_ids]
#     fft_entropy = -np.sum(power_interest*np.log(power_interest))
#     ###############################################################
#     # print('fft_entropy:', fft_entropy)
    
#     sig_means = []
#     index = 0
#     frequency = f[np.argmax(Pxx_den)]
#     power = max(Pxx_den)
    
#     if show:
#         fig, ax = plt.subplots(6, 1, figsize=(16, 18))
#         ax[0].plot(x, label='raw data')
#         ax[1].plot(denoised_sig, label='wavelet denoised data')
#         ax[2].plot(envelope, label='envelope extraction by Hilbert transform')
#         ax[3].plot(smoothed_envelope, label='smoothed envelope')
#         ax[4].plot(acf_x, label='ACF of smoothed envelope')
#         ax[5].plot(f, Pxx_den, label='spectrum of ACF')
#         for i in range(len(ax)):
#             ax[i].legend()

#     while (index + window_size < len(acf_x)):
#         sig_means.append(np.mean(acf_x[index:index + window_size]))
#         index = index + window_size
    
#     peak_ids, _ = signal.find_peaks(acf_x, height = np.mean(acf_x))
#     # check if the peaks is periodic
#     cadidates = []
#     for peak_id in peak_ids:
#         if (peak_id > acf_low and peak_id < acf_high):#0.25 * Fs, 1.2 * Fs
#             cadidates.append(peak_id)
    
#     if len(cadidates) == 0:
#         r = 0
#         hr = 0
#     else:
#         r = max(acf_x[cadidates])

#     if f_low < frequency < 6 and ((power > 0.18 and r > 0.25) or (r > 0.40)):
#         if fusion:
#             # ##### pdf fusion
#             fusion_pdf, acf_pdf, new_power_spectrum_pdf, new_f = periodogram_acf_fusion(Pxx_den, 
#                                         f, acf_x, Fs,
#                                        target_f_range = [f_low, f_high], 
#                                        target_acf_range = [acf_low, acf_high])
#             # ##### hr fusion
#             hr = estimate_vital_fusion(fusion_pdf, acf_pdf, new_f)
#         else:
#             # if fft_entropy < 3.15:
#             #     ##### complete signal power spectrum calculation ##########
#             #     N = len(acf_x)
#             #     fft_result = np.fft.fft(acf_x)
#             #     fft_freqs = np.fft.fftfreq(N, 1/Fs)
#             #     pos_inds = int(np.ceil(N/2))
#             #     pos_fft_freqs = fft_freqs[:pos_inds]
#             #     pos_fft = np.abs(fft_result[:pos_inds])
#             #     pos_fft = pos_fft[:]/np.sum(pos_fft[:])
#             #     hr = pos_fft_freqs[np.argmax(pos_fft)] * 60
#             # else:
#             max_id = cadidates[np.argmax(acf_x[cadidates])]
#             hr = Fs/max_id * 60
#         res = [True, np.std(sig_means), frequency, power, r, hr]

#         # if show:
#         #     fig2, ax2 = plt.subplots(4, 1, figsize=(16, 8))
#         #     ax2[0].plot(f[power_valid_ids], 
#         #                 power_spectrum_pdf, label='original power spectrum')
#         #     ax2[1].plot(new_f, new_power_spectrum_pdf, label='new power spectrum')
#         #     ax2[2].plot(new_f, acf_pdf, label='acf')
#         #     ax2[2].plot(new_f[1:], acf_pdf_d, 
#         #                 label='acf first d')
#         #     ax2[2].scatter(new_f[peak_ids], acf_pdf[peak_ids], label='acf')
#         #     ax2[3].plot(new_f, fusion_pdf, label='fusion pdf')
        
#         if show:
#             fig.suptitle('good data hr : %.2f'%(hr))
#     else:
#         hr = 0
#         res = [False, np.std(sig_means), frequency, power, r, hr]
#         if show:
#             fig.suptitle('bad data %.2f'%(fft_entropy))
#     return res

def signal_quality_assessment_v4(x, Fs, n_lag, low, high,
                                 acf_low, acf_high, f_low,
                                 diff = 1, smooth_filter = 'MA',
                                 denoised_method = 'bandpass',
                                 show = False):
    x = (x - np.mean(x))/(np.std(x)+1e-6)
    if denoised_method == 'bandpass':
        denoised_sig = band_pass_filter(data=x, Fs=Fs,
                                        low=low, high=high, order=3,
                                        causal = True)
       
    if diff:
        # denoised_sig = np.diff(np.diff(denoised_sig))
        denoised_sig = np.diff(denoised_sig)
    index = 0
    window_size = int(Fs)
    z= hilbert(denoised_sig) #form the analytical signal
    envelope = np.abs(z)
   
    if smooth_filter == 'MA':
        smoothed_envelope = moving_window_integration(signal = envelope,
                                                      fs = Fs,
                                                      win_duration = 0.040)
   
    smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope))/np.std(smoothed_envelope)
    acf_x = acf(smoothed_envelope, nlags=n_lag)
    acf_x = acf_x / acf_x[0]
   
    nfft = next_power_of_2(x = len(x) * 2)
    f, Pxx_den = periodogram(acf_x, fs = Fs, nfft = nfft)
   
 
    sig_means = []
    index = 0
    frequency = f[np.argmax(Pxx_den)]
    power = max(Pxx_den)
    
    if show:
        fig, ax = plt.subplots(6, 1, figsize=(16, 18))
        ax[0].plot(x, label='raw data')
        ax[1].plot(denoised_sig, label='wavelet denoised data')
        ax[2].plot(envelope, label='envelope extraction by Hilbert transform')
        ax[3].plot(smoothed_envelope, label='smoothed envelope')
        ax[4].plot(acf_x, label='ACF of smoothed envelope')
        ax[5].plot(f, Pxx_den, label='spectrum of ACF')
        for i in range(len(ax)):
            ax[i].legend()

    while (index + window_size < len(acf_x)):
        sig_means.append(np.mean(acf_x[index:index + window_size]))
        index = index + window_size
   
    peak_ids, _ = signal.find_peaks(acf_x, height = np.mean(acf_x))
    # check if the peaks is periodic
    cadidates = []
    for peak_id in peak_ids:
        if (peak_id > acf_low and peak_id < acf_high):#0.25 * Fs, 1.2 * Fs
            cadidates.append(peak_id)
   
    if len(cadidates) == 0:
        r = 0
        hr = 0
    else:
        r = max(acf_x[cadidates])
    if f_low < frequency < 6  and ((power > 0.18 and r > 0.25) or (r > 0.40)):
    # if f_low < frequency < 6 and r > 0.70:
        max_id = cadidates[np.argmax(acf_x[cadidates])]
        hr = Fs/max_id * 60
        res = [True, np.std(sig_means), frequency, power, r, hr]
       
        if show:
            fig.suptitle('good data')
    else:
        hr = 0
        res = [False, np.std(sig_means), frequency, power, r, hr]
        if show:
            fig.suptitle('bad data')
    return res
