from tkinter import SINGLE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use
from glob import glob
from sys import argv



def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def clean_figure(idx):
    plt.figure(idx)
    plt.clf()
    plt.cla()
    plt.close()

# Parameters to toggle activation of information output and plots
PLOT_SINGLE = True
PLOT_FINAL = False
AUTO_RUN = True
PRINT_STATS = True
LOG_STATS = True
SAVE_VIOLIN = True
SAVE_STATS = True

if AUTO_RUN: use('Agg')

lf_name = "logs/sync_batch_infer_analysis.log"
subplot_pos = [221,222,223,224]
statEnum = ['min','mean','median','max','std']
partList = [(0,15),(15,21),(21,-1)]
fname = ['single-op','dense','conv']
model_list = "relu_act,relu_act_stacked3,relu_act_stacked8,"+ \
             "leaky_relu_act,leaky_relu_act_stacked3,leaky_relu_act_stacked8,"+ \
             "tanh_act,tanh_act_stacked3,tanh_act_stacked8,"+ \
             "sigmoid_act,sigmoid_act_stacked3,sigmoid_act_stacked8,"+ \
             "scalar_mult,scalar_mult_stacked3,scalar_mult_stacked8,"+ \
             "small_dense,small_dense_stacked3,small_dense_stacked8,"+ \
             "big_dense,big_dense_stacked3,big_dense_stacked8,"+ \
             "simple_conv2d,simple_conv2d_stacked3,simple_conv2d_stacked8,"+ \
             "dilated_conv2d,dilated_conv2d_stacked3,dilated_conv2d_stacked8,"+ \
             "strided_conv2d,strided_conv2d_stacked3,"+ \
             "big_conv2d,big_conv2d_stacked3,big_conv2d_stacked8,"+ \
             "small_conv2d,small_conv2d_stacked3,small_conv2d_stacked8,"+ \
             "many_conv2d,many_conv2d_stacked3,many_conv2d_stacked8,"+ \
             "few_conv2d,few_conv2d_stacked3,few_conv2d_stacked8"
device_list = "CPU,GPU,MYRIAD,TPU"

if len(argv)>1: model_list = argv[1]
if len(argv)>2: device_list = argv[2]

devices = device_list.split(",")
models = model_list.split(",")
 
infer_times = dict()
for device in devices:
    infer_times[device] = dict()
    #folder = "OpenVINO" if device in "CPU,GPU,MYRIAD" else "Edge_TPU" 
    files = glob("*/*/avg_"+device+"_*_sync_*.csv")
    for model in models:
        infer_times[device][model] = {"first":list(), "avg":list()}
        mask = [model in file for file in files]
        model_files = [files[idx] for idx,marker in enumerate(mask) if marker]
        for file_name in model_files:
            with open(file_name,"r") as file:
                data = pd.read_csv(file,header=0)
                infer_times[device][model]["first"].append(data['first'].values)
                infer_times[device][model]["avg"].append(data['avgTail'].values) 

if LOG_STATS:
    log_file = open(lf_name,"w")
    log_file.close()

    log_file = open(lf_name,"a")
    log_file.write("AVG of Inference Batch Tail:\n----------------------------\n")
    log_file.close()
if PRINT_STATS: print("AVG of Inference Batch Tail:\n----------------------------")

statMap = dict()
for model in models:
    mini,maxi = (1,0)
    data,stats = ([],[])
    if LOG_STATS:
        log_file = open(lf_name,"a")
        log_file.write(model+':\n')
    if PRINT_STATS: print(model+':')
    for device in devices:
        new_list = list()
        for array in infer_times[device][model]["avg"]: new_list.extend(array.tolist())
        infer_times[device][model]["avg"] = new_list
        if len(new_list)>0:
            clean_figure(1)
            fig = plt.figure(1, figsize=(25.5,13.25))
            plt.hist(infer_times[device][model]["avg"],bins=16,edgecolor='None', alpha = 0.4)
            dev_stats = (min(infer_times[device][model]["avg"]),sum(infer_times[device][model]["avg"])/len(infer_times[device][model]["avg"]), \
                np.percentile(infer_times[device][model]["avg"],50),max(infer_times[device][model]["avg"]),np.std(infer_times[device][model]["avg"]))
            if PRINT_STATS: 
                print('->'+device+':')
                print("\tmin:   ",dev_stats[0])
                print("\tmean:  ",dev_stats[1])
                print("\tmedian:",dev_stats[2])
                print("\tmax:   ",dev_stats[3])
                print("\tstd:   ",dev_stats[4])
            if LOG_STATS:
                log_file.write('->'+device+':\n')
                log_file.write("\tmin:   "+str(dev_stats[0])+"\n")
                log_file.write("\tmean:  "+str(dev_stats[1])+"\n")
                log_file.write("\tmedian:"+str(dev_stats[2])+"\n")
                log_file.write("\tmax:   "+str(dev_stats[3])+"\n")
                log_file.write("\tstd:   "+str(dev_stats[4])+"\n")
            mini = dev_stats[0] if dev_stats[0]<mini else mini
            maxi = dev_stats[3] if dev_stats[3]>maxi else maxi    
            data.append(new_list) 
            stats.append(dev_stats)  
            plt.xlim(mini,maxi)
    statMap[model] = stats
    fig.suptitle("AVG in the last 63 out of a 64 Inferences batch on")
    plt.ylabel('Frequency')
    plt.xlabel('Runtime')
    plt.legend(devices)
    plt.title(model)
    plt.subplots_adjust(left=0.06,bottom=0.06,top=0.95,right=0.98)
    if len(data)>0:
        clean_figure(2)
        fig = plt.figure(2, figsize=(25.5,13.25))
        fig.suptitle("AVG in the last 63 out of a 64 Inferences batch on")
        parts = plt.violinplot(data,showmeans=False,showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        means, quartile1, medians, quartile3 = ([],[],[],[])
        for data_elem in data:
            q1,med,q3 = np.percentile(data_elem, [25, 50, 75])
            means.append(np.mean(data_elem))
            quartile1.append(q1)
            medians.append(med)
            quartile3.append(q3)
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
        inds = np.arange(1, len(medians) + 1)
        medi = plt.scatter(inds, medians, marker='o', color='lime', s=35, zorder=3)
        mean = plt.scatter(inds, means, marker='o', color='cyan', s=25, zorder=4)
        quart = plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        whisk = plt.vlines(inds, whiskers_min, whiskers_max, color='y', linestyle='-', lw=1)
        plt.xticks([*range(1,len(devices)+1)],devices)
        plt.legend(handles=[mean,medi,quart,whisk],labels=['mean','median','quartile','whiskers'])
        plt.ylabel('Runtime')
        plt.title(model)
        plt.subplots_adjust(left=0.06,bottom=0.06,top=0.95,right=0.98)
        if SAVE_VIOLIN: plt.savefig("violin_plots/sync_batch_avg_"+model+".png")
        if LOG_STATS:
            log_file.write("\n")
            log_file.close()
        if PLOT_SINGLE: plt.show()
    if not(AUTO_RUN):
        ui = input("To cancel enter 'q', otherwise you will continue with the next model!")
        if(ui.lower() == 'q'): break
    print("\n")


for idx in range(1,4): clean_figure(idx)
for part in range(3):
    fig = plt.figure(part+1, figsize=(25.5,13.25))
    for idx,dev in enumerate(devices):
        devStats = [val[idx] if len(val)>idx else None for val in statMap.values()][partList[part][0]:partList[part][1]] 
        ax = fig.add_subplot(subplot_pos[idx])
        x = [*range(len(statMap.keys()))][partList[part][0]:partList[part][1]]
        ticks = ["\nstacked3" if "stacked3" in key else "\n\nstacked8" if "stacked8" in key else "  big_conv2d" if "big_conv2d" == key else key \
            for key in statMap.keys()][partList[part][0]:partList[part][1]]
        for id,enum in enumerate(statEnum):
            ax.plot(x,[stat[id] if stat else float('NaN') for stat in devStats],label=enum)
        ax.set_title(dev)
        ax.legend()
        ax.set_xticks(x,ticks)
        ax.set_ylabel('Runtime')
        #ax.set_xlabel('Models')
    fig.suptitle('AVG in the last 63 out of a 64 Inferences batch on')
    plt.subplots_adjust(left=0.06,bottom=0.06,top=0.95,right=0.98)
    if SAVE_STATS: plt.savefig("plots/sync_batch_avg_"+fname[part]+"_stats.png")
if PLOT_SINGLE: plt.show()

if LOG_STATS:
    log_file = open(lf_name,"a")
    log_file.write("\n\nFirst of Inference Batch:\n-------------------------\n")
    log_file.close()
if PRINT_STATS: print("First of Inference Batch:\n-------------------------")

statMap = dict()
for model in models:
    mini,maxi = (1,0)
    data,stats = ([],[])
    if LOG_STATS:
        log_file = open(lf_name,"a")
        log_file.write(model+':\n')
    if PRINT_STATS: print(model+':')
    for device in devices:
        new_list = list()
        for array in infer_times[device][model]["first"]: new_list.extend(array.tolist())
        infer_times[device][model]["first"] = new_list
        if len(new_list)>0:
            clean_figure(4)
            fig = plt.figure(4, figsize=(25.5,13.25))
            plt.hist(infer_times[device][model]["first"],bins=16,edgecolor='None', alpha = 0.4)
            dev_stats = (min(infer_times[device][model]["first"]),sum(infer_times[device][model]["first"])/len(infer_times[device][model]["first"]), \
                np.percentile(infer_times[device][model]["first"],50),max(infer_times[device][model]["first"]),np.std(infer_times[device][model]["first"]))
            if PRINT_STATS: 
                print('->'+device+':')
                print("\tmin:   ",dev_stats[0])
                print("\tmean:  ",dev_stats[1])
                print("\tmedian:",dev_stats[2])
                print("\tmax:   ",dev_stats[3])
                print("\tstd:   ",dev_stats[4])
            if LOG_STATS:
                log_file.write('->'+device+':\n')
                log_file.write("\tmin:   "+str(dev_stats[0])+"\n")
                log_file.write("\tmean:  "+str(dev_stats[1])+"\n")
                log_file.write("\tmedian:"+str(dev_stats[2])+"\n")
                log_file.write("\tmax:   "+str(dev_stats[3])+"\n")
                log_file.write("\tstd:   "+str(dev_stats[4])+"\n")
            mini = dev_stats[0] if dev_stats[0]<mini else mini
            maxi = dev_stats[3] if dev_stats[3]>maxi else maxi    
            data.append(new_list) 
            stats.append(dev_stats)  
            plt.xlim(mini,maxi)
    statMap[model] = stats
    fig.suptitle("First Inference out of a 64 Inferences batch on")
    plt.ylabel('Frequency')
    plt.xlabel('Runtime')
    plt.legend(devices)
    plt.title(model)
    plt.subplots_adjust(left=0.06,bottom=0.06,top=0.95,right=0.98)
    if len(data)>0:
        clean_figure(5)
        fig = plt.figure(5, figsize=(25.5,13.25))
        fig.suptitle("First Inference out of a 64 Inferences batch on")
        parts = plt.violinplot(data,showmeans=False,showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        means, quartile1, medians, quartile3 = ([],[],[],[])
        for data_elem in data:
            q1,med,q3 = np.percentile(data_elem, [25, 50, 75])
            means.append(np.mean(data_elem))
            quartile1.append(q1)
            medians.append(med)
            quartile3.append(q3)
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
        inds = np.arange(1, len(medians) + 1)
        medi = plt.scatter(inds, medians, marker='o', color='lime', s=35, zorder=3)
        mean = plt.scatter(inds, means, marker='o', color='cyan', s=25, zorder=4)
        quart = plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        whisk = plt.vlines(inds, whiskers_min, whiskers_max, color='y', linestyle='-', lw=1)
        plt.xticks([*range(1,len(devices)+1)],devices)
        plt.legend(handles=[mean,medi,quart,whisk],labels=['mean','median','quartile','whiskers'])
        plt.ylabel('Runtime')
        plt.title(model)
        plt.subplots_adjust(left=0.06,bottom=0.06,top=0.95,right=0.98)
        if SAVE_VIOLIN: plt.savefig("violin_plots/sync_batch_first_"+model+".png")
        if LOG_STATS:
            log_file.write("\n")
            log_file.close()
        if PLOT_SINGLE: plt.show()
    if not(AUTO_RUN):
        ui = input("To cancel enter 'q', otherwise you will continue with the next model!")
        if(ui.lower() == 'q'): break
    print("\n")


for idx in range(4,7): clean_figure(idx)
for part in range(3):
    fig = plt.figure(part+4, figsize=(25.5,13.25))
    for idx,dev in enumerate(devices):
        devStats = [val[idx] if len(val)>idx else None for val in statMap.values()][partList[part][0]:partList[part][1]] 
        ax = fig.add_subplot(subplot_pos[idx])
        x = [*range(len(statMap.keys()))][partList[part][0]:partList[part][1]]
        ticks = ["\nstacked3" if "stacked3" in key else "\n\nstacked8" if "stacked8" in key else "  big_conv2d" if "big_conv2d" == key else key \
            for key in statMap.keys()][partList[part][0]:partList[part][1]]
        for id,enum in enumerate(statEnum):
            ax.plot(x,[stat[id] if stat else float('NaN') for stat in devStats],label=enum)
        ax.set_title(dev)
        ax.legend()
        ax.set_xticks(x,ticks)
        ax.set_ylabel('Runtime')
        #ax.set_xlabel('Models')
    fig.suptitle('First Inference out of a 64 Inferences batch on')
    plt.subplots_adjust(left=0.06,bottom=0.06,top=0.95,right=0.98)
    if SAVE_STATS: plt.savefig("plots/sync_batch_first_"+fname[part]+"_stats.png")
if PLOT_SINGLE: plt.show()

if LOG_STATS:
    log_file = open(lf_name,"a")
    log_file.write("\n\nDiffernce between First and AVG of the Tail in Inference Batch:\n---------------------------------------------------------------\n")
    log_file.close()
if PRINT_STATS: print("Differnce between First and AVG of the Tail in Inference Batch:\n---------------------------------------------------------------")

statMap = dict()
for model in models:
    mini,maxi = (1,0)
    data,stats = ([],[])
    if LOG_STATS:
        log_file = open(lf_name,"a")
        log_file.write(model+':\n')
    if PRINT_STATS: print(model+':')
    for device in devices:
        new_list = list()
        for first_arr,avg_arr in zip(infer_times[device][model]["first"],infer_times[device][model]["avg"]): 
            new_list.append(first_arr-avg_arr)
        if len(new_list)>0:
            clean_figure(7)
            fig = plt.figure(7, figsize=(25.5,13.25))
            plt.hist(new_list,bins=16,edgecolor='None', alpha = 0.4)
            dev_stats = (min(new_list),sum(new_list)/len(new_list),np.percentile(new_list,50),max(new_list),np.std(new_list))
            if PRINT_STATS: 
                print('->'+device+':')
                print("\tmin:   ",dev_stats[0])
                print("\tmean:  ",dev_stats[1])
                print("\tmedian:",dev_stats[2])
                print("\tmax:   ",dev_stats[3])
                print("\tstd:   ",dev_stats[4])
            if LOG_STATS:
                log_file.write('->'+device+':\n')
                log_file.write("\tmin:   "+str(dev_stats[0])+"\n")
                log_file.write("\tmean:  "+str(dev_stats[1])+"\n")
                log_file.write("\tmedian:"+str(dev_stats[2])+"\n")
                log_file.write("\tmax:   "+str(dev_stats[3])+"\n")
                log_file.write("\tstd:   "+str(dev_stats[4])+"\n")
            mini = dev_stats[0] if dev_stats[0]<mini else mini
            maxi = dev_stats[3] if dev_stats[3]>maxi else maxi    
            data.append(new_list) 
            stats.append(dev_stats)  
            plt.xlim(mini,maxi)
    statMap[model] = stats
    fig.suptitle("Differnce between First and AVG in a 64 Inference batch batch on")
    plt.ylabel('Frequency')
    plt.xlabel('Runtime')
    plt.legend(devices)
    plt.title(model)
    plt.subplots_adjust(left=0.06,bottom=0.06,top=0.95,right=0.98)
    if len(data)>0:
        clean_figure(8)
        fig = plt.figure(8, figsize=(25.5,13.25))
        fig.suptitle("Differnce between First and AVG in a 64 Inference batch batch on")
        parts = plt.violinplot(data,showmeans=False,showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        means, quartile1, medians, quartile3 = ([],[],[],[])
        for data_elem in data:
            q1,med,q3 = np.percentile(data_elem, [25, 50, 75])
            means.append(np.mean(data_elem))
            quartile1.append(q1)
            medians.append(med)
            quartile3.append(q3)
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
        inds = np.arange(1, len(medians) + 1)
        medi = plt.scatter(inds, medians, marker='o', color='lime', s=35, zorder=3)
        mean = plt.scatter(inds, means, marker='o', color='cyan', s=25, zorder=4)
        quart = plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        whisk = plt.vlines(inds, whiskers_min, whiskers_max, color='y', linestyle='-', lw=1)
        plt.xticks([*range(1,len(devices)+1)],devices)
        plt.legend(handles=[mean,medi,quart,whisk],labels=['mean','median','quartile','whiskers'])
        plt.ylabel('Runtime')
        plt.title(model)
        plt.subplots_adjust(left=0.06,bottom=0.06,top=0.95,right=0.98)
        if SAVE_VIOLIN: plt.savefig("violin_plots/sync_batch_diff_"+model+".png")
        if LOG_STATS:
            log_file.write("\n")
            log_file.close()
        if PLOT_SINGLE: plt.show()
    if not(AUTO_RUN):
        ui = input("To cancel enter 'q', otherwise you will continue with the next model!")
        if(ui.lower() == 'q'): break
    print("\n")


for idx in range(7,10): clean_figure(idx)
for part in range(3):
    fig = plt.figure(part+7, figsize=(25.5,13.25))
    for idx,dev in enumerate(devices):
        devStats = [val[idx] if len(val)>idx else None for val in statMap.values()][partList[part][0]:partList[part][1]] 
        ax = fig.add_subplot(subplot_pos[idx])
        x = [*range(len(statMap.keys()))][partList[part][0]:partList[part][1]]
        ticks = ["\nstacked3" if "stacked3" in key else "\n\nstacked8" if "stacked8" in key else "  big_conv2d" if "big_conv2d" == key else key \
            for key in statMap.keys()][partList[part][0]:partList[part][1]]
        for id,enum in enumerate(statEnum):
            ax.plot(x,[stat[id] if stat else float('NaN') for stat in devStats],label=enum)
        ax.set_title(dev)
        ax.legend()
        ax.set_xticks(x,ticks)
        ax.set_ylabel('Runtime')
        #ax.set_xlabel('Models')
    fig.suptitle('Differnce between First and AVG in a 64 Inference batch batch on')
    plt.subplots_adjust(left=0.06,bottom=0.06,top=0.95,right=0.98)
    if SAVE_STATS: plt.savefig("plots/sync_batch_diff_"+fname[part]+"_stats.png")
if PLOT_FINAL: plt.show()