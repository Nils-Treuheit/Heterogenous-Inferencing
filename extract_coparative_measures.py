import pandas as pd
import numpy as np
from glob import glob


def latency_advanced(one_run, three_run, eight_run):
    lat2 = one_run * 3 - three_run
    lat8 = one_run * 8 - eight_run
    lat2 = lat8 - lat2 * 3
    return lat2/2

def throughput_advanced(one_run, three_run, eight_run, lat):
    fps1 = 1/(one_run - lat)
    fps3 = 1/((three_run - lat)/3)
    fps8 = 1/((eight_run - lat)/8)
    return (fps1+fps3+fps8)/3

def latency_simple(one_run, three_run):
    lat2 = one_run * 3 - three_run
    return lat2/2

def throughput_simple(one_run, three_run, lat):
    fps1 = 1/(one_run - lat)
    fps3 = 1/((three_run - lat)/3)
    return (fps1+fps3)/2


log_folder = "./logs/"
result_folder = "./res/"

def scan_single_infer():
    f = open(log_folder+"single_infer_analysis.log","r")
    lines = f.readlines()
    runtimes = {1:{},3:{},8:{}}
    layer_conf = 0
    model = ""
    for idx, line in enumerate(lines):
        if idx%26==0: 
            model = line.split(":")[0].strip()
            if "stacked" not in model: layer_conf = 1
            elif "stacked3" in model: layer_conf = 3
            else: layer_conf = 8
            runtimes[layer_conf][model] = dict() 
        elif idx%26==3: 
            runtimes[layer_conf][model]["CPU"] = \
                float(line.split(":")[-1].strip())
        elif idx%26==9: 
            runtimes[layer_conf][model]["GPU"] = \
                float(line.split(":")[-1].strip())
        elif idx%26==15: 
            runtimes[layer_conf][model]["MYRIAD"] = \
                float(line.split(":")[-1].strip())
        elif idx%26==21:
            runtimes[layer_conf][model]["TPU"] = \
                float(line.split(":")[-1].strip())
    return runtimes

def single_infer_res():
    runtimes = scan_single_infer()
    devices = ["CPU","GPU","MYRIAD","TPU"]
    models = [key.split("_stacked")[0] for key in runtimes[8].keys()]
    incomplete_models = [key for key in runtimes[1].keys() if key not in models]
    results = open(result_folder+"single_infer_res.txt","w")

    for model in models:
        results.write(model+":\n")
        for device in devices:
            vals = [runtimes[1][model][device],
                    runtimes[3][model+"_stacked3"][device],
                    runtimes[8][model+"_stacked8"][device]]
            lat = latency_advanced(*vals)
            fps = throughput_advanced(*vals,lat)
            fps1,fps3,fps8 = (1/vals[0], 1/vals[1], 1/vals[2])
            results.write(" -> "+device+"\n")
            results.write("\tlatency: "+str(lat)+" sec\n")
            results.write("\tthroughput: "+str(fps)+" fps\n")
            results.write("\traw_singleLayer_throughput: "+str(fps1)+" fps\n")
            results.write("\traw_threeLayer_throughput:  "+str(fps3)+" fps\n")
            results.write("\traw_eightLayer_throughput:  "+str(fps8)+" fps\n")
        results.write("\n")

    for model in incomplete_models:
        results.write(model+":\n")
        for device in devices:
            vals = [runtimes[1][model][device],
                    runtimes[3][model+"_stacked3"][device]]
            lat = latency_simple(*vals)
            fps = throughput_simple(*vals,lat)
            fps1,fps3 = (1/vals[0], 1/vals[1])
            results.write(" -> "+device+"\n")
            results.write("\tlatency: "+str(lat)+" sec\n")
            results.write("\tthroughput: "+str(fps)+" fps\n")
            results.write("\traw_singleLayer_throughput: "+str(fps1)+" fps\n")
            results.write("\traw_threeLayer_throughput:  "+str(fps3)+" fps\n")
        results.write("\n")
    results.close()        
        
def scan_batch_infer():
    f = open(log_folder+"sync_batch_infer_analysis.log","r")
    lines = f.readlines()
    runtimes = {"first":{1:{},3:{},8:{}},"batch":{1:{},3:{},8:{}}}
    layer_conf,model = (0,"")
    key = "batch"
    for idx, line in enumerate(lines):
        lid = idx-2 if idx<1067 and idx>1 else idx-1072 if idx>1071 and idx<2137 else 25 
        if idx>1071: key = "first"
        if lid%26==0:
            model = line.split(":")[0].strip()
            if "stacked" not in model: layer_conf = 1
            elif "stacked3" in model: layer_conf = 3
            else: layer_conf = 8
            runtimes[key][layer_conf][model] = dict()
        elif lid%26==3: 
            runtimes[key][layer_conf][model]["CPU"] = \
                float(line.split(":")[-1].strip())
        elif lid%26==9: 
            runtimes[key][layer_conf][model]["GPU"] = \
                float(line.split(":")[-1].strip())
        elif lid%26==15: 
            runtimes[key][layer_conf][model]["MYRIAD"] = \
                float(line.split(":")[-1].strip())
        elif lid%26==21:
            runtimes[key][layer_conf][model]["TPU"] = \
                float(line.split(":")[-1].strip())
    return runtimes

def batch_infer_res():
    runtimes = scan_batch_infer()
    devices = ["CPU","GPU","MYRIAD","TPU"]
    models = [key.split("_stacked")[0] for key in runtimes["batch"][8].keys()]
    incomplete_models = [key for key in runtimes["batch"][1].keys() if key not in models]
    results = open(result_folder+"batch_infer_res.txt","w")

    for model in models:
        results.write(model+":\n")
        for device in devices:
            vals = [runtimes["batch"][1][model][device],
                    runtimes["batch"][3][model+"_stacked3"][device],
                    runtimes["batch"][8][model+"_stacked8"][device]]
            lat = latency_advanced(*vals)
            fps = throughput_advanced(*vals,lat)
            fps1,fps3,fps8 = (1/vals[0], 1/vals[1], 1/vals[2])
            results.write(" -> "+device+"\n")
            results.write("\tlatency: "+str(lat)+" sec\n")
            results.write("\tthroughput: "+str(fps)+" fps\n")
            results.write("\traw_singleLayer_throughput: "+str(fps1)+" fps\n")
            results.write("\traw_threeLayer_throughput:  "+str(fps3)+" fps\n")
            results.write("\traw_eightLayer_throughput:  "+str(fps8)+" fps\n")
        results.write("\n")

    for model in incomplete_models:
        results.write(model+":\n")
        for device in devices:
            vals = [runtimes["batch"][1][model][device],
                    runtimes["batch"][3][model+"_stacked3"][device]]
            lat = latency_simple(*vals)
            fps = throughput_simple(*vals,lat)
            fps1,fps3 = (1/vals[0], 1/vals[1])
            results.write(" -> "+device+"\n")
            results.write("\tlatency: "+str(lat)+" sec\n")
            results.write("\tthroughput: "+str(fps)+" fps\n")
            results.write("\traw_singleLayer_throughput: "+str(fps1)+" fps\n")
            results.write("\traw_threeLayer_throughput:  "+str(fps3)+" fps\n")
        results.write("\n")
    results.close()

    models = [key.split("_stacked")[0] for key in runtimes["batch"][8].keys()]
    incomplete_models = [key for key in runtimes["batch"][1].keys() if key not in models]
    results = open(result_folder+"first_batch_infer_res.txt","w")

    for model in models:
        results.write(model+":\n")
        for device in devices:
            vals = [runtimes["first"][1][model][device],
                    runtimes["first"][3][model+"_stacked3"][device],
                    runtimes["first"][8][model+"_stacked8"][device]]
            lat = latency_advanced(*vals)
            fps = throughput_advanced(*vals,lat)
            fps1,fps3,fps8 = (1/vals[0], 1/vals[1], 1/vals[2])
            results.write(" -> "+device+"\n")
            results.write("\tlatency: "+str(lat)+" sec\n")
            results.write("\tthroughput: "+str(fps)+" fps\n")
            results.write("\traw_singleLayer_throughput: "+str(fps1)+" fps\n")
            results.write("\traw_threeLayer_throughput:  "+str(fps3)+" fps\n")
            results.write("\traw_eightLayer_throughput:  "+str(fps8)+" fps\n")
        results.write("\n")

    for model in incomplete_models:
        results.write(model+":\n")
        for device in devices:
            vals = [runtimes["first"][1][model][device],
                    runtimes["first"][3][model+"_stacked3"][device]]
            lat = latency_simple(*vals)
            fps = throughput_simple(*vals,lat)
            fps1,fps3 = (1/vals[0], 1/vals[1])
            results.write(" -> "+device+"\n")
            results.write("\tlatency: "+str(lat)+" sec\n")
            results.write("\tthroughput: "+str(fps)+" fps\n")
            results.write("\traw_singleLayer_throughput: "+str(fps1)+" fps\n")
            results.write("\traw_threeLayer_throughput:  "+str(fps3)+" fps\n")
        results.write("\n")
    results.close()


# main
single_infer_res()
batch_infer_res()