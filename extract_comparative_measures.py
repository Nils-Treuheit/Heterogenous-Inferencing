import pandas as pd
import numpy as np
from glob import glob
import math


def latency_advanced(one_run, three_run, eight_run):
    lat23 = one_run - (three_run/3)
    lat78 = one_run - (eight_run/8)  
    lat_merge1 = ((lat78 * (8/14)) + (lat23 * (3/4)))
    lat2 = (one_run*3) - three_run
    lat7 = (one_run*8) - eight_run  
    lat_merge2 = lat7 - (lat2 * 3)
    return (lat_merge1+lat_merge2)/2

def throughput_advanced(one_run, three_run, eight_run, lat):
    fps = 1/((one_run*8+(three_run-lat)/3+(eight_run-lat)/8+2*lat)/10)
    if lat > one_run: return (math.inf, fps)
    if lat < 0: return (None, fps)
    tfps = 1/(((one_run-lat)*8+(three_run-lat)/3+(eight_run-lat)/8)/10)
    return (tfps, fps)

def latency_simple(one_run, three_run):
    lat_v1 = ((one_run * 3) - three_run)/2
    lat_v2 = (one_run - (three_run/3))*(3/2)
    return (lat_v1+lat_v2)/2

def throughput_simple(one_run, three_run, lat):
    fps = 1/((one_run*4+(three_run-lat)/3+lat)/5)
    if lat > one_run: return (math.inf, fps)
    if lat < 0: return (None, fps)
    tfps = 1/(((one_run-lat)*4+(three_run-lat)/3)/5)
    return (tfps, fps)


log_folder = "./logs/"
result_folder = "./res/"

def scan_single_infer():
    f = open(log_folder+"single_infer_analysis.log","r")
    lines = f.readlines()
    runtimes = {"1":{},"3":{},"8":{}}
    layer_conf = 0
    model = ""
    for idx, line in enumerate(lines):
        if idx%26==0: 
            model = line.split(":")[0].strip()
            if "stacked" not in model: layer_conf = "1"
            elif "stacked3" in model: layer_conf = "3"
            else: layer_conf = "8"
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
    f.close()
    return runtimes

def single_infer_res():
    runtimes = scan_single_infer()
    devices = ["CPU","GPU","MYRIAD","TPU"]
    models = [key.split("_stacked")[0] for key in runtimes["8"].keys()]
    incomplete_models = [key for key in runtimes["1"].keys() if key not in models]
    results = open(result_folder+"single_infer_res.txt","w")

    for model in models:
        results.write(model+":\n")
        for device in devices:
            vals = [runtimes["1"][model][device],
                    runtimes["3"][model+"_stacked3"][device],
                    runtimes["8"][model+"_stacked8"][device]]
            lat = latency_advanced(*vals)
            tfps,fps = throughput_advanced(*vals,lat)
            fps1,fps3,fps8 = (1/vals[0], 1/vals[1], 1/vals[2])
            results.write(" -> "+device+"\n")
            results.write("\tlatency: "+("no estimate possible" if lat<0 else ("{0:.6f} {1}sec".format(round(lat,12)*1000000,b'\xb5'.decode('latin-1')) if lat<vals[0] else "to close to runtime({0:.6f} {1}sec)".format(round(vals[0],12)*1000000,b'\xb5'.decode('latin-1'))))+"\n")
            results.write("\tthroughput: {0:d} fps\n".format(round(fps)))
            results.write("\traw_singleLayer_throughput: "+str(round(fps1))+" fps\n")
            results.write("\traw_threeLayer_throughput:  "+str(round(fps3))+" fps\n")
            results.write("\traw_eightLayer_throughput:  "+str(round(fps8))+" fps\n")
            results.write("\ttheoratical latency-free single-op throughput: "+("no estimate possible" if tfps==None else ("enormous" if tfps==math.inf else str(round(tfps))+" fps"))+"\n")
        results.write("\n")

    for model in incomplete_models:
        results.write(model+":\n")
        for device in devices:
            vals = [runtimes["1"][model][device],
                    runtimes["3"][model+"_stacked3"][device]]
            lat = latency_simple(*vals)
            tfps,fps =  throughput_simple(*vals,lat)
            fps1,fps3 = (1/vals[0], 1/vals[1])
            results.write(" -> "+device+"\n")
            results.write("\tlatency: "+("no estimate possible" if lat<0 else ("{0:.6f} {1}sec".format(round(lat,12)*1000000,b'\xb5'.decode('latin-1')) if lat<vals[0] else "to close to runtime({0:.6f} {1}sec)".format(round(vals[0],12)*1000000,b'\xb5'.decode('latin-1'))))+"\n")
            results.write("\tthroughput: {0:d} fps\n".format(round(fps)))
            results.write("\traw_singleLayer_throughput: "+str(round(fps1))+" fps\n")
            results.write("\traw_threeLayer_throughput:  "+str(round(fps3))+" fps\n")
            results.write("\traw_eightLayer_throughput:  "+str(round(fps8))+" fps\n")
            results.write("\ttheoratical latency-free single-op throughput: "+("no estimate possible" if tfps==None else ("enormous" if tfps==math.inf else str(round(tfps))+" fps"))+"\n")
        results.write("\n")
    results.close()        
        
def scan_batch_infer():
    f = open(log_folder+"sync_batch_infer_analysis.log","r")
    lines = f.readlines()
    runtimes = {"first":{"1":{},"3":{},"8":{}},"batch":{"1":{},"3":{},"8":{}}}
    layer_conf,model = (0,"")
    key = "batch"
    for idx, line in enumerate(lines):
        lid = idx-2 if idx<1067 and idx>1 else idx-1072 if idx>1071 and idx<2137 else 25 
        if idx>1071: key = "first"
        if lid%26==0:
            model = line.split(":")[0].strip()
            if "stacked" not in model: layer_conf = "1"
            elif "stacked3" in model: layer_conf = "3"
            else: layer_conf = "8"
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
    f.close()
    return runtimes

def batch_infer_res():
    runtimes = scan_batch_infer()
    devices = ["CPU","GPU","MYRIAD","TPU"]
    models = [key.split("_stacked")[0] for key in runtimes["batch"]["8"].keys()]
    incomplete_models = [key for key in runtimes["batch"]["1"].keys() if key not in models]
    results = open(result_folder+"batch_infer_res.txt","w")

    for model in models:
        results.write(model+":\n")
        for device in devices:
            vals = [runtimes["batch"]["1"][model][device],
                    runtimes["batch"]["3"][model+"_stacked3"][device],
                    runtimes["batch"]["8"][model+"_stacked8"][device]]
            lat = latency_advanced(*vals)
            tfps,fps = throughput_advanced(*vals,lat)
            fps1,fps3,fps8 = (1/vals[0], 1/vals[1], 1/vals[2])
            results.write(" -> "+device+"\n")
            results.write("\tlatency: "+("no estimate possible" if lat<0 else ("{0:.6f} {1}sec".format(round(lat,12)*1000000,b'\xb5'.decode('latin-1')) if lat<vals[0] else "to close to runtime({0:.6f} {1}sec)".format(round(vals[0],12)*1000000,b'\xb5'.decode('latin-1'))))+"\n")
            results.write("\tthroughput: {0:d} fps\n".format(round(fps)))
            results.write("\traw_singleLayer_throughput: "+str(round(fps1))+" fps\n")
            results.write("\traw_threeLayer_throughput:  "+str(round(fps3))+" fps\n")
            results.write("\traw_eightLayer_throughput:  "+str(round(fps8))+" fps\n")
            results.write("\ttheoratical latency-free single-op throughput: "+("no estimate possible" if tfps==None else ("enormous" if tfps==math.inf else str(round(tfps))+" fps"))+"\n")
        results.write("\n")

    for model in incomplete_models:
        results.write(model+":\n")
        for device in devices:
            vals = [runtimes["batch"]["1"][model][device],
                    runtimes["batch"]["3"][model+"_stacked3"][device]]
            lat = latency_simple(*vals)
            tfps,fps = throughput_simple(*vals,lat)
            fps1,fps3 = (1/vals[0], 1/vals[1])
            results.write(" -> "+device+"\n")
            results.write("\tlatency: "+("no estimate possible" if lat<0 else ("{0:.6f} {1}sec".format(round(lat,12)*1000000,b'\xb5'.decode('latin-1')) if lat<vals[0] else "to close to runtime({0:.6f} {1}sec)".format(round(vals[0],12)*1000000,b'\xb5'.decode('latin-1'))))+"\n")
            results.write("\tthroughput: {0:d} fps\n".format(round(fps)))
            results.write("\traw_singleLayer_throughput: "+str(round(fps1))+" fps\n")
            results.write("\traw_threeLayer_throughput:  "+str(round(fps3))+" fps\n")
            results.write("\traw_eightLayer_throughput:  "+str(round(fps8))+" fps\n")
            results.write("\ttheoratical latency-free single-op throughput: "+("no estimate possible" if tfps==None else ("enormous" if tfps==math.inf else str(round(tfps))+" fps"))+"\n")
        results.write("\n")
    results.close()

    models = [key.split("_stacked")[0] for key in runtimes["batch"]["8"].keys()]
    incomplete_models = [key for key in runtimes["batch"]["1"].keys() if key not in models]
    results = open(result_folder+"first_batch_infer_res.txt","w")

    for model in models:
        results.write(model+":\n")
        for device in devices:
            vals = [runtimes["first"]["1"][model][device],
                    runtimes["first"]["3"][model+"_stacked3"][device],
                    runtimes["first"]["8"][model+"_stacked8"][device]]
            lat = latency_advanced(*vals)
            tfps,fps = throughput_advanced(*vals,lat)
            fps1,fps3,fps8 = (1/vals[0], 1/vals[1], 1/vals[2])
            results.write(" -> "+device+"\n")
            results.write("\tlatency: "+("no estimate possible" if lat<0 else ("{0:.6f} {1}sec".format(round(lat,12)*1000000,b'\xb5'.decode('latin-1')) if lat<vals[0] else "to close to runtime({0:.6f} {1}sec)".format(round(vals[0],12)*1000000,b'\xb5'.decode('latin-1'))))+"\n")
            results.write("\tthroughput: {0:d} fps\n".format(round(fps)))
            results.write("\traw_singleLayer_throughput: "+str(round(fps1))+" fps\n")
            results.write("\traw_threeLayer_throughput:  "+str(round(fps3))+" fps\n")
            results.write("\traw_eightLayer_throughput:  "+str(round(fps8))+" fps\n")
            results.write("\ttheoratical latency-free single-op throughput: "+("no estimate possible" if tfps==None else ("enormous" if tfps==math.inf else str(round(tfps))+" fps"))+"\n")
        results.write("\n")

    for model in incomplete_models:
        results.write(model+":\n")
        for device in devices:
            vals = [runtimes["first"]["1"][model][device],
                    runtimes["first"]["3"][model+"_stacked3"][device]]
            lat = latency_simple(*vals)
            tfps,fps = throughput_simple(*vals,lat)
            fps1,fps3 = (1/vals[0], 1/vals[1])
            results.write(" -> "+device+"\n")
            results.write("\tlatency: "+("no estimate possible" if lat<0 else ("{0:.6f} {1}sec".format(round(lat,12)*1000000,b'\xb5'.decode('latin-1')) if lat<vals[0] else "to close to runtime({0:.6f} {1}sec)".format(round(vals[0],12)*1000000,b'\xb5'.decode('latin-1'))))+"\n")
            results.write("\tthroughput: {0:d} fps\n".format(round(fps)))
            results.write("\traw_singleLayer_throughput: "+str(round(fps1))+" fps\n")
            results.write("\traw_threeLayer_throughput:  "+str(round(fps3))+" fps\n")
            results.write("\traw_eightLayer_throughput:  "+str(round(fps8))+" fps\n")
            results.write("\ttheoratical latency-free single-op throughput: "+("no estimate possible" if tfps==None else ("enormous" if tfps==math.inf else str(round(tfps))+" fps"))+"\n")
        results.write("\n")
    results.close()

def scan_async_infer():
    f = open(log_folder+"async_batch_infer_analysis.log","r")
    lines = f.readlines()
    runtimes = {"1":{},"3":{},"8":{}}
    layer_conf = 0
    model = ""
    for idx, line in enumerate(lines):
        if idx%20==0: 
            model = line.split(":")[0].strip()
            if "stacked" not in model: layer_conf = "1"
            elif "stacked3" in model: layer_conf = "3"
            else: layer_conf = "8"
            runtimes[layer_conf][model] = dict() 
        elif idx%20==3: 
            runtimes[layer_conf][model]["CPU"] = \
                float(line.split(":")[-1].strip())
        elif idx%20==9: 
            runtimes[layer_conf][model]["GPU"] = \
                float(line.split(":")[-1].strip())
        elif idx%20==15: 
            runtimes[layer_conf][model]["MYRIAD"] = \
                float(line.split(":")[-1].strip())
    f.close()
    return runtimes

def async_infer_res():
    runtimes = scan_async_infer()
    devices = ["CPU","GPU","MYRIAD"]
    models = [key.split("_stacked")[0] for key in runtimes["8"].keys()]
    incomplete_models = [key for key in runtimes["1"].keys() if key not in models]
    results = open(result_folder+"async_batch_infer_res.txt","w")

    for model in models:
        results.write(model+":\n")
        for device in devices:
            vals = [runtimes["1"][model][device],
                    runtimes["3"][model+"_stacked3"][device],
                    runtimes["8"][model+"_stacked8"][device]]
            lat = latency_advanced(*vals)
            tfps,fps = throughput_advanced(*vals,lat)
            fps1,fps3,fps8 = (1/vals[0], 1/vals[1], 1/vals[2])
            results.write(" -> "+device+"\n")
            results.write("\tlatency: "+("no estimate possible" if lat<0 else ("{0:.6f} {1}sec".format(round(lat,12)*1000000,b'\xb5'.decode('latin-1')) if lat<vals[0] else "to close to runtime({0:.6f} {1}sec)".format(round(vals[0],12)*1000000,b'\xb5'.decode('latin-1'))))+"\n")
            results.write("\tthroughput: {0:d} fps\n".format(round(fps)))
            results.write("\traw_singleLayer_throughput: "+str(round(fps1))+" fps\n")
            results.write("\traw_threeLayer_throughput:  "+str(round(fps3))+" fps\n")
            results.write("\traw_eightLayer_throughput:  "+str(round(fps8))+" fps\n")
            results.write("\ttheoratical latency-free single-op throughput: "+("no estimate possible" if tfps==None else ("enormous" if tfps==math.inf else str(round(tfps))+" fps"))+"\n")
        results.write("\n")

    for model in incomplete_models:
        results.write(model+":\n")
        for device in devices:
            vals = [runtimes["1"][model][device],
                    runtimes["3"][model+"_stacked3"][device]]
            lat = latency_simple(*vals)
            tfps,fps = throughput_simple(*vals,lat)
            fps1,fps3 = (1/vals[0], 1/vals[1])
            results.write(" -> "+device+"\n")
            results.write("\tlatency: "+("no estimate possible" if lat<0 else ("{0:.6f} {1}sec".format(round(lat,12)*1000000,b'\xb5'.decode('latin-1')) if lat<vals[0] else "to close to runtime({0:.6f} {1}sec)".format(round(vals[0],12)*1000000,b'\xb5'.decode('latin-1'))))+"\n")
            results.write("\tthroughput: {0:d} fps\n".format(round(fps)))
            results.write("\traw_singleLayer_throughput: "+str(round(fps1))+" fps\n")
            results.write("\traw_threeLayer_throughput:  "+str(round(fps3))+" fps\n")
            results.write("\traw_eightLayer_throughput:  "+str(round(fps8))+" fps\n")
            results.write("\ttheoratical latency-free single-op throughput: "+("no estimate possible" if tfps==None else ("enormous" if tfps==math.inf else str(round(tfps))+" fps"))+"\n")
        results.write("\n")
    results.close()


# main
single_infer_res()
batch_infer_res()
async_infer_res()