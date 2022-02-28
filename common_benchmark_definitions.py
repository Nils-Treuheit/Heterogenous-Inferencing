import os
import os.path
import csv
from pathlib import Path
from datetime import date
from datetime import datetime

import numpy as np

tf_net_names=[
"relu_act",
"leaky_relu_act",
"tanh_act",
"sigmoid_act",
"scalar_mult",
"small_dense",
"big_dense",
"simple_conv2d",
"strided_conv2d",
"dilated_conv2d",
"small_conv2d",
"big_conv2d",
"few_conv2d",
"many_conv2d", # TODO: Wieso problematisch?
]

iterations=10#512
iterations_single=32
global_iterations=2#32

models_openvino=os.path.join(".","OpenVINO-Models")

measurements_openvino="OpenVINO-Measurements"
measurements_coral="Edge_TPU-Measurements"
if not os.path.isdir(measurements_openvino): os.mkdir(measurements_openvino)

def startOpenvinoNet(name,infCore,target):
    model_path=os.path.join(models_openvino,name+".xml")
    weights_path=os.path.join(models_openvino,name+".bin")
    n=infCore.read_network(model=model_path,weights=weights_path)
    netw=infCore.load_network(network=n,device_name=target,num_requests=iterations)
    return netw

def writeResults(target,measurements,measured_property,toolkit,mode):
    if toolkit=="openvino":
        directory=measurements_openvino
    else:
        directory=measurements_coral
    if not os.path.isdir(directory): os.mkdir(directory)
    time_stamp=str(date.today())+"_"+str(datetime.now().hour)+"-"+str(datetime.now().minute)
    target_path=os.path.join(directory,measured_property+"_"+target+"_"+toolkit+"_"+mode+"_"+time_stamp+".csv")
    rows=len(measurements[list(measurements)[0]])
    measurements_to_write=[[measurements[name][k] for name in list(measurements)] for k in range(rows)]

    if not Path(target_path).exists():
        with open(target_path,"w") as file:
            wr=csv.writer(file)
            wr.writerow(list(measurements))
    with open(target_path,"a") as file:
        wr=csv.writer(file)
        wr.writerows(measurements_to_write)

def getOpenvinoExampelData(data_format):
    return np.random.uniform(np.finfo(np.half).min,np.finfo(np.half).max,data_format).astype(np.float16)