import os
import os.path
import csv
from pathlib import Path
from datetime import date
from datetime import datetime

import numpy as np

from csv_helpers import getTimeStamp

tf_net_names=[
"relu_act",
#"leaky_relu_act",
#"tanh_act",
#"sigmoid_act",
"scalar_mult",
#"small_dense",
"big_dense",
"simple_conv2d",
"strided_conv2d",
"dilated_conv2d",
#"small_conv2d",
"big_conv2d",
"few_conv2d",
#"many_conv2d", # TODO: Wieso problematisch?
]

iterations=1024
iterations_single=32
global_iterations=3#32

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
    rows=len(measurements[list(measurements)[0]])
    criteria=list(measurements)
    for net in tf_net_names: # Messungen fuer alle gegebennen Netze
        net_measurements=[]
        net_measurements_names=[]
        for c in criteria: 
            if c.find(net)!=-1:
                net_measurements.append(c)
                measurement=c[0:str(c).find("(")]
                net_measurements_names.append(measurement)

        results=[]
        for i in range(rows):
            single_row=[]
            for c in net_measurements:
                single_row.append(measurements[c][i])

            results.append(single_row)
        
        time_stamp=getTimeStamp(date.today(),datetime.now())

        directory2=os.path.join(directory,net)
        if not os.path.isdir(directory2): os.mkdir(directory2)

        target_path=os.path.join(directory2,measured_property+"_"+target+"_"+toolkit+"_"+mode+"_"+time_stamp+".csv")
        writeFile(target_path,net_measurements_names,results)


def writeFile(target_path,measurements,measurements_to_write):
    if not Path(target_path).exists():
        with open(target_path,"w") as file:
            wr=csv.writer(file)
            wr.writerow(list(measurements))
    with open(target_path,"a") as file:
        wr=csv.writer(file)
        wr.writerows(measurements_to_write)

def getOpenvinoExampelData(data_format):
    return np.random.uniform(np.finfo(np.half).min,np.finfo(np.half).max,data_format).astype(np.float16)


picture_software="ffmpeg "
picture_software="streamer -f jpeg -o "

class EnergyMeasurementEnclosure:
    def __init__(self,netw) -> None:
        time_stamp=str(datetime.date.today())+"_"+str(datetime.now().hour)+"-"+str(datetime.now().minute)
        prefix=netw+"_"+time_stamp
        self.command_start=picture_software+prefix+"_begin.jpeg"
        self.command_end=picture_software+prefix+"_"+time_stamp+"_end.jpeg"

    def start(self):
        os.system(self.command_start)

    def end(self):
        os.system(self.command_end)