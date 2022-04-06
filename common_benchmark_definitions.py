import os
import os.path
import csv
from pathlib import Path
from datetime import date
from datetime import datetime
import csv_helpers
import numpy as np

from csv_helpers import getTimeStamp

#tf_net_names=[
#"relu_act",
##"leaky_relu_act",
##"tanh_act",
##"sigmoid_act",
#"scalar_mult",
##"small_dense",
##"big_dense",
##"simple_conv2d",
##"strided_conv2d",
##"dilated_conv2d",
##"small_conv2d",
#"big_conv2d",
##"few_conv2d",
#   "many_conv2d",
#   'many_conv2d_stacked3',
#   'many_conv2d_stacked8',#TODO: Wieso problematisch?
#
#]
tf_net_names=[
    'big_conv2d',
    'big_conv2d_stacked3',
    'big_conv2d_stacked8',
    
    'big_dense', 
    'big_dense_stacked3',
    'big_dense_stacked8',
    
    'dilated_conv2d',
    'dilated_conv2d_stacked3',
    'dilated_conv2d_stacked8',
    
    'few_conv2d',
    'few_conv2d_stacked3',
    'few_conv2d_stacked8', 
    
    'leaky_relu_act',
    'leaky_relu_act_stacked3',
    'leaky_relu_act_stacked8',
    
#    'many_conv2d',
#    'many_conv2d_stacked3',
#    'many_conv2d_stacked8',
    
    'relu_act',
    'relu_act_stacked3',
    'relu_act_stacked8',
    
    'scalar_mult',
    'scalar_mult_stacked3',
    'scalar_mult_stacked8',
    
    'sigmoid_act',
    'sigmoid_act_stacked3',
    'sigmoid_act_stacked8',
    
    'simple_conv2d',
    'simple_conv2d_stacked3',
    'simple_conv2d_stacked8',
    
    'small_conv2d',
    'small_conv2d_stacked3',
    'small_conv2d_stacked8',
    
    'small_dense',
    'small_dense_stacked3',
    'small_dense_stacked8',

    'strided_conv2d',
    'strided_conv2d_stacked3',
    
    'tanh_act',
    'tanh_act_stacked3',
    'tanh_act_stacked8'
]
#alternate way:
#a=[]
#for x in Path(".").glob("*"):
#     a.append(str(x)) 
#sorted(a)
#

iterations=64#2048#64 
iterations_single=64#2048#64
global_iterations=32#16#32

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
    time_stamp=getTimeStamp(date.today(),datetime.now())
    writeConfig(measured_property+"_"+target+"_"+toolkit+"_"+mode+"_"+time_stamp)
    criteria=list(measurements)
    for net in tf_net_names: # Messungen fuer alle gegebennen Netze
        net_measurements=[]
        net_measurements_names=[]
        for c in criteria: 
            c: str=c
            if c.find(net)!=-1:
                langel=str(c).find("(")
                
                if langel==-1:
                    if c==net:
                        net_measurements.append(c)
                        net_measurements_names.append(
                            csv_helpers.single_row_prefix
                        )
                else:
                    if c[langel+1:-1]==net:
                        net_measurements.append(c)
                        net_measurements_names.append(c[0:langel])

        results=[]
        for i in range(len(measurements[c])):
            single_row=[]
            for c in net_measurements:
                single_row.append(measurements[c][i])

            results.append(single_row)

        directory2=os.path.join(directory,net)
        if not os.path.isdir(directory2): os.mkdir(directory2)

        target_path=os.path.join(directory2,measured_property+"_"+target+"_"+toolkit+"_"+mode+"_"+time_stamp+".csv")
        writeFile(target_path,net_measurements_names,results)

config_list="confs.txt"
measurments_conf="measurements_conf"

def writeConfig(file_name):
    """
    write numbers of iterations used for measurements into a seperate file,
    so we know how much iterations we used.  
    """
    open_option="w"
    config_list_path=os.path.join(measurments_conf,config_list)

    if not os.path.isdir(measurments_conf): os.mkdir(measurments_conf)

    if Path(config_list_path).exists():
        open_option="a"
    outp=[
            file_name,
            "iterations: "+str(iterations),
            "iterations_single: "+str(iterations_single),
            "global_iterations: "+str(global_iterations),
            ""
        ]
    outp2=map(lambda s:s+"\n",outp)
    with open(config_list_path,open_option) as file:
            file.writelines(outp2)


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

def getDStr(t:datetime)->str:
    #return str(t.hour)+":"+str(t.minute)+":"+str(t.second)+"."+str(t.microsecond)
    return t.isoformat(sep=" ")

