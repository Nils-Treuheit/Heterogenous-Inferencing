import gc
from pathlib import Path
from openvino.inference_engine import IECore
import os.path
import numpy as np
import time
import csv
import common_benchmark_definitions as common

infCore=IECore()
measurements_openvino="OpenVINO-Measurements"
if not os.path.isdir(measurements_openvino): os.mkdir(measurements_openvino)

iterations=common.iterations
#350-> python braucht 10 GB RAm
global_iterations=common.global_iterations


nets_to_run=common.tf_net_names #[:12] #memory problems in many_conv2d, at least at the CPU
#openvino_nets=[startNet(x) for x in nets_to_run]

for target in ["GPU"]:#,"CPU","MYRIAD"]:#"GPU","CPU",

    measurements=dict()
    for name in nets_to_run:
        measurements["first("+name+")"]=[]
        measurements["avgTail("+name+")"]=[]

    for l in range(global_iterations):
        for i in range(len(nets_to_run)):
            loaded_net=common.startOpenvinoNet(nets_to_run[i],infCore,target)
            #network_input="input_1"
            network_input=next(iter(loaded_net.input_info))

            data_format=[iterations]
            data_format.extend(loaded_net.input_info[network_input].tensor_desc.dims)

            #res2=[]
            
            data=common.getOpenvinoExampelData(data_format)
            #first inference
            start_first=time.perf_counter()
            loaded_net.infer({network_input:data[0]})
            end_first=time.perf_counter()
            measurements["first("+nets_to_run[i]+")"].append(end_first-start_first)
            #measure time start 
            start=time.perf_counter()
            for j in range(iterations):
                loaded_net.infer({network_input:data[j]})
                # get the request, measure time https://github.com/openvinotoolkit/openvino/blob/master/tools/benchmark_tool/openvino/tools/benchmark/benchmark.py
            #measure time end
            end=time.perf_counter()
            data=None
            gc.collect()
            measurements["avgTail("+nets_to_run[i]+")"].append((end-start)/(iterations-1))
            
            print(nets_to_run[i])

    common.writeResults(target,measurements,"avg","openvino","sync")

