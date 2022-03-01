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

for target in ["GPU","CPU"]:#"GPU","CPU",,"MYRIAD"

    measurements=dict()
    shapes=dict()
    input_names=dict()
    for name in nets_to_run:
        measurements[name]=[]
        loaded_net=common.startOpenvinoNet(name,infCore,target)
        network_input=next(iter(loaded_net.input_info))
        data_format=loaded_net.input_info[network_input].tensor_desc.dims
        shapes[name]=data_format
        input_names[name]=network_input
        loaded_net=None
    gc.collect()

    for l in range(common.iterations_single):
        for i in range(len(nets_to_run)):
            input_name=input_names[nets_to_run[i]]
            data=common.getOpenvinoExampelData(shapes[nets_to_run[i]])

            start=time.perf_counter()
            infCore2=IECore()
            loaded_net=common.startOpenvinoNet(nets_to_run[i],infCore2,target)
            loaded_net.infer({input_name:data})
                
            end=time.perf_counter()

            data=None
            gc.collect()
            measurements[nets_to_run[i]].append(end-start)
            
            print(nets_to_run[i])

    common.writeResults(target,measurements,"init","openvino","sync")

