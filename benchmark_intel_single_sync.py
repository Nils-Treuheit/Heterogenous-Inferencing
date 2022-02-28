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

for target in ["GPU"]:#"GPU","CPU",,"CPU","MYRIAD"

    measurements=dict()
    for name in nets_to_run:
        measurements["newData("+name+")"]=[]
        measurements["noData("+name+")"]=[]

    for l in range(global_iterations):
        for i in range(len(nets_to_run)):
            loaded_net=common.startOpenvinoNet(nets_to_run[i],infCore,target)
            #network_input="input_1"
            network_input=next(iter(loaded_net.input_info))

            data_format=[common.iterations_single]
            data_format.extend(loaded_net.input_info[network_input].tensor_desc.dims)
            
            data=common.getOpenvinoExampelData(data_format)
            #supply new data with every inference
            for j in range(common.iterations_single):
                start=time.perf_counter()
                loaded_net.infer({network_input:data[0]})
                end=time.perf_counter()
                measurements["newData("+nets_to_run[i]+")"].append(end-start)

            #do not supply new data with every inference
            for j in range(common.iterations_single):
                start=time.perf_counter()
                loaded_net.infer()
                end=time.perf_counter()
                measurements["noData("+nets_to_run[i]+")"].append(end-start)

            data=None
            gc.collect()
            
            print(nets_to_run[i])
            
    common.writeResults(target,measurements,"single","openvino","sync")

