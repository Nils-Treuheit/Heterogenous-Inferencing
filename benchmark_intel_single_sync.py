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

#for target in ["GPU","CPU",]:#"GPU","CPU",,"CPU","MYRIAD"
#
#    measurements=dict()
#    measurements2=dict()
#    for name in nets_to_run:
#        measurements["newData("+name+")"]=[]
#        measurements["noData("+name+")"]=[]
#        measurements2[name]=[]
#
#    for l in range(global_iterations):
#        for i in range(len(nets_to_run)):
#            loaded_net=common.startOpenvinoNet(nets_to_run[i],infCore,target)
#            #network_input="input_1"
#            network_input=next(iter(loaded_net.input_info))
#
#            data_format=[common.iterations_single]
#            data_format.extend(loaded_net.input_info[network_input].tensor_desc.dims)
#            
#            data=common.getOpenvinoExampelData(data_format)
#            #supply new data with every inference
#            for j in range(common.iterations_single):
#                start=time.perf_counter()
#                loaded_net.infer({network_input:data[0]})
#                end=time.perf_counter()
#                measurements["newData("+nets_to_run[i]+")"].append(end-start)
#                start2=time.perf_counter()
#                loaded_net.infer()
#                end2=time.perf_counter()
#                measurements2[nets_to_run[i]].append((end-start)-(end2-start2))
#
#            #do not supply new data with every inference
#            for j in range(common.iterations_single):
#                start=time.perf_counter()
#                loaded_net.infer()
#                end=time.perf_counter()
#                measurements["noData("+nets_to_run[i]+")"].append(end-start)
#
#            data=None
#            gc.collect()
#            
#            print(nets_to_run[i])
#            
#    common.writeResults(target,measurements2,"setData","openvino","sync")
#    common.writeResults(target,measurements,"single","openvino","sync")

for target in ["GPU","CPU","MYRIAD"]:#"GPU","CPU",,"CPU","MYRIAD"
    print(target)
    measurements=dict()
    #measurements2=dict()
    for name in nets_to_run:
        measurements[name]=[]
        #measurements2["min("+name+")"]=[]
        #measurements2["max("+name+")"]=[]
        #measurements2["avg("+name+")"]=[]
        #measurements2["std("+name+")"]=[]

    for l in range(global_iterations):
        print(l)
        for i in range(len(nets_to_run)):

            loaded_net=common.startOpenvinoNet(nets_to_run[i],infCore,target)
            #network_input="input_1"
            network_input=next(iter(loaded_net.input_info))

            data_format=[common.iterations_single]
            data_format.extend(loaded_net.input_info[network_input].tensor_desc.dims)
            
            data=common.getOpenvinoExampelData(data_format)
            #supply new data with every inference
            #min=2147483647
            #max=-1
            #list_measurements=[]
            for j in range(common.iterations_single):
                
                start=time.perf_counter()
                loaded_net.infer({network_input:data[j]})
                end=time.perf_counter()
                
                measured_time=end-start
                measurements[nets_to_run[i]].append(measured_time)
                #if measured_time<min:
                #    min=measured_time
                #if measured_time>max:
                #    max=measured_time
                #
                #list_measurements.append(measured_time)
                
            
            #measurements2["min("+nets_to_run[i]+")"].append(min)
            #measurements2["max("+nets_to_run[i]+")"].append(max)
            #m=np.array(list_measurements)
            #measurements2["std("+nets_to_run[i]+")"].append(np.std(m))
            #measurements2["avg("+nets_to_run[i]+")"].append(np.average(m))


            data=None
            gc.collect()
            
            print(nets_to_run[i])
            
    #common.writeResults(target,measurements2,"statistic","openvino","sync")
    common.writeResults(target,measurements,"single","openvino","sync")
