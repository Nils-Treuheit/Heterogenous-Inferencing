import gc
from datetime import datetime
from openvino.inference_engine import IECore
import os.path
from os import system
import common_benchmark_definitions as common

infCore=IECore()
measurements_openvino="OpenVINO-Measurements"
if not os.path.isdir(measurements_openvino): os.mkdir(measurements_openvino)

iterations=common.iterations
#350-> python braucht 10 GB RAm
global_iterations=common.global_iterations


nets_to_run=common.tf_net_names #[:12] #memory problems in many_conv2d, at least at the CPU
#openvino_nets=[startNet(x) for x in nets_to_run]

target="MYRIAD"

measurements=dict()
for name in nets_to_run:
    measurements["start("+name+")"]=[]
    measurements["end("+name+")"]=[]
for l in range(global_iterations):
    print(l)
    for i in range(len(nets_to_run)):
        loaded_net=common.startOpenvinoNet(nets_to_run[i],infCore,target)
        #network_input="input_1"
        network_input=next(iter(loaded_net.input_info))
        data_format=[iterations]
        data_format.extend(loaded_net.input_info[network_input].tensor_desc.dims)
        #res2=[]
        
        data=common.getOpenvinoExampelData(data_format)
        
        #input("start")
        #encl=common.EnergyMeasurementEnclosure(nets_to_run[i])
        #encl.start()
        print("\a")
        start=datetime.now()
        for j in range(iterations):
            loaded_net.infer({network_input:data[j]})
            # get the request, measure time https://github.com/openvinotoolkit/openvino/blob/master/tools/benchmark_tool/openvino/tools/benchmark/benchmark.py
        
        #encl.end()
        end=datetime.now()
        print("\a")
        
        print(nets_to_run[i])
        measurements["start("+nets_to_run[i]+")"].append(common.getDStr(start))
        measurements["end("+nets_to_run[i]+")"].append(common.getDStr(end))
        #input("end")
        data=None
        gc.collect()

print(measurements)
common.writeResults("MYRIAD",measurements,"energy-times","openvino","sync")
