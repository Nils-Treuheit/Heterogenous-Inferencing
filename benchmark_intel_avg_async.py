import gc
from openvino.inference_engine import IECore
import numpy as np
import time
import common_benchmark_definitions as common

infCore=IECore()

iterations=common.iterations
#vorher ca. 10000, dann Speicher bei CPU/GPU ausgelastet
#350-> python braucht 10 GB RAm
global_iterations=common.global_iterations

nets_to_run=common.tf_net_names #[:12] #memory problems in many_conv2d, at least at the CPU

for target in ["GPU","CPU","MYRIAD"]:#"GPU","CPU"

    measurements=dict()
    for name in nets_to_run:
        measurements[name]=[]

    for l in range(global_iterations):
        for i in range(len(nets_to_run)):
            loaded_net=common.startOpenvinoNet(nets_to_run[i],infCore,target)
            #network_input="input_1"
            network_input=next(iter(loaded_net.input_info))

            #data=(np.random.random([iterations,512,3,128,128])-0.5)*8
            data_format=[iterations]
            data_format.extend(loaded_net.input_info[network_input].tensor_desc.dims)

            #res2=[]
            
            res=[]
            data=common.getOpenvinoExampelData(data_format)
            #measure time start 
            start=time.perf_counter()
            for j in range(iterations):
                r=loaded_net.start_async(request_id=j,inputs={network_input:data[j]})
                res.append(r)
            #loaded_net.wait() #entweder das oder die for-Schleife:
            for x in res:
                x.wait()
            #measure time end
            end=time.perf_counter()
            single_measurement=(end - start)/iterations
            measurements[nets_to_run[i]].append(single_measurement)
            data=None
            gc.collect()
            print(nets_to_run[i])

 
    common.writeResults(target,measurements,"avg","openvino","async")

