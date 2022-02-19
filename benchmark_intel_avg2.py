import gc
from pathlib import Path
from openvino.inference_engine import IECore
import os.path
import numpy as np
import time
import csv

infCore=IECore()
models_openvino=os.path.join(".","OpenVINO-Models")
measurements_openvino="OpenVINO-Measurements"
if not os.path.isdir(measurements_openvino): os.mkdir(measurements_openvino)
target="GPU"
#target="CPU"
#target="MYRIAD"

#iterations=32
iterations=350
#vorher ca. 10000, dann Speicher bei CPU/GPU ausgelastet
#350-> python braucht 10 GB RAm
global_iterations=32

def startNet(name):
    model_path=os.path.join(models_openvino,name+".xml")
    weights_path=os.path.join(models_openvino,name+".bin")
    n=infCore.read_network(model=model_path,weights=weights_path)
    netw=infCore.load_network(network=n,device_name=target,num_requests=iterations)
    return netw


#np.float16.
#data=np.zeros([512,3,128,128],dtype=np.float16)-np.float16(1)
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
"many_conv2d",

]
syn_inference=False


nets_to_run=tf_net_names #[:12] #memory problems in many_conv2d, at least at the CPU
#openvino_nets=[startNet(x) for x in nets_to_run]

for target in ["GPU","CPU","MYRIAD"]:#"GPU","CPU",

    measurements=dict()
    for name in nets_to_run:
        measurements[name]=[]

    for l in range(global_iterations):
        for i in range(len(nets_to_run)):
            loaded_net=startNet(nets_to_run[i])
            #network_input="input_1"
            network_input=next(iter(loaded_net.input_info))

            #data=(np.random.random([iterations,512,3,128,128])-0.5)*8
            data_format=[iterations]
            data_format.extend(loaded_net.input_info[network_input].tensor_desc.dims)



            #res2=[]
            if syn_inference:
                data=(np.random.random(data_format)-0.5)*8

                #first inference
                start_first=time.perf_counter()
                loaded_net.infer({network_input:data[0]})
                end_first=time.perf_counter()
                measurements["first("+nets_to_run[i]+")"].append(end_first-start_first)

                #measure time start 
                start=time.perf_counter()

                for j in range(iterations):
                    loaded_net.infer({network_input:data[j]})

                #measure time end
                end=time.perf_counter()
                data=None
                gc.collect()
                measurements["avgTail("+nets_to_run[i]+")"].append((end-start)/(iterations-1))
            else:
                res=[]
                data=(np.random.random(data_format)-0.5)*8
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

    if syn_inference:
        target_path=os.path.join(measurements_openvino,"avg_"+target+"_openvino_sync.csv")
        measurements_to_write=[[measurements[name][k] for name in list(measurements)] for k in range(global_iterations)]
    else:
        target_path=os.path.join(measurements_openvino,"avg_"+target+"_openvino_async.csv")
        measurements_to_write=[[measurements[name][k] for name in list(measurements)] for k in range(global_iterations)]

    if not Path(target_path).exists():
        with open(target_path,"w") as file:
            wr=csv.writer(file)
            wr.writerow(list(measurements))
    with open(target_path,"a") as file:
        wr=csv.writer(file)
        wr.writerows(measurements_to_write)

