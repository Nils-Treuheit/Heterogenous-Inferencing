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
#target="GPU"
target="CPU"
#target="MYRIAD"
#
#
#
#
#
#

#iterations=32
iterations=3

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
single_net=False#True
syn_inference=True

if single_net:
    i=13
    loaded_net=startNet(tf_net_names[i])
    print(tf_net_names[i])
    #network_input="input_1"
    network_input=next(iter(loaded_net.input_info))

    #data=(np.random.random([iterations,512,3,128,128])-0.5)*8
    data_format=[iterations]
    data_format.extend(loaded_net.input_info[network_input].tensor_desc.dims)

    
    
    #res2=[]
    if syn_inference:
        single_measurements=[]
        first_measurement=[]
        data=(np.random.random(data_format)-0.5)*8
        for j in range(iterations):
            #measure time start 
            start=time.perf_counter()
            res=loaded_net.infer({network_input:data[j]})

            #loaded_net.infer({network_input:data[i]})
            #measure time end
            end=time.perf_counter()

            if len(single_measurements)==0:
                first_measurement.append([end-start,"first"])
            else:
                single_measurements.append([end-start])
            #res2.append(res)
        print(iter(single_measurements))
        with open(os.path.join(measurements_openvino,target+"_"+tf_net_names[i]+".csv"),"w") as file:
            wr=csv.writer(file)
            wr.writerows(first_measurement)
            wr.writerows(single_measurements)
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
        #write to csv
        single_measurement=(end - start)/iterations
        with open(os.path.join(measurements_openvino,tf_net_names[i]+"_async.csv"),"a") as file:
            wr=csv.writer(file)
            wr.writerows(single_measurement)
    
else:
    nets_to_run=tf_net_names[:12] #memory problems in many_conv2d, at least at the CPU
    #openvino_nets=[startNet(x) for x in nets_to_run]
    measurements=dict()
    
    for i in range(len(nets_to_run)):
        loaded_net=startNet(nets_to_run[i])
        #network_input="input_1"
        network_input=next(iter(loaded_net.input_info))
    
        #data=(np.random.random([iterations,512,3,128,128])-0.5)*8
        data_format=[iterations]
        data_format.extend(loaded_net.input_info[network_input].tensor_desc.dims)
    
        
        
        #res2=[]
        if syn_inference:
            single_measurements=[]
            data=(np.random.random(data_format)-0.5)*8
            for j in range(iterations):
                #measure time start 
                start=time.perf_counter()
                res=loaded_net.infer({network_input:data[j]})
    
                #loaded_net.infer({network_input:data[i]})
                #measure time end
                end=time.perf_counter()
    
                single_measurements.append(end-start)
                #res2.append(res)
            measurements[nets_to_run[i]]=single_measurements
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
            measurements[nets_to_run[i]]=single_measurement
    print(measurements)

    if syn_inference:
        target_path=os.path.join(measurements_openvino,target+"_openvino_sync.csv")
        measurements_to_write=[[measurements[name][k] for name in nets_to_run] for k in range(iterations)]
    else:
        target_path=os.path.join(measurements_openvino,target+"_openvino_async.csv")
        measurements_to_write=[[measurements[name] for name in nets_to_run]]
    
    if not Path(target_path).exists():
        with open(target_path,"w") as file:
            wr=csv.writer(file)
            wr.writerow(nets_to_run)
    with open(target_path,"a") as file:
        wr=csv.writer(file)
        wr.writerow(['' for _ in nets_to_run])
        wr.writerows(measurements_to_write)
