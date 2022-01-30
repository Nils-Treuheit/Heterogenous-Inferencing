from openvino.inference_engine import IECore
import os.path
import numpy as np
import time
#import benchmark_google_coral

infCore=IECore()
models_openvino=os.path.join(".","OpenVINO-Models")
#target="GPU"
target="CPU"
#target="MYRIAD"

#iterations=32
iterations=8 

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
"many_conv2d",
"few_conv2d",
]
single_net=True
syn_inference=True

if single_net:
    i=9
    loaded_net=startNet(tf_net_names[i])
    #network_input="input_1"
    network_input=next(iter(loaded_net.input_info))

    #data=(np.random.random([iterations,512,3,128,128])-0.5)*8
    data_format=[iterations]
    data_format.extend(loaded_net.input_info[network_input].tensor_desc.dims)

    
    
    #res2=[]
    if syn_inference:
        single_measurements=[]
        data=(np.random.random(data_format)-0.5)*8
        for i in range(iterations):
            #measure time start 
            start=time.perf_counter()
            res=loaded_net.infer({network_input:data[i]})

            #loaded_net.infer({network_input:data[i]})
            #measure time end
            end=time.perf_counter()

            single_measurements.append(end-start)
            #res2.append(res)

    else:
        res=[]
        data=(np.random.random(data_format)-0.5)*8
        #measure time start 
        start=time.perf_counter()
        for i in range(iterations):
            r=loaded_net.start_async(request_id=i,inputs={network_input:data[i]})
            res.append(r)


        #loaded_net.wait() #entweder das oder die for-Schleife:

        for x in res:
            x.wait()
        
        #measure time end
        end=time.perf_counter()
        single_measurement=(end - start)/iterations
    #write to csv
else:
    openvino_nets=[startNet(x) for x in tf_net_names]
    for i in range(len(tf_net_names)):
        loaded_net=openvino_nets[i]
        #network_input="input_1"
        network_input=next(iter(loaded_net.input_info))
    
        #data=(np.random.random([iterations,512,3,128,128])-0.5)*8
        data_format=[iterations]
        data_format.extend(loaded_net.input_info[network_input].tensor_desc.dims)
    
        
        
        #res2=[]
        if syn_inference:
            single_measurements=[]
            data=(np.random.random(data_format)-0.5)*8
            for i in range(iterations):
                #measure time start 
                start=time.perf_counter()
                res=loaded_net.infer({network_input:data[i]})
    
                #loaded_net.infer({network_input:data[i]})
                #measure time end
                end=time.perf_counter()
    
                single_measurements.append(end-start)
                #res2.append(res)
    
        else:
            res=[]
            data=(np.random.random(data_format)-0.5)*8
            #measure time start 
            start=time.perf_counter()
            for i in range(iterations):
                r=loaded_net.start_async(request_id=i,inputs={network_input:data[i]})
                res.append(r)
    
    
            #loaded_net.wait() #entweder das oder die for-Schleife:
    
            for x in res:
                x.wait()
            
            #measure time end
            end=time.perf_counter()
            single_measurement=(end - start)/iterations
