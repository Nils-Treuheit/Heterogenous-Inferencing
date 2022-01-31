#import tflite_runtime.interpreter as tflite
import csv
import tensorflow.lite as tflite
import os
import numpy as np
import time
from pathlib import Path

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
net_dir=os.path.join(".","Edge_TPU-Models")
iterations=512
measurements_coral="Edge_TPU-Measurements"

if not os.path.isdir(measurements_coral): os.mkdir(measurements_coral)

if single_net:
    i=6
    if os.uname().sysname=="Linux":
        #interpreter=tflite.Interpreter(model_path=os.path.join(net_dir,tf_net_names[i]+"_edgetpu.tflite"),experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")])
        interpreter=tflite.Interpreter(model_path=os.path.join(net_dir,tf_net_names[i]))
        #interpreter=tflite.Interpreter(tflite_models[i])#,experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")])
    else:
        interpreter=tflite.Interpreter(model_path=os.path.join(net_dir,tf_net_names[i]+"_edgetpu.tflite"),experimental_delegates=[tflite.load_delegate("edgetpu.dll")])

    
    #tflite_res=[]
    shape2=[iterations] # ich denke es macht sinn die selben Iterationen wie bei openvino zu verwenden 
    shape2.extend(interpreter.get_input_details()[0]['shape'])
    data4TPU=np.random.randint(-127,128,shape2,dtype=np.int8)
    output_tensor=interpreter.get_output_details()[0]['index']
    input_tensor=interpreter.get_input_details()[0]['index']
    measurements_single=[]
    for j in range(iterations):
        interpreter.allocate_tensors() 
        #print(interpreter.tensor(0))
        
        start=time.perf_counter() #measure time somewhere here begin
        
        interpreter.set_tensor(input_tensor,value=data4TPU[j])
        interpreter.invoke()
        tmp_res=interpreter.get_tensor(output_tensor)

        end=time.perf_counter() #measure time somewhere here stop
        
        measurements_single.append(end-start)
        #tflite_res.append(tmp_res)

    # Ueberpruefen:
    #for j in range(iterations):
    #    print(data4TPU[j][0][0])
    #    print(tflite_res[j][0][0])

    #write to csv
    with open(os.path.join(measurements_coral,tf_net_names[i]+"_coral.csv"),"w") as file:
        wr=csv.writer(file)
        wr.writerows(measurements_single)
        #wr.writerows([[measurements[name][k] for name in nets_to_run] for k in range(iterations)])

else:   #Iterate over all networks
    num_nets=len(tf_net_names)
    
    results=dict()
    for i in range(num_nets):

        if os.uname().sysname=="Linux":
            interpreter=tflite.Interpreter(model_path=os.path.join(net_dir,tf_net_names[i]+"_edgetpu.tflite"),experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")])
        else:
            interpreter=tflite.Interpreter(model_path=os.path.join(net_dir,tf_net_names[i]+"_edgetpu.tflite"),experimental_delegates=[tflite.load_delegate("edgetpu.dll")])

        time_measurements=[]
        #tflite_res=[]
        shape2=[iterations] # ich denke es macht sinn die selben Iterationen wie bei openvino zu verwenden 
        shape2.extend(interpreter.get_input_details()[0]['shape'])
        data4TPU=np.random.randint(-127,128,shape2,dtype=np.int8)
        output_tensor=interpreter.get_output_details()[0]['index']
        input_tensor=interpreter.get_input_details()[0]['index']
        for j in range(iterations):
            interpreter.allocate_tensors()
            #print(interpreter.tensor(0))
            
            start=time.perf_counter() #measure time somewhere here begin
            
            interpreter.set_tensor(input_tensor,value=data4TPU[j])
            interpreter.invoke()
            tmp_res=interpreter.get_tensor(output_tensor)
            
            end=time.perf_counter() #measure time somewhere here stop

            time_measurements.append(end-start)
            #tflite_res.append(tmp_res)

        results[tf_net_names[i]]=time_measurements
        print(tf_net_names[i])

    #write to csv:
    target_path=os.path.join(measurements_coral,"all_coral.csv")
    if not Path(target_path).exists():
        with open(target_path,"w") as file:
            wr=csv.writer(file)
            wr.writerows(tf_net_names)    
    with open(target_path,"a") as file:
        wr=csv.writer(file)
        wr.writerow([' ' for k in tf_net_names])
        wr.writerows([[results[name][k] for name in tf_net_names] for k in range(iterations)])
