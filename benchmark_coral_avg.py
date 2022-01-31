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

global_iterations=5
net_dir=os.path.join(".","Edge_TPU-Models")
iterations=8
measurements_coral="Edge_TPU-Measurements"

if not os.path.isdir(measurements_coral): os.mkdir(measurements_coral)

results=dict()

for name in tf_net_names:
    results["first("+name+")"]=[]
    results["avgTail("+name+")"]=[]

for l in range(global_iterations):  
    num_nets=len(tf_net_names)
    
    
    for i in range(num_nets):
        if os.uname().sysname=="Linux":
            #interpreter=tflite.Interpreter(model_path=os.path.join(net_dir,tf_net_names[i]+"_edgetpu.tflite"),experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")])
            interpreter=tflite.Interpreter(model_path=os.path.join("TF_Lite-Models",tf_net_names[i]+".tflite"))
        else:
            interpreter=tflite.Interpreter(model_path=os.path.join(net_dir,tf_net_names[i]+"_edgetpu.tflite"),experimental_delegates=[tflite.load_delegate("edgetpu.dll")])

        time_measurements=[]
        #tflite_res=[]
        shape2=[iterations] # ich denke es macht sinn die selben Iterationen wie bei openvino zu verwenden 
        shape2.extend(interpreter.get_input_details()[0]['shape'])
        data4TPU=np.random.randint(-127,128,shape2,dtype=np.int8)
        output_tensor=interpreter.get_output_details()[0]['index']
        input_tensor=interpreter.get_input_details()[0]['index']

        interpreter.allocate_tensors() #mitmessen?
        
        start_first=time.perf_counter()
        interpreter.set_tensor(input_tensor,value=data4TPU[0])
        interpreter.invoke()
        interpreter.get_tensor(output_tensor)
        end_first=time.perf_counter()
        #print(end_first-start_first)

        results["first("+tf_net_names[i]+")"].append(end_first-start_first)

        start=time.perf_counter() #measure time somewhere here begin
        for j in range(1,iterations):
            interpreter.set_tensor(input_tensor,value=data4TPU[j])
            interpreter.invoke()
            interpreter.get_tensor(output_tensor)
            
        end=time.perf_counter() #measure time somewhere here stop
        #print(end-start)
        results["avgTail("+tf_net_names[i]+")"].append((end-start)/(iterations-1))
        print(tf_net_names[i])

    #write to csv:
target_path=os.path.join(measurements_coral,"avg_all_coral.csv")
if not Path(target_path).exists():
    with open(target_path,"w") as file:
        wr=csv.writer(file)
        wr.writerow(list(results))    
with open(target_path,"a") as file:
    wr=csv.writer(file)
    wr.writerows([[results[name][k] for name in list(results)] for k in range(global_iterations)])
