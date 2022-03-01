import gc
import tflite_runtime.interpreter as tflite
#import tensorflow.lite as tflite
import os
import numpy as np
import time
import common_benchmark_definitions as common

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

global_iterations=32
net_dir=os.path.join(".","Edge_TPU-Models")
iterations=350

results=dict()

input_tensors=dict()
output_tensors=dict()
shapes=dict()

for name in tf_net_names:
    results[name]=[]
    if os.uname().sysname=="Linux":
            #interpreter=tflite.Interpreter(model_path=os.path.join(net_dir,name+"_edgetpu.tflite"),experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")])
        interpreter=tflite.Interpreter(model_path=os.path.join("TF_Lite-Models",name+".tflite"))
    else:
        interpreter=tflite.Interpreter(model_path=os.path.join(net_dir,name+"_edgetpu.tflite"),experimental_delegates=[tflite.load_delegate("edgetpu.dll")])
    output_tensor=interpreter.get_output_details()[0]['index']
    input_tensor=interpreter.get_input_details()[0]['index']
    input_tensors[name]=input_tensor
    output_tensors[name]=output_tensor
    shapes[name]=interpreter.get_input_details()[0]['shape']
for l in range(common.iterations_single):  
    num_nets=len(tf_net_names)
    
    
    for i in range(num_nets):
        #data4TPU=np.random.randint(-128,128,shapes[tf_net_names[i]],dtype=np.int8)
        data4TPU=np.random.uniform(np.finfo(np.half).min,np.finfo(np.half).max,shapes[tf_net_names[i]]).astype(np.float32)
        #print(shapes[tf_net_names[i]])
        output_tensor=input_tensors[tf_net_names[i]]
        #input_tensor=output_tensors[tf_net_names[i]]
        input_tensor=0
        #print(output_tensor)
        start=time.perf_counter()

        if os.uname().sysname=="Linux":
            #interpreter=tflite.Interpreter(model_path=os.path.join(net_dir,tf_net_names[i]+"_edgetpu.tflite"),experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")])
            interpreter=tflite.Interpreter(model_path=os.path.join("TF_Lite-Models",tf_net_names[i]+".tflite"))
        else:
            interpreter=tflite.Interpreter(model_path=os.path.join(net_dir,tf_net_names[i]+"_edgetpu.tflite"),experimental_delegates=[tflite.load_delegate("edgetpu.dll")])
        #input_tensor=interpreter.get_input_details()[0]['index']
        interpreter.allocate_tensors() 
        interpreter.set_tensor(input_tensor,value=data4TPU)
        interpreter.invoke()
        interpreter.get_tensor(output_tensor)

        end=time.perf_counter()

        results[tf_net_names[i]].append(end-start)
        
        data4TPU=None
        gc.collect()
        print(tf_net_names[i])

    #write to csv:
common.writeResults("coral",results,"init","coral","sync")

