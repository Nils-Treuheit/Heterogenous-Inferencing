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

results2=dict()

for name in tf_net_names:
    results["first("+name+")"]=[]
    results["avgTail("+name+")"]=[]
    results2["newData("+name+")"]=[]
    results2["noData("+name+")"]=[]
for l in range(common.global_iterations):  
    num_nets=len(tf_net_names)
    
    
    for i in range(num_nets):
        if os.uname().sysname=="Linux":
            interpreter=tflite.Interpreter(model_path=os.path.join(net_dir,tf_net_names[i]+"_edgetpu.tflite"),experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")])
            #interpreter=tflite.Interpreter(model_path=os.path.join("TF_Lite-Models",tf_net_names[i]+".tflite"))
        else:
            interpreter=tflite.Interpreter(model_path=os.path.join(net_dir,tf_net_names[i]+"_edgetpu.tflite"),experimental_delegates=[tflite.load_delegate("edgetpu.dll")])

        time_measurements=[]
        #tflite_res=[]
        shape2=[common.iterations] # ich denke es macht sinn die selben Iterationen wie bei openvino zu verwenden 
        shape2.extend(interpreter.get_input_details()[0]['shape'])
        shape3=[common.iterations_single]
        shape3.extend(interpreter.get_input_details()[0]['shape'])
        data4TPU=np.random.randint(-128,128,shape3,dtype=np.int8)
        output_tensor=interpreter.get_output_details()[0]['index']
        input_tensor=interpreter.get_input_details()[0]['index']
        interpreter.allocate_tensors() #mitmessen?

        for j in range(common.iterations_single):
            start=time.perf_counter()
            interpreter.set_tensor(input_tensor,value=data4TPU[j])
            interpreter.invoke()
            interpreter.get_tensor(output_tensor)
            end=time.perf_counter()
            results["newData("+tf_net_names[i]+")"].append(end-start)


        for j in range(common.iterations_single):
            start=time.perf_counter()
            interpreter.invoke()
            end=time.perf_counter()
            results["noData("+tf_net_names[i]+")"].append(end-start)

        #TODO Zeiten messen
        #TODO min /max
        #TODO Zeiten messen
        #sudo apt-get install libedgetpu1-max?

        ## set and retrieve data
        #start_first=time.perf_counter()
        #interpreter.set_tensor(input_tensor,value=data4TPU[0])
        #interpreter.invoke()
        #interpreter.get_tensor(output_tensor)
        #end_first=time.perf_counter()
#
        ## set data
        #start_first=time.perf_counter()
        #interpreter.set_tensor(input_tensor,value=data4TPU[0])
        #interpreter.invoke()
        #end_first=time.perf_counter()
#
        ## only run inference
        #interpreter.set_tensor(input_tensor,value=data4TPU[0])
        #start_first=time.perf_counter()
        #interpreter.invoke()
        #end_first=time.perf_counter()

        data4TPU=np.random.randint(-128,128,shape2,dtype=np.int8)
        
        start_first=time.perf_counter()
        interpreter.set_tensor(input_tensor,value=data4TPU[0])
        interpreter.invoke()
        interpreter.get_tensor(output_tensor)
        end_first=time.perf_counter()
        #print(end_first-start_first)

        results["first("+tf_net_names[i]+")"].append(end_first-start_first)

        start=time.perf_counter() #measure time somewhere here begin
        for j in range(1,common.iterations):
            interpreter.set_tensor(input_tensor,value=data4TPU[j])
            interpreter.invoke()
            interpreter.get_tensor(output_tensor)
            
        end=time.perf_counter() #measure time somewhere here stop
        #print(end-start)
        results["avgTail("+tf_net_names[i]+")"].append((end-start)/(common.iterations-1))
        data4TPU=None
        gc.collect()
        print(tf_net_names[i])

    #write to csv:
common.writeResults("coral",results,"avg","coral","sync")
common.writeResults("coral",results2,"single","coral","sync")
