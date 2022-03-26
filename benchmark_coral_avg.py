import gc
import tflite_runtime.interpreter as tflite
#import tensorflow.lite as tflite
import os
import numpy as np
import time
import common_benchmark_definitions as common

tf_net_names=common.tf_net_names


net_dir=os.path.join(".","Edge_TPU-Models")

measurements=dict()

#measurements2=dict()
results=dict()

for name in tf_net_names:
        measurements[name]=[]
        results["first("+name+")"]=[]
        results["avgTail("+name+")"]=[]
        #measurements2["min("+name+")"]=[]
        #measurements2["max("+name+")"]=[]
        #measurements2["avg("+name+")"]=[]
        #measurements2["std("+name+")"]=[]
for l in range(common.global_iterations):  
    num_nets=len(tf_net_names)
    
    
    for i in range(num_nets):
        if os.uname().sysname=="Linux":
            interpreter=tflite.Interpreter(model_path=os.path.join(net_dir,tf_net_names[i]+"_int8_edgetpu.tflite"),experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")])
            #interpreter=tflite.Interpreter(model_path=os.path.join("TF_Lite-Models",tf_net_names[i]+".tflite"))
        else:
            interpreter=tflite.Interpreter(model_path=os.path.join(net_dir,tf_net_names[i]+"_int8_edgetpu.tflite"),experimental_delegates=[tflite.load_delegate("edgetpu.dll")])

        time_measurements=[]
        #tflite_res=[]
        shape2=[common.iterations] # ich denke es macht sinn die selben Iterationen wie bei openvino zu verwenden 
        shape2.extend(interpreter.get_input_details()[0]['shape'])
        shape3=[common.iterations_single]
        shape3.extend(interpreter.get_input_details()[0]['shape'])
        data4TPU=np.random.randint(-128,128,shape3,dtype=np.int8)
        #data4TPU=np.random.uniform(np.finfo(np.half).min,np.finfo(np.half).max,shape3).astype(np.float32)
        output_tensor=interpreter.get_output_details()[0]['index']
        input_tensor=interpreter.get_input_details()[0]['index']
        interpreter.allocate_tensors() 

        #min=2147483647
        #max=-1

        #list_measurements=[]
        for j in range(common.iterations_single):
                
            start=time.perf_counter()
            interpreter.set_tensor(input_tensor,value=data4TPU[j])
            interpreter.invoke()
            interpreter.get_tensor(output_tensor)
            end=time.perf_counter()
            
            measured_time=end-start
            measurements[tf_net_names[i]].append(measured_time)
            #if measured_time<min:
            #    min=measured_time
            #if measured_time>max:
            #    max=measured_time
            #
            
            #list_measurements.append(measured_time)
            
            
        #measurements2["min("+tf_net_names[i]+")"].append(min)
        #measurements2["max("+tf_net_names[i]+")"].append(max)
        #m=np.array(list_measurements)
        #measurements2["std("+tf_net_names[i]+")"].append(np.std(m))
        #measurements2["avg("+tf_net_names[i]+")"].append(np.average(m))


        data4TPU=np.random.randint(-128,128,shape2,dtype=np.int8)
        #data4TPU=np.random.uniform(np.finfo(np.half).min,np.finfo(np.half).max,shape2).astype(np.float32)
        
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
common.writeResults("coral",measurements,"single","coral","sync")
#common.writeResults("coral",measurements2,"statistic","coral","sync")
