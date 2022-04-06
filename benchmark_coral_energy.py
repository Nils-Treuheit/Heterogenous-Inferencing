from datetime import datetime
import gc
import tflite_runtime.interpreter as tflite
#import tensorflow.lite as tflite
import os
import numpy as np
import common_benchmark_definitions as common

tf_net_names=common.tf_net_names

net_dir=os.path.join(".","Edge_TPU-Models")

measurements=dict()
for name in tf_net_names:
    measurements["start("+name+")"]=[]
    measurements["end("+name+")"]=[]

for l in range(common.global_iterations):  
    num_nets=len(tf_net_names)
    print(l)    
    
    for i in range(num_nets):
        interpreter=tflite.Interpreter(model_path=os.path.join(net_dir,tf_net_names[i]+"_int8_edgetpu.tflite"),experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")])
        #if os.uname().sysname=="Linux":
        #    interpreter=tflite.Interpreter(model_path=os.path.join(net_dir,tf_net_names[i]+"_int8_edgetpu.tflite"),experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")])
        #    #interpreter=tflite.Interpreter(model_path=os.path.join("TF_Lite-Models",tf_net_names[i]+".tflite"))
        #else:
        #    interpreter=tflite.Interpreter(model_path=os.path.join(net_dir,tf_net_names[i]+"_edgetpu.tflite"),experimental_delegates=[tflite.load_delegate("edgetpu.dll")])

        #tflite_res=[]
        shape2=[common.iterations] # ich denke es macht sinn die selben Iterationen wie bei openvino zu verwenden 
        shape2.extend(interpreter.get_input_details()[0]['shape'])
        output_tensor=interpreter.get_output_details()[0]['index']
        input_tensor=interpreter.get_input_details()[0]['index']
        interpreter.allocate_tensors() #mitmessen?

        
        data4TPU=np.random.randint(-128,128,shape2,dtype=np.int8)
        #data4TPU=np.random.uniform(np.finfo(np.half).min,np.finfo(np.half).max,shape2).astype(np.float32)
        print("\a")
        start=datetime.now()
        for j in range(common.iterations):
            interpreter.set_tensor(input_tensor,value=data4TPU[j])
            interpreter.invoke()
            interpreter.get_tensor(output_tensor)

        end=datetime.now()
        print("\a")
        measurements["start("+tf_net_names[i]+")"].append(common.getDStr(start))
        measurements["end("+tf_net_names[i]+")"].append(common.getDStr(end))

        data4TPU=None
        gc.collect()
        print(tf_net_names[i])
        
common.writeResults("TPU",measurements,"energy-times","coral","sync")
