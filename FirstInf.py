import time
from openvino.inference_engine import IECore
import os
import numpy as np

def runInferenceOnIntelNeural2(model_name): #https://docs.openvino.ai/latest/openvino_docs_IE_DG_Integrate_with_customer_application_new_API.html?sw_type=switcher-python
    infCore=IECore()
    netwPath=os.path.join(".","Converted_Models",model_name)
    n=infCore.read_network(model=os.path.join(netwPath,"model.xml"),weights=os.path.join(netwPath,"model.bin"))
    netw=infCore.load_network(network=n,device_name="CPU")

    print(netw.infer({(next(iter(n.input_info))):(np.zeros([1,1000,1000],dtype=np.float16)+np.float16(1.0))}))

def runInferenceOnIntelNeural2(model_name): #https://docs.openvino.ai/latest/openvino_docs_IE_DG_Integrate_with_customer_application_new_API.html?sw_type=switcher-python
    infCore=IECore()
    netwPath=os.path.join(".","Converted_Models",model_name)
    n=infCore.read_network(model=os.path.join(netwPath,"model.xml"),weights=os.path.join(netwPath,"model.bin"))
    netw=infCore.load_network(network=n,device_name="MYRIAD")

    print(netw.infer({(next(iter(n.input_info))):(np.zeros([1,1000,1000],dtype=np.float16)+np.float16(1.0))}))

