import csv
import os
from pathlib import Path
import common_benchmark_definitions as common 
import csv_helpers
import numpy as np
"""
It's not really helpful to compute the min max values per iteration ...
"""

def getAllMeasurements(measured_property,mode="")->dict:
    """
    Returns a Dictionary.
    results[<toolkit>][<target>][<network>][<value from prefix_property_mapping>]=
    array(<single data points as float values>)

    please consider that results[<toolkit>][<target>][<network>]
    can return an empty dictonary if we haven't yet measured the desired metrics so that no 
    file matches the given regex in the directory of the network.
    """ # It makes sense if I translate the remaining documentation like in csv_helpers to English, doesn't it?

    files_read=[]
    results=dict()
    for tk in csv_helpers.args_toolkits: #in targets unterscheiden
        results[tk]=dict()
        for target in csv_helpers.args_target:
            regex=measured_property+"*"+target+"*"+mode
            results[tk][target]=dict()
            for name in common.tf_net_names:
                results[tk][target][name]=dict()
                path_net=Path(csv_helpers.getNetPath(tk,name))
                if not os.path.isdir(path_net):
                    continue
                files=path_net.glob(regex)

                for file in files:
                    with open(file) as f:
                        files_read.append(str(file))
                        rea=csv.DictReader(f)
                        column_names=list(rea.fieldnames)
                        for column_name in column_names:
                            results[tk][target][name][column_name]=[]
                        for x in rea:
                            for column_name in column_names:
                                data_point=float(x[column_name])
                                results[tk][target][name][column_name].append(data_point)

    return results,files_read

                
#print(getAllMeasurements("single"))

def getArray(results,toolkit,target,net,property_name):
    ar=results[0][toolkit][target][net][property_name]
    return ar
    
def StdAvgMinMax(toolkit,target,net):
    res=getAllMeasurements("single")

    ar=getArray(res,toolkit,target,net,csv_helpers.single_row_prefix)
    np_ar=np.array(ar)
    
    std=np.std(np_ar)
    avg=np.average(np_ar)
    min=np.min(np_ar)
    max=np.max(np_ar)

    return std,avg,min,max

print(StdAvgMinMax("openvino","CPU","relu_act"))
#for tk in csv_helpers.args_toolkits: #in targets unterscheiden
#    for target in csv_helpers.args_target:
#        regex="single*"+target+"*"
#        for net in common.tf_net_names:
#            path_net=Path(csv_helpers.getNetPath(tk,net))
#            files=path_net.glob(regex)
#            
#            measurements=[]
#            for file in files:
#                with open(file) as f:
#                    rea=csv.reader(f)
#                    next(rea) # remove titel
#                    for x in rea:
#                        measurements.append(float(x[0]))
#                        #measurements
#
#            if len(measurements)==0:
#                continue
#            m=np.array(measurements)
#            std=np.std(m)
#            avg=np.average(m)
#            min=np.min(m)
#            max=np.max(m)
#
#            print((net,std,avg,min,max,target))
        