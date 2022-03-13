import csv
from pathlib import Path
import common_benchmark_definitions as common 
import csv_helpers
import numpy as np
"""
It's not really helpful to compute the min max values per iteration ...
"""

for tk in csv_helpers.args_toolkits: #in targets unterscheiden
    for target in csv_helpers.args_target:
        regex="*single*"+target+"*"
        for net in common.tf_net_names:
            path_net=Path(csv_helpers.getNetPath(tk,net))
            files=path_net.glob(regex)
            
            measurements=[]
            for file in files:
                with open(file) as f:
                    rea=csv.reader(f)
                    next(rea) # remove titel
                    for x in rea:
                        measurements.append(float(x[0]))
                        #measurements

            if len(measurements)==0:
                continue
            m=np.array(measurements)
            std=np.std(m)
            avg=np.average(m)
            min=np.min(m)
            max=np.max(m)

            print((net,std,avg,min,max,target))
        