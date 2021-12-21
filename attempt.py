import os
import numpy as np
import time

weights=np.random.random(size=100000000)+0.0001

def operation(x): 
    result=weights*x
    return result

with open("conf","r") as config:
    iterations=config.readline()
    iterations=int(iterations)
    end_time=[]
    start_time=[]
    delta_time=[]
    delta_power=0
    with open("/sys/class/power_supply/BAT0/charge_now","r") as power:
        delta_power=int(power.readline())

    for x in range(iterations):
        start_time.insert(x,time.time_ns())
        operation(2)
        end_time.insert(x,time.time_ns())

    with open("/sys/class/power_supply/BAT0/charge_now","r") as power:
        delta_power-=int(power.readline())

for x in range(iterations):
    delta_time.append(end_time[x]-start_time[x])

average = np.average(delta_time)



with open(os.path.join("Benchmark_Results","result"),"w") as resfile:
    resfile.write('BEGINN BENCHMARK \n')
    lines=[]
    for x in range(iterations):
        lines.append("    "+str(delta_time[x])+"\n")
    resfile.writelines(lines)
    if delta_power >= 0:
        resfile.write('power diff: '+str(delta_power))

    resfile.write("\n avg: "+str(average))
    resfile.write('\nENDE BENCHMARK\n\n\n\n')

