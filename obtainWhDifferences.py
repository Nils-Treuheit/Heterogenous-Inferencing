import csv
from datetime import datetime
from datetime import timedelta
from io import IOBase
import common_benchmark_definitions as common 
from pathlib import Path
import csv_helpers
import os



def getRelTime(offset: datetime,measured_time: str)->datetime:
    #str(t.hour)+":"+str(t.minute)+":"+str(t.second)+"."+str(t.microsecond)
    t: datetime=datetime.fromisoformat(measured_time)
    return offset-t

def readTime()->datetime:
    print(datetime.now())
    while True:
        t_str=input("Time when the video starts: ")
        t=datetime.fromisoformat(t_str)
        print(t)
        if input("Is that correct?[y]")=="y":
            break
    return t



def getTimeStamps(csv_file,offset):
    start_measurements=[]
    end_measurements=[]
    with open(csv_file) as f:
        r=csv.DictReader(f)
        for times in r:
            start_str=times["start"]
            end_str=times["end"]
            start=datetime.fromisoformat(start_str)-offset
            end=datetime.fromisoformat(end_str)-offset
            start_measurements.append(start)
            end_measurements.append(end)

    return start_measurements,end_measurements

ffmpeg_cmd="ffmpeg -ss 01:23:45 -vframes 1 "

def getDt(x:timedelta)->datetime:
    d=datetime(1970,1,1,0,0,0,0)
    return d+x

def extractFrames(video: str,target: str,start_measurements,end_measurements):
    i=20
    for start in start_measurements:

        start_str:str=getDt(start).isoformat(sep=" ")
        start_t=start_str.split(" ")[1]
        #os.system("ffmpeg -i "+video+" -ss "+start_t+" -vframes 1 "+target+"_start_"+i+".jpg")
        os.system("ffmpeg -ss "+start_t+" -i "+video+" -vframes 1 "+target+str(i)+"_begin"+".jpg")
        i+=1
    
    i=20
    for end in end_measurements:
        end_str:str=getDt(end).isoformat(sep=" ")
        end_t=end_str.split(" ")[1]
        #os.system("ffmpeg -i "+video+" -ss "+end_t+" -vframes 1 "+target+"_end_"+i+".jpg")
        os.system("ffmpeg -ss "+end_t+" -i "+video+" -vframes 1 "+target+str(i)+"_end"+".jpg")
        i+=1

def createWhCsvPerFile(file):
    measurements=[]
    with open(file) as f:
        r=csv.DictReader(f)
        for times in r:
            print("start:"+times["start"])
            startWh=float(input())
            print("end:"+times["end"])
            endWh=float(input())
            measurements.append(startWh-endWh)

    return measurements
        
def getAllPictures(video,time_stamp,tk):
    time_offset=readTime()
    
    for target in csv_helpers.args_target:
        regex="energy-times_*"+target+"*_"+time_stamp+".csv"
        for name in common.tf_net_names:
            path_net=Path(csv_helpers.getNetPath(tk,name))
            if not os.path.isdir(path_net):
                continue
            files=path_net.glob(regex)
            for f in files:
                start2,end2=getTimeStamps(
                    f,
                    time_offset
                )
                path_pictures="/home/martin/Bilder/"+tk+"_"+name
                if not os.path.isdir(path_pictures):
                    os.mkdir(path_pictures)
                    #with open("/home/martin/Bilder/"):
                    #    os.mkdir(path_pictures,dir_fd=IOBase.fileno())
                    #
                extractFrames(
                    video,
                    path_pictures+"/p"+time_stamp,
                    start2,
                    end2
                )

#start1,end1=getTimeStamps(
#    "/home/martin/Dokumente/HardAccel/Version3/Heterogenous-Inferencing/Edge_TPU-Measurements/big_conv2d/energy-times_coral_coral_sync_2022-03-30_19-41.csv",
#    readTime()
#)
#extractFrames(
#    "/home/martin/Videos/20220330_185122.mp4",
#    "pictures/big_conv2d_coral_b",
#    start1,
#    end1
#)
#2022-03-30 18:51:23.001190
getAllPictures("/home/martin/Videos/20220330_222858.mp4","2022-03-30_23-4","openvino")