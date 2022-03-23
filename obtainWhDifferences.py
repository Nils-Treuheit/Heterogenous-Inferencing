import csv
from datetime import datetime
from datetime import timedelta

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
    i=0
    for start in start_measurements:

        start_str:str=getDt(start).isoformat(sep=" ")
        start_t=start_str.split(" ")[1]
        #os.system("ffmpeg -i "+video+" -ss "+start_t+" -vframes 1 "+target+"_start_"+i+".jpg")
        print("ffmpeg -i "+video+" -ss "+start_t+" -vframes 1 "+target+"_start_"+str(i)+".jpg")
        i+=1
    
    i=0
    for end in end_measurements:
        end_str:str=getDt(end).isoformat(sep=" ")
        end_t=end_str.split(" ")[1]
        #os.system("ffmpeg -i "+video+" -ss "+end_t+" -vframes 1 "+target+"_end_"+i+".jpg")
        print("ffmpeg -i "+video+" -ss "+end_t+" -vframes 1 "+target+"_end_"+str(i)+".jpg")
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
        

start1,end1=getTimeStamps(
    "Edge_TPU-Measurements/few_conv2d/energy-times_coral_coral_sync_2022-03-23_18-15.csv",
    readTime()
)
extractFrames("video.mp4","folder/net",start1,end1)