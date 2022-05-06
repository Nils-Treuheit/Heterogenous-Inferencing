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
    i=0
    for start in start_measurements:
        tmp: datetime=getDt(start)
        if tmp.date()!=datetime(1970,1,1,0,0,0,0).date():
            continue
        start_str:str=tmp.isoformat(sep=" ")
        start_t=start_str.split(" ")[1]
        #os.system("echo \""+start_str+"\" >> /home/martin/Dokumente/HardAccel/debug/debug3.txt")
        #os.system("ffmpeg -i "+video+" -ss "+start_t+" -vframes 1 "+target+"_start_"+i+".jpg")
        os.system("ffmpeg -ss "+start_t+" -i "+video+" -vframes 1 "+target+str(i)+"_begin"+".jpg")
        i+=1
    
    i=0
    for end in end_measurements:
        tmp: datetime=getDt(end)
        if tmp.date()!=datetime(1970,1,1,0,0,0,0).date():
            continue
        end_str:str=tmp.isoformat(sep=" ")
        end_t=end_str.split(" ")[1]
        #os.system("echo \""+end_str+"\" >> /home/martin/Dokumente/HardAccel/debug/debug3.txt")
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
    if tk=="openvino":
        target="MYRIAD"
    else:
        target="TPU"
    
    regex="energy-times_*"+time_stamp+".csv"
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
            path_pictures="/home/martin/Bilder/Messungen/openvino11/"+tk+"_"+name
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

# Video letzte/MYRIAD/VID_20220418_145213.mp4 : 2022-04-18 14:51:50.124389
# Video letzte/MYRIAD/VID_20220419_170752.mp4 : 2022-04-19 17:07:29.735812
# Video letzte/MYRIAD/VID_20220419_061816.mp4 : 2022-04-19 061816
# Video letzte/MYRIAD/VID_20220420_135913.mp4 : 2022-04-20 13:58:53.348414
# Video letzte/MYRIAD/VID_20220420_083357.mp4 : 2022-04-20 10:17:02.605812

# Video TPU/VID_20220417_211431.mp4 : 2022-04-17 21:14:08.333849
# Video TPU/VID_20220418_124036.mp4 : 2022-04-18 124036
#
#
#from datetime import timedelta,datetime
#d=timedelta(hours=17,minutes=21,seconds=4.64)
#a=datetime.fromisoformat("2022-04-18 14:51:50.124389")
#(a+d).isoformat()
#'2022-04-19T08:12:54.764389'






video="/home/martin/Videos/Messungen/letzte/OpenVino/VID_20220420_083357.mp4"
print(video)
#getAllPictures(video,"2022-04-20_8-35","openvino")
getAllPictures(video,"2022-04-20_8-35","openvino")
#2022-04-19 08:12:54.764389