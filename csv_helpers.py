"""
Dokument fuer Definitionen fuer csv-Dateien, sodass Lesen und Schreiben leichter werden
"""

import os


measurements_openvino="OpenVINO-Measurements"
measurements_coral="Edge_TPU-Measurements"
def getTimeStamp(d,dt):
    """
    Gibt einen Zeitstempel zurück (damit wir gleiches Format verwenden).
    d - date bzw. gewünschtes Datum
    dt - datetime bzw. gewünschter Zeitpunkt
    """
    return str(d)+"_"+str(dt.hour)+"-"+str(dt.minute)

args_toolkits=["openvino","coral"] #vllt. durch Enums besser
args_mode=["sync","async"]
args_target=["GPU","CPU","MYRIAD","coral"]
args_measured_property= ["init","avg","single","energy-times","statistic"]

single_row_prefix="time"
prefix_property_mapping=dict() #Praefixe, die die Tabellenkoepfe haben koennen (vllt. Netznamen weglassen)
prefix_property_mapping["init"]=[single_row_prefix]
prefix_property_mapping["avg"]=["first","avgTail"]
prefix_property_mapping["single"]=[single_row_prefix]
prefix_property_mapping["energy-times"]=["end","start"]
prefix_property_mapping["statistic"]=["min","max","avg","std"]

def getNetPath(toolkit,net):
    if toolkit=="openvino":
        directory=measurements_openvino
    else:
        directory=measurements_coral
    return os.path.join(directory,net)

def getTagetPath(toolkit,net,mode,target,measured_property,timestamp):
    """
    Argumente:
    toolkit - "openvino" oder "coral" 
    net - der Name des Netzes
    mode - "sync" oder "async" 
    target - "GPU","CPU","MYRIAD","coral"
    measured_property - "init","avg","single","energy"
    timestamp - Zeitpunkt der Messung, als String, generiert durch getTimeStamp(d,dt)
    """
    return os.path.join(getNetPath(toolkit,net),measured_property+"_"+target+"_"+toolkit+"_"+mode+"_"+timestamp+".csv")