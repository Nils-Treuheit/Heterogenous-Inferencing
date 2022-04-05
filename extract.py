import common_benchmark_definitions as common
import sys

# Exampel usage: python3 extract.py TPU coral sync /media/martin/fedora_localhost-live/home/martin/Dokumente/HardAccel/Energie_coral.csv

nets=dict()
kit="coral"
prefix="wh-MYRIAD"
prefix="wh-"
suffix=""
with open(sys.argv[4],"r") as f:
	currentnet=""
	for l in f:
		row=l.split(",")
		
		if row[0].isdecimal() and row[1].isdecimal():
			if not currentnet in nets:
				nets[currentnet]=[]
			nets[currentnet].append(row[3])
		else:
			if row[0]!="":
				currentnet=row[0]


common.writeResults(
	sys.argv[1],#target,
	nets,
	"wh",
	sys.argv[2],#toolkit,
	sys.argv[3]
)
