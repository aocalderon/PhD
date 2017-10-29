#!/bin/bash 

APP=$1
 
for i in {1..500} 
do
	WID=`xdotool search --name "MaximalFinder"` 
	xdotool windowfocus $WID 
	xdotool key r 
	sleep 1
done
