#!/bin/bash

xvfb-run --server-args="-screen 0, 1680x1050x24" cutycapt --min-width=1680 --min-height=1050 --delay=5000 --url=file:///home/and/Documents/PhD/Code/Y2Q2/Research/notebooks/${1}.html --out=${1}.png
