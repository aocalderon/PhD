#!/bin/bash
echo "Reading files..."
more 2006.txt >> data.txt
more 2007.txt >> data.txt
more 2008.txt >> data.txt
more 2009.txt >> data.txt
echo "Extracting columns 1, 3 and 4..."
awk 'BEGIN{FS="[ ]+"};{print $1","$3","$4}' data.txt > data.csv
echo "Removing extra headers..."
sed '/STN---/d' data.csv > data2.csv
rm -f data.txt data.csv
mv data2.csv data.csv
echo "Number of lines..."
wc data.csv
echo "Done!!!"

