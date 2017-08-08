#!/bin/bash
cd $1
echo "[" > a.tmp
cat *.json > b.tmp
sed 's/$/,/g' b.tmp > c.tmp
rm b.tmp
echo "]" > d.tmp

cat *.tmp > data.json
rm *.tmp
cd ..
