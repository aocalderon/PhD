#!/bin/bash
HADOOP_CLASSPATH=$(hadoop classpath) 
hdfs dfs -rm -R ${3}
javac -classpath $HADOOP_CLASSPATH ${1}.java
jar -cvf ${1}.jar ${1}*
hadoop jar ${1}.jar ${1} ${2} ${3} 
hdfs dfs -cat ${3}final/part-r-00000
