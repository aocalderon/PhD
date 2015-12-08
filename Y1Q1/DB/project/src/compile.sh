#!/bin/bash
#Usage: ./compile.sh DataReducer /path/to/locations/ /path/to/recordings/ /path/to/output/
#For example:
#		./compile.sh DataReducer /user/acald013/ /user/acald013/ /user/acald013/output/
HADOOP_CLASSPATH=$(hadoop classpath) 
hdfs dfs -rm -R ${4}
javac -classpath $HADOOP_CLASSPATH ${1}.java
jar -cvf ${1}.jar ${1}*
hadoop jar ${1}.jar ${1} ${2} ${3} ${4} 
hdfs dfs -cat ${4}final/part-r-00000
