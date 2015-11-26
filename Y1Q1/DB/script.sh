javac -classpath /usr/lib/hadoop/hadoop-common.jar:/usr/lib/hadoop-0.20-mapreduce/hadoop-core.jar:/usr/lib/hadoop/lib/commons-cli-1.2.jar WordCount.java
jar -cvf wordcount.jar .
hadoop jar wordcount.jar WordCount /user/acald013/test.txt /user/acald013/output
hdfs dfs -cat /user/acald013/output/part-00000
