
1. Extract spark-2.1.0-bin-hadoop2.7.tar.gz on some place on disk...

i.e.
tar -zxvf spark-2.1.0-bin-hadoop2.7.tar.gz /opt/Spark/spark-2.1.0-bin-hadoop2.7

2. Sync that folder in the other machines in the cluster...

i.e. 
rsync -avz -i ~/.ssh/PRIVATE_KEY_NODE1 -e 'ssh' /opt/Spark/spark-2.1.0-bin-hadoop2.7 IP_NODE1:/opt/Spark/spark-2.1.0-bin-hadoop2.7
rsync -avz -i ~/.ssh/PRIVATE_KEY_NODE2 -e 'ssh' /opt/Spark/spark-2.1.0-bin-hadoop2.7 IP_NODE2:/opt/Spark/spark-2.1.0-bin-hadoop2.7
rsync -avz -i ~/.ssh/PRIVATE_KEY_NODE3 -e 'ssh' /opt/Spark/spark-2.1.0-bin-hadoop2.7 IP_NODE3:/opt/Spark/spark-2.1.0-bin-hadoop2.7
rsync -avz -i ~/.ssh/PRIVATE_KEY_NODE4 -e 'ssh' /opt/Spark/spark-2.1.0-bin-hadoop2.7 IP_NODE4:/opt/Spark/spark-2.1.0-bin-hadoop2.7

3. Set SPARK_HOME as environment variable...

i.e. 
export SPARK_HOME="/home/acald013/Spark/spark-2.1.0-bin-hadoop2.7"

4. Extract datasets.tar.gz on some place on disk...

i.e.
tar -zxvf datasets.tar.gz /home/acald013/Datasets/

5. Sync that folder in the other machines in the cluster...

i.e. 
rsync -avz -i ~/.ssh/PRIVATE_KEY_NODE1 -e 'ssh' /home/acald013/Datasets/ IP_NODE1:/home/acald013/Datasets/
rsync -avz -i ~/.ssh/PRIVATE_KEY_NODE2 -e 'ssh' /home/acald013/Datasets/ IP_NODE2:/home/acald013/Datasets/
rsync -avz -i ~/.ssh/PRIVATE_KEY_NODE3 -e 'ssh' /home/acald013/Datasets/ IP_NODE3:/home/acald013/Datasets/
rsync -avz -i ~/.ssh/PRIVATE_KEY_NODE4 -e 'ssh' /home/acald013/Datasets/ IP_NODE4:/home/acald013/Datasets/

6. Set DATA_HOME as environment variable...

i.e. 
export DATA_HOME="/home/acald013/Datasets/"

7. In file runTester.sh, set the location of the PFLOCK_JAR variable to pflock_2.11-2.0.jar file (it should be the same folder that runTester.sh)...

8. In file runTester2.sh, set the varibles CORES_PER_NODE, MASTER (Master node of the cluster), 
NODE1,NODE2,NODE3,NODE4 (IP addresses of the workers) and N (Number of runs for each experiment)...

9. Run the file runTester3.sh as background task saving the output...

i.e.
./runTester3.sh >> output.txt

10. Done.  You can share me that file for plotting...
