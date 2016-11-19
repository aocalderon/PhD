
# coding: utf-8

# In[64]:

from pyspark.sql.types import *
from pyspark.sql import Row

def toTag(n):
    if n < 1000:
        return n
    elif n < 1000000:
        return "{0}K".format(int(round(n/1000.0,0)))
    elif n < 1000000000:
        return "{0}M".format(int(round(n/1000000.0,0)))
    else:
        return "{0}B".format(int(round(n/1000000000.0,0)))

df = sqlContext.read.format('com.databricks.spark.csv')    .options(header='true')    .load('file:///opt/GISData/TaxiPorto/train.csv')


# In[65]:

points = df.select('POLYLINE').flatMap(lambda row: row[0][2:-2].split("],[")).distinct()
points.cache()


# In[66]:

points.count()
# 83415287
# 17722273 without duplicates...


# In[85]:

schema = StructType([StructField('coord', StringType(), True)])
x = 16002000


# In[86]:

percentage = x / 17722273.0
sample = points.sample(False, percentage, 42)
n = sample.count()
print(n)
sample.map(lambda x: Row(x)).toDF(schema).write    .format('com.databricks.spark.csv')    .option('quote', None)    .save('output')


# In[87]:

command = "cat output/part-00* > L{0}.csv".format(toTag(n))
os.system(command)
os.system("rm -fR output/")


# In[ ]:



