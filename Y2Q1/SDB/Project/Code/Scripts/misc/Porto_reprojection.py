
# coding: utf-8

# In[23]:

from pyspark.sql import Row
dataset = "/opt/Datasets/Porto/L16M.csv"
n = 16000000


# In[24]:

from pyproj import Proj, transform

def transformCoords(row):
    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:27493')
    x2,y2 = transform(inProj,outProj, row.y, row.x)
    return Row(x=float(x2), y=float(y2))

def toTag(n):
    if n < 1000:
        return n
    elif n < 1000000:
        return "{0}K".format(int(round(n/1000.0,0)))
    elif n < 1000000000:
        return "{0}M".format(int(round(n/1000000.0,0)))
    else:
        return "{0}B".format(int(round(n/1000000000.0,0)))


# In[25]:

points = sc.textFile(dataset).map(lambda line: line.split(",")).map(lambda p: Row(x=float(p[0]), y=float(p[1])))


# In[26]:

points.take(5)


# In[27]:

points = points.map(transformCoords).toDF()
points.show()


# In[28]:

points.write.format('com.databricks.spark.csv').save('/opt/Datasets/Porto/output')
command = "cat /opt/Datasets/Porto/output/part-00* > /opt/Datasets/Porto/P{0}.csv".format(toTag(n))
os.system(command)
os.system("rm -fR /opt/Datasets/Porto/output/")

