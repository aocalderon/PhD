#!/opt/miniconda3/bin/python

import pandas as pd

dataset = 'B1K_RTree_MBRs'
points = pd.read_csv('{0}.csv'.format(dataset), header=None)

def makeWKT(points):
    wkt = "POLYGON (({0} {1}, {2} {1}, {2} {3}, {0} {3}, {0} {1}))".format(points[1], points[2], points[3], points[4])
    return pd.Series([points[0], wkt])

points.apply(makeWKT, axis=1).to_csv('{0}.wkt'.format(dataset), header=None, index=False)