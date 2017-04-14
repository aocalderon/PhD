#!/opt/miniconda3/bin/python

import pandas as pd

path = '/opt/Datasets/Beijing'
input = 'P10K.csv'
output = 'B1K.csv'
N = 1000

points = pd.read_csv('{0}/{1}'.format(path, input), header=None)
points.sample(n=N).to_csv('{0}//{1}'.format(path, output), header=None, index=False)



