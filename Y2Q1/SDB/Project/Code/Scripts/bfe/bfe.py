#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  bfe.py
#  
#  Copyright 2014 Omar Ernesto Cabrera Rosero <omarcabrera@udenar.edu.co>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import csv
import scipy.spatial as ss
import math
import time
import copy
import Maximal
import io
import sys

class Flock(Maximal.Disk):
    def __init__(self, disk, timestamp):
        self.disk = disk
        self.id = self.disk.id
        self.center = self.disk.center
        self.members  = self.disk.members
        self.diskFlock = [self.disk.id]
        self.start = timestamp
        self.end = timestamp

    def getDuration(self):
        return (self.end - self.start) + 1

    def __str__(self):
        return "%s %s" % (self.disk.id, self.disk.members)    

class BFEFlock(object):
    """ This class is intanced with epsilon, mu and delta"""
    def __init__(self, epsilon, mu, delta):
        self.epsilon = epsilon
        self.mu = mu
        self.delta = delta
        
    def flocks(maximalDisks, previousFlocks, timestamp, keyFlock, stdin):
        """Receive maximal disks, previous flocks, tiemstamp, key Flock 
        return previos flock and write filename with flocks """
        currentFlocks = []
        for md in maximalDisks:
            for f in previousFlocks:
                inter = len(maximalDisks[md].members.intersection(f.members))
                if(inter>=mu):
                    f1 = copy.copy(f)
                    f1.members = maximalDisks[md].members.intersection(f1.members)
                    f1.diskFlock.append(maximalDisks[md].id)
                    f1.end = timestamp

                    if f1.getDuration() == delta:
                        b = list(f1.members)
                        b.sort()
                        stdin.append('{0}\t{1}\t{2}\t{3}'.format(keyFlock, f1.start, f1.end, b))
                        keyFlock += 1
                        f1.start = timestamp - delta + 1
                        
                    if f1.start > timestamp  - delta:
                        currentFlocks.append(f1)
                                            
            currentFlocks.append(Flock(maximalDisks[md],timestamp))
                    
        previousFlocks = currentFlocks
        return (previousFlocks, keyFlock, stdin)
    
    
    def flockFinder(self, filename):
        global delta
        global mu
        
        Maximal.epsilon = self.epsilon
        mu = Maximal.mu = self.mu
        delta = self.delta
        Maximal.precision = 0.0
        tag = filename.split(".")[0].split("/")[-1][1:]
        
                
        dataset = csv.reader(open(filename, 'r'),delimiter=',')
        # next(dataset)
        
        t1 = time.time()
        
        points = Maximal.pointTimestamp(dataset)
        
        input_size = len(points['0'])
        
        # timestamps = list(map(int,points.keys()))
        # timestamps.sort()
                
        previousFlocks = []
        keyFlock = 1
        diskID = 1
        stdin = []
        
        # for timestamp in timestamps:
        centers, treeCenters = Maximal.disksTimestamp(points, '0')    
        centers_size = len(centers)

            # if centersDiskCompare == 0:
            #     continue
            #print(timestamp, len(centersDiskCompare))
            # maximalDisks, diskID = Maximal.maximalDisksTimestamp(centersDiskCompare, treeCenters,disksTime, timestamp, diskID)
            #print("Maximal",len(maximalDisks))
            # previousFlocks, keyFlock, stdin = BFEFlock.flocks(maximalDisks, previousFlocks, int(timestamp), keyFlock, stdin)
        
        # table = ('flocksBFE')
        # print("Flocks: ",len(stdin))
        # flocks = len(stdin)
        # stdin = '\n'.join(stdin)
        # db = Pdbc.DBConnector()
        # db.createTableFlock(table)
        # db.resetTable(table.format(filename))
        # db.copyToTable(table,io.StringIO(stdin))
        

        t2 = round(time.time()-t1,3)
        print("BFE,{3},{0},{1},{2}".format(tag, centers_size, t2, Maximal.epsilon))
        
        # db.createTableTest()
        # db.insertTest(filename,self.epsilon,mu, delta, t2, flocks, tag)
        
def main():
    # bfe = BFEFlock(100,2,3)
    # bfe.flockFinder('sample_small_EPGS-4799.csv')
    
    bfe = BFEFlock(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
    bfe.flockFinder(str(sys.argv[4]))
    

if __name__ == '__main__':
    main()
