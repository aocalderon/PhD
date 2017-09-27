trajs = read.csv('Berlin_Trajs.csv', header=F)
names(trajs) = c('tid', 'oid', 'x', 'y')
B80K = trajs[trajs$tid == 624,]
B90K = trajs[trajs$tid == 658,]
B100K = trajs[trajs$tid == 692,]
B110K = trajs[trajs$tid == 726,]
B120K = trajs[trajs$tid == 760,]
B130K = trajs[trajs$tid == 794,]
B140K = trajs[trajs$tid == 829,]
B150K = trajs[trajs$tid == 863,]
B160K = trajs[trajs$tid == 899,]
B170K = trajs[trajs$tid == 934,]
B180K = trajs[trajs$tid == 970,]
B80K$oid = str_sub(B80K$oid, 4)
B90K$oid = str_sub(B90K$oid, 4)
B100K$oid = str_sub(B100K$oid, 4)
B110K$oid = str_sub(B110K$oid, 4)
B120K$oid = str_sub(B120K$oid, 4)
B130K$oid = str_sub(B130K$oid, 4)
B140K$oid = str_sub(B140K$oid, 4)
B150K$oid = str_sub(B150K$oid, 4)
B160K$oid = str_sub(B160K$oid, 4)
B170K$oid = str_sub(B170K$oid, 4)
B180K$oid = str_sub(B180K$oid, 4)
write.table(B80K[,2:4],'/home/acald013/Datasets/Berlin/B80K.csv',row.names=F,col.names=F,quote=F,sep=',')
write.table(B90K[,2:4],'/home/acald013/Datasets/Berlin/B90K.csv',row.names=F,col.names=F,quote=F,sep=',')
write.table(B100K[,2:4],'/home/acald013/Datasets/Berlin/B100K.csv',row.names=F,col.names=F,quote=F,sep=',')
write.table(B110K[,2:4],'/home/acald013/Datasets/Berlin/B110K.csv',row.names=F,col.names=F,quote=F,sep=',')
write.table(B120K[,2:4],'/home/acald013/Datasets/Berlin/B120K.csv',row.names=F,col.names=F,quote=F,sep=',')
write.table(B130K[,2:4],'/home/acald013/Datasets/Berlin/B130K.csv',row.names=F,col.names=F,quote=F,sep=',')
write.table(B140K[,2:4],'/home/acald013/Datasets/Berlin/B140K.csv',row.names=F,col.names=F,quote=F,sep=',')
write.table(B150K[,2:4],'/home/acald013/Datasets/Berlin/B150K.csv',row.names=F,col.names=F,quote=F,sep=',')
write.table(B160K[,2:4],'/home/acald013/Datasets/Berlin/B160K.csv',row.names=F,col.names=F,quote=F,sep=',')
write.table(B170K[,2:4],'/home/acald013/Datasets/Berlin/B170K.csv',row.names=F,col.names=F,quote=F,sep=',')
write.table(B180K[,2:4],'/home/acald013/Datasets/Berlin/B180K.csv',row.names=F,col.names=F,quote=F,sep=',')



