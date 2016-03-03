require(tiff)

im = readTIFF('pools3b.tif')
im = as.raster(im[,,1:3])
#pdf("figures/grid3.pdf", 7.83, 5.17)
plot(im)
for(i in seq(0,500,50)){
  lines(c(i,i), c(0,350),lwd=1.1,lty=2,col=4)
}
for(i in seq(0,350,50)){
  lines(c(0,520), c(i,i),lwd=1.1,lty=2,col=4)
}
#dev.off()

for(i in seq(0,500,50)){
  for(j in seq(0,350,50)){
    # text(i+2,j+2,paste0(i,',',j),cex=0.5,adj=c(0,0))
  }
}

points = read.csv('valpools.csv', header = F)
points = points[,1:2]
points(points[,1],328-points[,2],cex=0.8,pch=21,bg=1)
#dev.off()

# im = readTIFF('pools3.tif')
# im = as.raster(im[,,1:3])
# #pdf("figures/image1.pdf", 7.83, 5.17)
# plot(im)
# #dev.off()
# 
# im = readTIFF('pools3b.tif')
# im = as.raster(im[,,1:3])
# pdf("figures/image2.pdf", 7.83, 5.17)
# plot(im)
# dev.off()
