library(plotrix)

circle <- function(point, color){
  draw.circle(point['lng'], point['lat'], 
              epsilon/2, nv = 1000, border = point['ID'], 
              col = adjustcolor(point['ID'], alpha=0.05), lty = 2, lwd = 0.75)
#  label = bquote("C"[.(point['ID'])])
  label = point['ID']
  points(point['lng'], point['lat'] + epsilon/2, pch = 15, cex=0.45, col = label)
  text(point['lng'], point['lat'] + epsilon/2, label, cex=0.15, col = 'white')
}

openPlot <- function(filename){
  pdf(filename,width=3,height=3)
  par(mar=c(0,0,0,0))
  plot(1, asp=1, axes = F, , xlab = "", ylab = "", type='n',
       xlim = c(min(f0$lng) - epsilon/2,  max(f0$lng) + epsilon/2), 
       ylim = c(min(f0$lat) - epsilon/2,  max(f0$lat) + epsilon/2))
}

closePlot <- function(){
  points(points$lng, points$lat, pch = 21, cex = 0.65, col = 1, bg = 1 )
  text(points$lng, points$lat, points$ID, cex = 0.25, col = 'white')
  box(lwd=4)
  dev.off()
}

epsilon = 100

points = read.csv('p_4799.csv', header = F)
names(points) = c("ID","lng","lat")
points = points[points$lng > -326499.3,]
points$ID = 1:nrow(points)
f0 = read.csv('f0_4799.csv', header = F)
names(f0) = c("ID","lng","lat")
f0 = f0[f0$lng > -326499.3,]
f0$ID = 1:nrow(f0)

openPlot("zoom1.pdf")
apply(f0, 1, circle, 'red')
closePlot()
