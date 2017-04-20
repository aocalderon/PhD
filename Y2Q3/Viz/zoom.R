library(plotrix)

circle <- function(point, color){
  draw.circle(point['lng'], point['lat'], 
              epsilon/2, nv = 1000, border = color, 
              col = adjustcolor(color, alpha=0.05), lty = 2, lwd = 0.75)
  label = bquote("C"[.(point['ID'])])
  text(point['lng'] - 2, point['lat'] + 3 + epsilon/2, label, cex=0.25)
}

openPlot <- function(filename){
  pdf(filename,width=4,height=3)
  par(mar=c(0.01,0.01,0.01,0.01))
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
f0 = read.csv('f0_4799.csv', header = F)
names(f0) = c("ID","lng","lat")
f0 = f0[f0$lng > -326499.3,]

openPlot("zoom1.pdf")
apply(f0, 1, circle, 'red')
closePlot()
