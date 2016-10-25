size = 1000
j = 0
dim = 20
for(n in seq(10,100,10)){
  epsilon = n / 10
  r2 = (epsilon / 2)^2

  x = runif(n, 0, dim)
  y = runif(n, 0, dim)
  
  par(mar=c(0.1,0.1,0.1,0.1))
  pointset=data.frame(x=x,y=y)
  data = sqldf("SELECT p1.x AS x1, p1.y AS y1, p2.x AS x2, p2.y AS y2 FROM pointset p1 CROSS JOIN pointset p2 ")
  
  png(paste0("n/c-",j,".png"),width=size,height=size)
  j=j+1
  par(mar=c(0.1,0.1,0.1,0.1))
  plot(1, asp=1, axes = F, xlab = "", ylab = "", type='n',
       xlim = c(0 - epsilon, dim + epsilon), 
       ylim = c(0 - epsilon, dim + epsilon))
  pointset=data.frame(x=x,y=y)
  data = sqldf("SELECT p1.x AS x1, p1.y AS y1, p2.x AS x2, p2.y AS y2 FROM pointset p1 CROSS JOIN pointset p2 ")
  
  for(i in 1:nrow(data)){
    x1 = data[i,1]
    y1 = data[i,2]
    x2 = data[i,3]
    y2 = data[i,4]
    d = distance(x1,y1,x2,y2)
    if(d <= epsilon && d != 0){
      centers = calculateDisk(x1,y1,x2,y2)
      draw.circle(centers[1], centers[2], epsilon/2, nv = 1000, border = 2, col = NA, lty = 2, lwd = 0.5)
      draw.circle(centers[3], centers[4], epsilon/2, nv = 1000, border = 2, col = NA, lty = 2, lwd = 0.5)    
    }
  }
  points(x, y, pch = 21, cex = 2, col = 1, bg = 1 )
  box()
  obj = list(n=n, e=epsilon)
  text(10, -epsilon, bquote("n" == .(obj$n) ~ epsilon == .(obj$e)), cex=3)
  dev.off()
}