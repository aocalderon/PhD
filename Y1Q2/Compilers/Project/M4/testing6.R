plotSpline2 = function(x, y, dataset, k, plotExtras = F){
  K = 100   # Number of knots
  knots = (1:K)/(K+1)
  X = bs(x,knots=knots,intercept=TRUE)
  lambda = 10                           # Smoothing parameter
  lmfit = fit.p.spline(y,X,lambda=lambda)
  
  plot(x,lmfit$fit,main=paste(dataset,'k =',k),ylim=c(0,1),xlab="p",ylab="Error rate",type='l',col="red",lwd=2,lty=2)
  #points(x,y,pch=21,bg=1,cex=0.4)
  if(plotExtras){
    plot(data$p,lmfit$resid,xlab="p",ylab="Residual") 
    abline(h=0)
    plot(data$Accuracy,lmfit$resid,xlab="Accuracy",ylab="Residual") 
    abline(h=0)    
  }
}

datasets = c('iris', 'zoo', 'seeds', 'wine', 'cancer')
datasets = c('iris', 'zoo', 'seeds', 'wine')
ani.options(interval=0.5,autobrowse=F)

for(dataset in datasets[1:length(datasets)]){
  saveGIF({
    for(k in seq(1,20,2)){
      filename = paste0('Temp2/k',k,'_',dataset,'.dat')
      y = as.matrix(read.table(filename))
      x = seq(0.01,1,length.out = length(y))
      #pdf("figures/iris.pdf", 7.83, 5.17)
      plotSpline2(x,y,dataset,k)
      #lines(x,(1-a[2])*exp(b[2]*(x-1))+c,col='blue',lwd=2,lty=2)
      #legend('topleft', legend=c("observed error", "fit model"),inset=c(0.01,0.02)
      #       ,col=c(2,4),lty=c(1,2),lwd=c(2,2),cex=0.9)
      #dev.off()
    }
  }, movie.name = paste0(dataset,".gif"))
}
for(dataset in datasets[1:length(datasets)]){
  file.copy(sprintf('%s.gif',dataset),sprintf('Temp2/%s.gif',dataset), overwrite = T)
  file.remove(sprintf('%s.gif',dataset))
}
