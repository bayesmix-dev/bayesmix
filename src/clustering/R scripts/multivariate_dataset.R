library(mvtnorm)

mu1<-c(2,2)
mu2<-c(-2,2)
mu3<-c(-2,-2)
mu4<-c(2,-2)
Sigma1<-diag(1,2)
Sigma2<-diag(0.25,2)
Sigma3<-diag(1,2)
Sigma4<-diag(2.25,2)

n<-200
#then set n=40, n=80, n=120, n=160, n=240, n=280, n=320, n=360, n=400

set.seed(12071997)
data<-rbind(rmvnorm(n/4,mu1,Sigma1), rmvnorm(n/4,mu2,Sigma2), rmvnorm(n/4,mu3,Sigma3), rmvnorm(n/4,mu4,Sigma4))
plot(data, main='True Data')
points(data[1:(n/4),],col='blue',pch=16)
points(data[(n/4+1):(n/2),], col='red',pch=16)
points(data[(n/2+1):(3*n/4),], col='black', pch=16)
points(data[(3*n/4+1):n,],col='green',pch=16)

model_MAR_mv <- PYdensity(data, 
                          mcmc = list(niter = 6000, nburn = 1000, 
                                      method = "MAR", model = "LS", hyper = F, print_message = T),
                          prior = list(m0 = c(0,0), k0 = 1/2, n0 = 4, Sigma0 = diag(1, 2)),
                          output = list(grid = expand.grid(seq(-4, 4, by = 0.1),
                                                           seq(-4, 4, by = 0.1))))

print(model_MAR_mv)
summary(model_MAR_mv)
plot(model_MAR_mv, show_hist = T, show_clust = T)

clusters_mv<-model_MAR_mv$clust
write.table(clusters_mv, file="multivariate_clusters.csv")

part_MAR_mvH <- partition(model_MAR_mv, dist = "VI")
table(part_MAR_mvH$partitions[1,])

part_MAR_mvBH <- partition(model_MAR_mv, dist = "Binder")
table(part_MAR_mvBH$partitions[1,])
