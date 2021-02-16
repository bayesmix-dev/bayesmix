library(coda)
library(BNPmix)
library(MASS)

# for plots
library(ggplot2)
library(tidyr)
library(dplyr)
library(purrr)
library(ggsci)
require(gplots)
require(ggpubr)

data<-galaxies
mean<-mean(data)
s2<-var(data)

model_LS <- PYdensity(data, 
                      mcmc = list(niter = 10000, nburn = 1000, method = "MAR", model = "LS", hyper = F),
                      prior = list(m0 = mean, k0 = 1/2, a0 = 2, b0 = s2),
                      output = list(grid=seq(6000, 40000, by=100)))

print(model_LS)
summary(model_LS)
plot(model_LS, show_hist = T, show_clust = T)
clusters.gal<-model_LS$clust+1
write.table(clusters.gal, file="galaxies_clusters.csv")

grid<-model_LS$grideval
density<-model_LS$density
mean_density<-colMeans(density)

fun.value<-numeric(length(data))
for(k in 1:length(data)){
  for(j in 1:(length(grid)-1)){
    if(data[k]>=grid[j] && data[k]<grid[j+1]){
      fun.value[k]<- mean_density[j]+(mean_density[j+1] - mean_density[j])*(data[k] - grid[j])/(grid[j+1] - grid[j])
    }
  }
}


opt_part_LS <- partition(model_LS, dist = "Binder")
opt_part_VI <- partition(model_LS, dist= "VI")

best_LS<-opt_part_LS$partitions[1,]
best_VI<-opt_part_VI$partitions[1,]
table(best_VI, best_LS)


plot(grid, mean_density, type='l', main='Binder Loss Function')
points(data, fun.value, pch=16, col=best_LS)

plot(grid, mean_density, type='l', main='VI Function')
points(data, fun.value, pch=16, col=best_VI)
