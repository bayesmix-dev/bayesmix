library(coda)
library(BNPmix)

# for plots
library(ggplot2)
library(tidyr)
library(dplyr)
library(purrr)
library(ggsci)
require(gplots)
require(ggpubr)

#Simulated data
set.seed(12071997)
data<-c(rnorm(50,mean=4,sd=1),rnorm(50,mean=-4,sd=1))
hist(data, breaks = 25, freq = F, main='Data Distribution')
sq <- seq(-6, 13, by = 0.1)
fsq <- .5 * dnorm(sq, 4, 1) + .5 * dnorm(sq, -4, 1)
lines(sq, fsq, col = 2) 

#-----------------------------------------------------------
# LOCATION-SCALE MODEL
model_LS <- PYdensity(data, 
                      mcmc = list(niter = 2000, nburn = 1000, method = "MAR", model = "LS", hyper = F),
                      prior = list(m0 = 0, k0 = 0.1, a0 = 2, b0 = 1))

print(model_LS)
summary(model_LS)
plot(model_LS, show_hist = T, show_clust = T)
clusters<-model_LS$clust+1
write.table(clusters, file="simulated_clusters.csv")


# point estimate for the latent partition
opt_part_LS <- partition(model_LS, dist = "Binder")
opt_part_VI <- partition(model_LS, dist= "VI")

best_LS<-opt_part_LS$partitions[1,]
best_VI<-opt_part_VI$partitions[1,]
table(best_VI, best_LS)

