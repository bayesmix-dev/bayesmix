# Load library, after installing it if not present
if (!require('BNPmix')) {
  install.packages('BNPmix', dependencies=TRUE)
  require('BNPmix')
}

# Create logs and CSV folders
log.fold = 'log'
csv.fold = 'csv'
dir.create(log.fold)
dir.create(csv.fold)

# Initialize containers
methods = c('MAR', 'ICS')
mcmc = list(niter=5000, nburn=1000, hyper=TRUE)
rng.seed = 20201124


####################
## GALAXY DATASET ##
####################

# Read data
data.folder = '../../resources/datasets/'
galaxy = as.numeric(readLines(paste(data.folder, 'galaxy.csv', sep='')))

# Set prior
PYprior = list(strength=1.0, discount=0.1, model='LS', m1=25, s21=4,
               tau1=0.4, zeta1=0.2, a0=4, a1=4, b1=2)
output = list(grid=galaxy, out_type='FULL')

for(met in methods){
  # Run sampler
  mcmc$method = met
  set.seed(rng.seed)
  log.file = paste(log.fold, '/bnpmix_galaxy_', met, '.log', sep='')
  sink(log.file); fit = PYdensity(y=galaxy, mcmc=mcmc, prior=PYprior,
                                  output=output); sink()
  # Save density estimation to file
  csv.file = paste(csv.fold, '/bnpmix_galaxy_dens_', met, '.csv', sep='')
  conn = file(csv.file, 'wb')
  write.table(fit['density'], file=conn, row.names=FALSE, col.names=FALSE,
                              sep=',')
  close(conn)
  # Save number of clusters to file
  nclust = apply(fit$clust, 1, max)
  csv.file = paste(csv.fold, '/bnpmix_galaxy_nclu_', met, '.csv', sep='')
  conn = file(csv.file, 'wb')
  write.table(nclust, file=conn, row.names=FALSE, col.names=FALSE, sep=',')
  close(conn)
}


######################
## FAITHFUL DATASET ##
######################

# Read data
data.folder = '../../resources/datasets/'
faithful = data.matrix(read.csv(paste(data.folder, 'faithful.csv', sep=''),
                                sep=',', header=FALSE))

# Set prior
PYprior = list(strength=1.0, discount=0.1, model='LS', m1=c(3,3),
               S1=0.25*diag(2), tau1=0.4, zeta1=0.2, n0=4, n1=4,
               Sigma1=4*diag(2))
output = list(grid=faithful, out_type='FULL')

for(met in methods){
  # Run sampler
  mcmc$method = met
  set.seed(rng.seed)
  log.file = paste(log.fold, '/bnpmix_faithful_', met, '.log', sep='')
  sink(log.file); fit = PYdensity(y=faithful, mcmc=mcmc, prior=PYprior,
                                  output=output); sink()
  # Save density estimation to file
  csv.file = paste(csv.fold, '/bnpmix_faithful_dens_', met, '.csv', sep='')
  conn = file(csv.file, 'wb')
  write.table(fit['density'], file=conn, row.names=FALSE, col.names=FALSE,
                              sep=',')
  close(conn)
  # Save number of clusters to file
  nclust = apply(fit$clust, 1, max)
  conn = file(csv.file, 'wb')
  csv.file = paste(csv.fold, '/bnpmix_faithful_nclu_', met, '.csv', sep='')
  write.table(nclust, file=conn, row.names=FALSE, col.names=FALSE, sep=',')
  close(conn)
}


#####################
## HIGHDIM DATASET ##
#####################

# Read data
highdim = data.matrix(read.csv('csv/highdim_data.csv', sep=',', header=FALSE))

# Set prior
PYprior = list(strength=1.0, discount=0.1, model='LS', m1=c(0,0,0,0),
               S1=0.1*diag(4), tau1=0.2, zeta1=2.0, n0=10, n1=10,
               Sigma1=0.1*diag(4))
output = list(grid=highdim, out_type='FULL')

for(met in methods){
  # Run sampler
  mcmc$method = met
  set.seed(rng.seed)
  log.file = paste(log.fold, '/bnpmix_highdim_', met, '.log', sep='')
  sink(log.file); fit = PYdensity(y=highdim, mcmc=mcmc, prior=PYprior,
                                  output=output); sink()
  # Save density estimation to file
  csv.file = paste(csv.fold, '/bnpmix_highdim_dens_', met, '.csv', sep='')
  conn = file(csv.file, 'wb')
  write.table(fit['density'], file=conn, row.names=FALSE, col.names=FALSE,
                              sep=',')
  close(conn)
  # Save number of clusters to file
  nclust = apply(fit$clust, 1, max)
  csv.file = paste(csv.fold, '/bnpmix_highdim_nclu_', met, '.csv', sep='')
  conn = file(csv.file, 'wb')
  write.table(nclust, file=conn, row.names=FALSE, col.names=FALSE, sep=',')
  close(conn)
}
