from google.protobuf.text_format import PrintMessage
from math import sqrt
import sys
sys.path.insert(1, 'proto/py')
import mixing_prior_pb2
import hierarchy_prior_pb2

# DP gamma hyperprior
dp_prior = mixing_prior_pb2.DPPrior()
dp_prior.gamma_prior.shape = 4.0
dp_prior.gamma_prior.rate = 2.0
with open("resources/dp_gamma_prior.asciipb", "w") as f:
  PrintMessage(dp_prior, f)

# NNIG NGG hyperprior
nnig_prior = hierarchy_prior_pb2.NNIGPrior()
nnig_prior.ngg_prior.mean_prior.mean = 5.5
nnig_prior.ngg_prior.mean_prior.var = 2.25
nnig_prior.ngg_prior.var_scaling_prior.shape = 0.2
nnig_prior.ngg_prior.var_scaling_prior.rate = 0.6
nnig_prior.ngg_prior.shape = 1.5
nnig_prior.ngg_prior.scale_prior.shape = 4.0
nnig_prior.ngg_prior.scale_prior.rate = 2.0
with open("resources/nnig_ngg_prior.asciipb", "w") as f:
  PrintMessage(nnig_prior, f)

# NNW NGIW hyperprior
nnw_prior = hierarchy_prior_pb2.NNWPrior()
mu00 = [5.5, 5.5]
mat = [1.0, 0.0, 0.0, 1.0]
nu0 = 5.0
nnw_prior.ngiw_prior.mean_prior.mean.size = len(mu00)
nnw_prior.ngiw_prior.mean_prior.mean.data[:] = mu00
sig00 = [m/nu0 for m in mat]
nnw_prior.ngiw_prior.mean_prior.var.rows = int(sqrt(len(sig00)))
nnw_prior.ngiw_prior.mean_prior.var.cols = int(sqrt(len(sig00)))
nnw_prior.ngiw_prior.mean_prior.var.data[:] = sig00
nnw_prior.ngiw_prior.mean_prior.var.rowmajor = False
nnw_prior.ngiw_prior.var_scaling_prior.shape = 0.2
nnw_prior.ngiw_prior.var_scaling_prior.rate = 0.6
nnw_prior.ngiw_prior.deg_free = nu0
nnw_prior.ngiw_prior.scale_prior.deg_free = nu0
tau00 = [nu0*m for m in mat]
nnw_prior.ngiw_prior.scale_prior.scale.rows = int(sqrt(len(tau00)))
nnw_prior.ngiw_prior.scale_prior.scale.cols = int(sqrt(len(tau00)))
nnw_prior.ngiw_prior.scale_prior.scale.data[:] = tau00
nnw_prior.ngiw_prior.scale_prior.scale.rowmajor = False
with open("resources/nnw_ngiw_prior.asciipb", "w") as f:
  PrintMessage(nnw_prior, f)
