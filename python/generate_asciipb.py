from google.protobuf.text_format import PrintMessage
from math import sqrt
from proto.py import mixing_prior_pb2
from proto.py import hierarchy_prior_pb2

# Run this from root with python -m python.generate_asciipb

if __name__ == "__main__":
  # DP gamma hyperprior
  dp_prior = mixing_prior_pb2.DPPrior()
  dp_prior.gamma_prior.totalmass_prior.shape = 4.0
  dp_prior.gamma_prior.totalmass_prior.rate = 2.0
  with open("resources/asciipb/dp_gamma_prior.asciipb", "w") as f:
    PrintMessage(dp_prior, f)

  # PY fixed values
  py_prior = mixing_prior_pb2.PYPrior()
  py_prior.fixed_values.strength = 1.0
  py_prior.fixed_values.discount = 0.1
  with open("resources/asciipb/py_fixed.asciipb", "w") as f:
    PrintMessage(py_prior, f)


  # NNIG NGG hyperprior
  nnig_prior = hierarchy_prior_pb2.NNIGPrior()
  nnig_prior.ngg_prior.mean_prior.mean = 5.5
  nnig_prior.ngg_prior.mean_prior.var = 2.25
  nnig_prior.ngg_prior.var_scaling_prior.shape = 0.2
  nnig_prior.ngg_prior.var_scaling_prior.rate = 0.6
  nnig_prior.ngg_prior.shape = 1.5
  nnig_prior.ngg_prior.scale_prior.shape = 4.0
  nnig_prior.ngg_prior.scale_prior.rate = 2.0
  with open("resources/asciipb/nnig_ngg_prior.asciipb", "w") as f:
    PrintMessage(nnig_prior, f)

  nnig_prior = hierarchy_prior_pb2.NNIGPrior()
  nnig_prior.fixed_values.mean = 0.0
  nnig_prior.fixed_values.var_scaling = 0.1
  nnig_prior.fixed_values.shape = 2.0
  nnig_prior.fixed_values.scale = 2.0
  with open("resources/asciipb/nnig_fixed.asciipb", "w") as f:
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
  with open("resources/asciipb/nnw_ngiw_prior.asciipb", "w") as f:
    PrintMessage(nnw_prior, f)



  # LinRegUni fixed values
  lru_prior = hierarchy_prior_pb2.LinRegUnivPrior()
  dim = 3
  beta0 = dim*[0]
  Lambda0 = [1.0, 0.0, 0.0, 1.0]
  lru_prior.fixed_values.mean.size = len(beta0)
  lru_prior.fixed_values.mean.data[:] = beta0
  lru_prior.fixed_values.var_scaling.rows = int(sqrt(len(Lambda0)))
  lru_prior.fixed_values.var_scaling.cols = int(sqrt(len(Lambda0)))
  lru_prior.fixed_values.var_scaling.data[:] = Lambda0
  lru_prior.fixed_values.var_scaling.rowmajor = False
  lru_prior.fixed_values.shape = 2.0
  lru_prior.fixed_values.scale = 2.0
  with open("resources/asciipb/lin_reg_univ_fixed.asciipb", "w") as f:
    PrintMessage(lru_prior, f)
