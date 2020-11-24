from google.protobuf.text_format import PrintMessage
import sys
sys.path.insert(1, 'proto/py')
import mixing_prior_pb2
import hierarchy_prior_pb2

dp_prior = mixing_prior_pb2.DPPrior()
# dp_prior.fixed_value.value = 2.0
dp_prior.gamma_prior.shape = 4.0
dp_prior.gamma_prior.rate = 2.0
with open("resources/dp_gamma_prior.asciipb", "w") as f:
  PrintMessage(dp_prior, f)

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
