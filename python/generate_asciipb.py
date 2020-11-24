from google.protobuf.text_format import PrintMessage
import sys
sys.path.insert(1, 'proto/py')
import mixing_prior_pb2

dp_prior = mixing_prior_pb2.DPPrior()
# dp_prior.fixed_value.value = 2.0
dp_prior.gamma_prior.shape = 4.0
dp_prior.gamma_prior.rate = 2.0

with open("resources/dp_prior.asciipb", "w") as f:
  PrintMessage(dp_prior, f)
