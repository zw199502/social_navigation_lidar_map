from algos.RGL.policy.cadrl import CADRL
from algos.RGL.policy.lstm_rl import LstmRL
from algos.RGL.policy.sarl import SARL
from algos.RGL.policy.gcn import GCN

policy_factory = dict()
policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['gcn'] = GCN
