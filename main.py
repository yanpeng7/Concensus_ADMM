import json
from jsonargparse import ArgumentParser
import jsonargparse
import pprint
from pkg.utility.common import get_dict_key_iterate
import os
from method import proposed, baseline


with open('config.json', 'r') as f:
    config = json.load(f)

parser = ArgumentParser(default_config_files=['config.json'])
for k in get_dict_key_iterate(config):
    parser.add_argument('--' + k)

config = parser.parse_args()

print("=======================\nCONFIGURATION\n=======================\n")
pprint.pprint(jsonargparse.core.namespace_to_dict(config))
print("\n")

os.environ["CUDA_VISIBLE_DEVICES"] = config.setting.gpu_index
# set the number of cpu cores
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

if config.setting.method == 'proposed':
    proposed.run(config)
elif config.setting.method == 'baseline':
    baseline.run(config)
else:
    raise NotImplementedError('only baseline and proposed is implemented')
