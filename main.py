import json
from jsonargparse import ArgumentParser
import jsonargparse
import pprint
from pkg.utility.common import get_dict_key_iterate
import os
from method import proposed


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

if config.setting.method == 'proposed':
    proposed.run(config)
else:
    raise NotImplementedError('only method = proposed is implemented')
