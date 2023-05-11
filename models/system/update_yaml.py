import os
import yaml

for root, dirs, files in os.walk("./"):
    if "spec.yaml" in files:
        file_name = os.path.join(root, 'spec.yaml')
        print(file_name)
        with open(file_name, 'r') as f:
            spec_yaml = yaml.safe_load(f)
        print(type(spec_yaml))
        spec_yaml['version'] = spec_yaml['version'] + 1
        with open(file_name, 'w') as f:
            yaml.dump(spec_yaml, f)
