import yaml

stream = file("experimental_data/data.yaml", "r")
print(yaml.load(stream))

