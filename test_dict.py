import json

with open('test.json') as f:
  data = json.load(f)
  
dictionary = dict(data)

print(dictionary)