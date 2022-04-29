import json
a= {'img0': "abc}=="}
print(json.loads(json.dumps(a))['img0'])