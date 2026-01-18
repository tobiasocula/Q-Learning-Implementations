import json

data = {"a": 1, "b": 2}
with open("test.json", "w") as f:
    json.dump(data, f)