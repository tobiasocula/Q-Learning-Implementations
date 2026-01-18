import json
from pathlib import Path

with open(Path.cwd() / "generic_tests" / "test.json", "r") as f:
    data = json.load(f)