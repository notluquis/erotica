"""Rewrite JSON files with NaN/Infinity as null (repo hook ``json-estricto``)."""

import json
import math
import sys


def clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


if __name__ == "__main__":
    for p in sys.argv[1:]:
        d = json.loads(open(p).read())
        open(p, "w").write(json.dumps(clean(d), indent=1) + "\n")
