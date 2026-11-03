import json
import random
import sys
import urllib.request

URL = "http://127.0.0.1:8000"


def post(rain, elev, slope):
    body = json.dumps(
        {"rainfall_mm": rain, "elevation_m": elev, "slope_deg": slope}
    ).encode()
    req = urllib.request.Request(
        URL + "/predict", data=body, headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req).read())


mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
random.seed(1)
for _ in range(40):
    if mode == "normal":  # looks like the training data
        post(random.gammavariate(2, 30), random.uniform(0, 300), random.uniform(0, 15))
    else:  # unusual: very wet, below sea level, steep
        post(random.uniform(300, 600), random.uniform(-100, -10), random.uniform(30, 80))
print(f"Sent 40 {mode} requests")
print(json.loads(urllib.request.urlopen(URL + "/drift").read()))