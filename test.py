import requests
import time

for i in range(10):
    r = requests.get("http://localhost:8001/process_frame")
    if r.status_code == 200:
        with open(f"frame_{i}.jpg", "wb") as f:
            f.write(r.content)
    else:
        print("No frame / error from edge service:", r.status_code)
    time.sleep(1)
