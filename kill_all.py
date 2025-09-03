import json, os, signal, time

try:
    with open("pids.json", "r") as f:
        pids = json.load(f)
except FileNotFoundError:
    print("No pids.json found.")
    raise SystemExit(0)

all_pids = []
if pids.get("scheduler"):
    all_pids.append(pids["scheduler"])
all_pids += pids.get("services", [])
all_pids += pids.get("servers", [])
print("Killing processes:", all_pids)
for pid in all_pids:
    try:
        os.kill(pid, signal.SIGTERM)
        print("SIGTERM", pid)
    except ProcessLookupError:
        pass

time.sleep(1)
for pid in all_pids:
    try:
        os.kill(pid, 0)
        os.kill(pid, signal.SIGKILL)
        print("SIGKILL", pid)
    except ProcessLookupError:
        pass
print("All processes terminated.")