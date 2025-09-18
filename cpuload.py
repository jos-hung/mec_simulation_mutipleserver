import psutil
import time

interval = 1
top_n_process = 5

def print_cpu_usage():
    total = psutil.cpu_percent(interval=interval)
    print(f"Total CPU usage: {total:.2f}%")

    per_core = psutil.cpu_percent(interval=None, percpu=True)
    for i, pct in enumerate(per_core):
        print(f"Core {i}: {pct:.2f}%")

    # Top process an toàn
    processes = []
    for p in psutil.process_iter():
        try:
            processes.append((p.pid, p.name(), p.cpu_percent(interval=None)))
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue

    top_process = sorted(processes, key=lambda x: x[2], reverse=True)[:top_n_process]
    print("Top processes by CPU usage:")
    for pid, name, cpu in top_process:
        print(f"  PID {pid}, Name: {name}, CPU: {cpu:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    try:
        while True:
            print_cpu_usage()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopped.")
