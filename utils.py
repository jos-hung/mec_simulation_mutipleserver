import subprocess
import re
import docker
import os
import os



def parse_size(value_str):
    """
     GiB
     B, KiB, MiB, GiB, TiB, kB, MB, GB, TB ---> GiB
    """
    value_str = value_str.strip()
    try:
        # Binary units
        if value_str.endswith("KiB"):
            num = float(value_str.replace("KiB", ""))
            return num / (1024**2)  # GiB
        elif value_str.endswith("MiB"):
            num = float(value_str.replace("MiB", ""))
            return num / 1024
        elif value_str.endswith("GiB"):
            num = float(value_str.replace("GiB", ""))
            return num
        elif value_str.endswith("TiB"):
            num = float(value_str.replace("TiB", ""))
            return num * 1024

        # Decimal units
        elif value_str.endswith("kB"):
            num = float(value_str.replace("kB", ""))
            return num / (1000**3)
        elif value_str.endswith("MB"):
            num = float(value_str.replace("MB", ""))
            return num / 1000
        elif value_str.endswith("GB"):
            num = float(value_str.replace("GB", ""))
            return num
        elif value_str.endswith("TB"):
            num = float(value_str.replace("TB", ""))
            return num * 1000

        # Bytes
        elif value_str.endswith("B"):
            num = float(value_str.replace("B", ""))
            return num / (1024**3)

        # Không có đơn vị, giả sử là GiB
        else:
            return float(value_str)

    except Exception as e:
        print(f"⚠️ Warning: cannot parse '{value_str}': {e}")
        return 0.0

def get_docker_metrics_by_name():
    # Lấy danh sách container
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        return []

    container_names = [name for name in result.stdout.splitlines() if name]
    container_names.sort(key=lambda name: int(re.search(r'(\d+)$', name).group(1) if re.search(r'(\d+)$', name) else 0))

    # Lấy stats tất cả container
    stats = subprocess.run(
        ["docker", "stats", "--no-stream", "--format",
         "{{.Name}} {{.CPUPerc}} {{.MemPerc}} {{.MemUsage}} {{.NetIO}}"], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if stats.returncode != 0:
        return []

    values = []
    for line in stats.stdout.splitlines():
        if not line.startswith("mec"):
            continue
        parts = line.split()
        try:
            # parts = [Name, CPU%, MEM%, MemUsage, ..., NetIO1, ..., NetIO2, ...]
            values.append(float(parts[1].strip('%')))
            values.append(float(parts[2].strip('%')))
            values.append(parts[3])  # MemUsage
            values.append(parts[5])  # NetIO1
            values.append(parts[7])  # NetIO2
        except Exception as e:
            print(f"Error parsing line {line}: {e}")

    return values