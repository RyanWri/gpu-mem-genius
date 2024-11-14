import psutil


def get_cpu_metrics():
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_freq": psutil.cpu_freq()._asdict(),
    }


def get_memory_metrics():
    memory = psutil.virtual_memory()
    return {
        "total_memory": memory.total,
        "available_memory": memory.available,
        "used_memory": memory.used,
        "memory_percent": memory.percent,
    }


def get_disk_metrics():
    disk = psutil.disk_usage("/")
    return {
        "total_disk": disk.total,
        "used_disk": disk.used,
        "free_disk": disk.free,
        "disk_percent": disk.percent,
    }


def collect_system_metrics():
    return {
        "cpu": get_cpu_metrics(),
        "memory": get_memory_metrics(),
        "disk": get_disk_metrics(),
    }


if __name__ == "__main__":
    metrics = collect_system_metrics()
    print(metrics)
