import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager

try:
    from zeus.monitor import ZeusMonitor
except Exception:
    ZeusMonitor = None


def get_gpu_info():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,power.limit,memory.total",
            "--format=csv,noheader,nounits",
        ]).decode().strip()
        return out
    except Exception as e:
        return f"nvidia-smi unavailable: {e}"


def get_env_info():
    info = {
        "python": sys.version,
    }

    try:
        import jax
        info["jax"] = jax.__version__
        info["jax_devices"] = [str(x) for x in jax.devices()]
    except Exception as e:
        info["jax"] = f"unavailable: {e}"

    try:
        import zeus
        info["zeus"] = getattr(zeus, "__version__", "unknown")
    except Exception as e:
        info["zeus"] = f"unavailable: {e}"

    return info




def write_meta(logdir, model_name, env_name, task, script, args):
    os.makedirs(str(logdir), exist_ok=True)

    path = os.path.join(str(logdir), "meta.json")

    record = {
        "model": model_name,
        "env": env_name,
        "task": task,
        "script": script,
        "steps": int(getattr(args, "steps", -1)),
        "gpu": get_gpu_info(),
        "env_info": get_env_info(),
    }

    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def block_until_ready(x=None):
    """
    Important for JAX: timing is async unless we block.

    Usage:
      block_until_ready(output)

    If x is None, this function simply does nothing.
    """
    if x is None:
        return

    try:
        import jax
        leaves = jax.tree_util.tree_leaves(x)
        for leaf in leaves:
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
    except Exception:
        pass

def make_monitor(enabled=True, gpu_indices=(0,)):
    if not enabled:
        print("[zeus_energy] Energy monitor disabled.", flush=True)
        return None

    if ZeusMonitor is None:
        print("[zeus_energy] ZeusMonitor unavailable.", flush=True)
        return None

    try:
        return ZeusMonitor(gpu_indices=list(gpu_indices))
    except Exception as e:
        print(f"[zeus_energy] Failed to create ZeusMonitor: {e}", flush=True)
        return None

@contextmanager
def energy_window(monitor, label, logfile):
    """
    Safe Zeus measurement window.

    If Zeus fails, training still continues.
    """
    if monitor is None:
        yield
        return

    measuring = False
    start_time = None

    try:
        monitor.begin_window(label)
        start_time = time.perf_counter()
        measuring = True
    except Exception as e:
        print(f"[zeus_energy] Failed to begin measurement {label}: {e}", flush=True)
        measuring = False

    try:
        yield
    finally:
        if measuring:
            try:
                duration_sec = time.perf_counter() - start_time
                measurement = monitor.end_window(label)

                record = {
                    "label": label,
                    "joules": float(measurement.total_energy),
                    "seconds_wall": float(duration_sec),
                    "seconds_zeus": float(getattr(measurement, "time", duration_sec)),
                }

                try:
                    os.makedirs(os.path.dirname(str(logfile)), exist_ok=True)
                except Exception:
                    pass

                with open(str(logfile), "a") as f:
                    json.dump(record, f)
                    f.write("\n")

            except Exception as e:
                print(f"[zeus_energy] Failed to end measurement {label}: {e}", flush=True)