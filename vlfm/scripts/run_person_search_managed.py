"""Own only this experiment's model processes; always release them on exit."""
import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--run-name", default="run_01")
    parser.add_argument("--person-confidence", type=float, default=.65)
    parser.add_argument("--asset", choices=("cesium", "rocketbox"), default="cesium")
    parser.add_argument("--yaw", type=float, default=0)
    parser.add_argument("--iplanner", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("Run name must be a single directory name")
    if args.max_steps < 1 or not 0 <= args.person_confidence <= 1:
        raise ValueError("Invalid step budget or confidence threshold")
    asset = repo / "data/person_search" / ("CesiumMan.glb" if args.asset == "cesium"
                                            else "rocketbox/Male_Adult_01_standing.glb")
    if not asset.is_file():
        raise FileNotFoundError(asset)
    out = repo / "results/person_search" / args.run_name
    if out.exists():
        raise FileExistsError(f"Use a new --run-name to preserve previous results: {out}")
    out.mkdir(parents=True)
    services = [("grounding_dino", 12281), ("blip2itm", 12282), ("sam", 12283)]
    for _, port in services:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    env = os.environ.copy()
    env.update(GROUNDING_DINO_PORT="12281", BLIP2ITM_PORT="12282", SAM_PORT="12283",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
    processes, handles = [], []
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        for module, port in services:
            log = (out / f"service_{module}.log").open("w")
            handles.append(log)
            proc = subprocess.Popen([sys.executable, "-m", f"vlfm.vlm.{module}", "--port", str(port)],
                                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(proc)
        deadline = time.monotonic() + 240
        pending = set(port for _, port in services)
        while pending:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("A model service exited; inspect service logs")
            for port in list(pending):
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        pending.remove(port)
                        print("READY", port, flush=True)
                except OSError:
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"Model startup timeout: {pending}")
            time.sleep(2)
        with (out / "navigation.log").open("w") as log:
            nav_cmd = [sys.executable, str(repo / "scripts/person_search_demo.py"),
                                    "--max-steps", str(args.max_steps), "--output", str(out),
                                    "--person-confidence", str(args.person_confidence),
                                    "--asset", args.asset, "--yaw", str(args.yaw)]
            if args.iplanner:
                nav_cmd.append("--iplanner")
            nav = subprocess.Popen(nav_cmd,
                                   env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(nav)
            code = nav.wait(timeout=1800)
            if code:
                raise RuntimeError(f"Navigation exited {code}; inspect navigation.log")
        print((out / "metrics.json").read_text(), flush=True)
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc in processes:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for log in handles:
            log.close()
        print("All model processes owned by this experiment have stopped.", flush=True)


if __name__ == "__main__":
    main()
