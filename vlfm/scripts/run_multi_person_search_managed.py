"""Start/stop only the services owned by the multi-person experiment."""
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
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--person-confidence", type=float, default=.58)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("Run name must be a single directory name")
    repo = Path(__file__).resolve().parents[1]
    output = repo / "results/multi_person_search" / args.run_name
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    services = [("grounding_dino", 12381), ("blip2itm", 12382), ("sam", 12383)]
    for _, port in services:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    env = os.environ.copy()
    env.update(GROUNDING_DINO_PORT="12381", BLIP2ITM_PORT="12382", SAM_PORT="12383",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
    processes, logs = [], []
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for module, port in services:
            handle = (output / f"service_{module}.log").open("w")
            logs.append(handle)
            processes.append(subprocess.Popen([sys.executable, "-m", f"vlfm.vlm.{module}", "--port", str(port)],
                                              env=env, stdout=handle, stderr=subprocess.STDOUT,
                                              start_new_session=True))
        pending, deadline = {p for _, p in services}, time.monotonic()+240
        while pending:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("A model service exited; inspect logs")
            for port in list(pending):
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        pending.remove(port)
                        print("READY", port, flush=True)
                except OSError:
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError(pending)
            time.sleep(2)
        with (output / "navigation.log").open("w") as nav_log:
            nav = subprocess.Popen([sys.executable, str(repo / "scripts/multi_person_search_demo.py"),
                                    "--output", str(output), "--max-steps", str(args.max_steps),
                                    "--person-confidence", str(args.person_confidence)],
                                   env=env, stdout=nav_log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
            processes.append(nav)
            code = nav.wait(timeout=3600)
            if code:
                raise RuntimeError(f"Navigation exited {code}; inspect navigation.log")
        print((output / "metrics.json").read_text(), flush=True)
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for handle in logs:
            handle.close()
        print("All model processes owned by this experiment have stopped.", flush=True)


if __name__ == "__main__":
    main()
