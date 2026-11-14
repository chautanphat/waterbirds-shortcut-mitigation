"""Bounded GPU/CPU-per-source processes, retaining every failed or unfinished job."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .grid import DEFAULT, load_config
from .reporting import report
from .storage import atomic_json, frozen_json, implementation, read_json


def launch(sources_path, output, devices=("cpu",), config_path=DEFAULT):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    specs = read_json(sources_path)["sources"]
    config = load_config(config_path)
    if not specs or not devices:
        raise ValueError("Saved sources and execution devices are required")
    expected = {(recipe, seed) for recipe, seeds in config["sources"].items() for seed in seeds}
    actual = {(s["recipe"], s["seed"]) for s in specs}
    if actual != expected or len(specs) != len(actual):
        raise ValueError("Launcher requires all seven distinct declared source identities")
    implementation(output)
    workers = {s["id"]: str(output/"jobs"/s["id"]) for s in specs}
    frozen_json(output/"study.json", dict(sources=specs, config=config, crossfit=True,
                                         devices=list(devices), worker_outputs=workers))
    outcomes = read_json(output/"workers.json")["attempts"] if (output/"workers.json").exists() else []
    pending, running = list(specs), {}
    started = time.monotonic()
    try:
        while pending or running:
            for slot, device in enumerate(devices):
                if slot in running or not pending:
                    continue
                source = pending.pop(0)
                log = (output/(source["id"]+".log")).open("a")
                command = [sys.executable, "-m", "src.methods.dpe_independent", "run", "--sources", str(Path(sources_path).resolve()),
                           "--output", workers[source["id"]], "--source-id", source["id"], "--device", device, "--config", str(config_path)]
                environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3], env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running[slot] = (source["id"], process, log)
                print("Started", source["id"], "on", device, flush=True)
            for slot, (identifier, process, log) in list(running.items()):
                if process.poll() is None:
                    continue
                outcomes.append(dict(source=identifier, returncode=process.returncode))
                log.close(); del running[slot]
                atomic_json(output/"workers.json", dict(attempts=outcomes))
                print("Finished", identifier, "returncode", process.returncode, flush=True)
            if time.monotonic()-started > config["timeout_hours"]*3600:
                raise TimeoutError("Session limit reached; resume the same source bundle and output")
            if running:
                time.sleep(10)
    finally:
        for identifier, process, log in running.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            log.close()
            outcomes.append(dict(source=identifier, returncode=process.returncode, status="interrupted"))
        atomic_json(output/"workers.json", dict(attempts=outcomes, pending=[s["id"] for s in pending]))
        report(output)
    return read_json(output/"completion.json")
