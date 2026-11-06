import hashlib
import json
from dataclasses import asdict
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
import torch.nn.functional as F

def get_wandb_run_from_name(name: str, wandb_entity: str, wandb_project: str):
    import wandb
    api = wandb.Api()
    runs = api.runs(f"{wandb_entity}/{wandb_project}", per_page=100000)
    for run in runs:
        if run.name == name:
            run_id = run.id
            break
    run = api.run(f"{wandb_entity}/{wandb_project}/{run_id}")
    return run

def best_effort_notify(message: str):
    '''Send a notification using ntfy service, best effort basis.'''
    try:
        from python_ntfy import NtfyClient
        client = NtfyClient(topic="vla_alerts", server=os.getenv("NTFY_SERVER"), auth=os.getenv("NTFY_AUTH_TOKEN"))
        client.send(message)
    except:
        pass

def stable_short_hash(cfg: object) -> str:
    
    hash = hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True, default=str).encode()).hexdigest()
    return hash[:8]

def write_post_train_info_to_screen_specific_tmpfile(info: dict) -> str:
    sty = os.environ.get("STY")
    if not sty:
        print("STY not set (not running inside screen). Will not write to file")
        return None

    final_path = Path(f"/tmp/{sty}_train_complete.json")
    fd, tmp_path = tempfile.mkstemp(dir="/tmp", prefix=f".{sty}_train_complete.", suffix=".tmp")

    try:
        with os.fdopen(fd, "w") as f:
            json.dump(info, f)
            f.flush()
            os.fsync(f.fileno())

        # atomic on POSIX
        os.replace(tmp_path, final_path)
        return str(final_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def read_info_from_screen_specific_tmpfile() -> dict:
    sty = os.environ.get("STY")
    if not sty:
        print("STY not set (not running inside screen). Will not write to file")
        return None
    
    final_path = Path(f"/tmp/{sty}_train_complete.json")
    if not final_path.exists():
        print("WARNING: Could not read info from screen specific tempfile - file not found!")
        return None
    
    print("reading from ", final_path)
    with open(final_path, "r") as f:
        info = json.load(f)
    return info

def hash_str(s: str, out_len: int = 6) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:out_len]

def get_pc_hostname(hashed=True):
    import socket
    name = socket.gethostname()
    for i in range(10):
        if f"sydney{i}" in name:
            name = f"sydney{i}"
            break
    if hashed:
        name = hash_str(name)
    return name
