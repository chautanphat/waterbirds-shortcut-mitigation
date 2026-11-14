"""Atomic writes with bounded retry for transient Windows/OneDrive rename locks."""
import json
import os
from pathlib import Path
import time


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        f.write(data.encode('utf-8') if isinstance(data,str) else data);f.flush();os.fsync(f.fileno())
    for attempt in range(12):
        try:os.replace(tmp,path);return
        except PermissionError:
            if attempt==11:raise
            time.sleep(min(.02*2**attempt,.5))


def dump(path,value):atomic(path,json.dumps(value,sort_keys=True,indent=2,allow_nan=False))
