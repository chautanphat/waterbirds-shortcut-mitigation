import io
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from .persistence import atomic


def load(path,keys=None):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in (f.files if keys is None else keys)}


def save_npz(path,**arrays):
    buf=io.BytesIO();np.savez_compressed(buf,**arrays);atomic(path,buf.getvalue())
    with np.load(path,allow_pickle=False) as f:
        for k,v in arrays.items():np.testing.assert_array_equal(f[k],v)


def figure(path,fig):
    import matplotlib.pyplot as plt
    buf=io.BytesIO();fig.savefig(buf,format='png',dpi=140,bbox_inches='tight');plt.close(fig);atomic(path,buf.getvalue())


@contextmanager
def single_writer(root):
    """OS releases the advisory writer lock even when the process is killed."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'writer.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0)
        if __import__('os').name=='nt':
            import msvcrt
            try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as exc:raise RuntimeError('Another v5 writer is active; do not run concurrent resumes') from exc
            try:yield
            finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:yield
            finally:fcntl.flock(f,fcntl.LOCK_UN)
