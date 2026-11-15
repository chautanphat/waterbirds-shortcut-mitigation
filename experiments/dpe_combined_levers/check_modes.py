"""Checks for combined_levers.py label modes (no test labels are scored).

1. official mode leaves the source identical to the frozen pipeline's source;
2. relabel changes exactly the 18 affected validation labels, nothing else;
3. drop removes exactly those 18 rows and keeps features aligned;
4. three test views have the expected sizes (5,794 / 5,794 / 5,704).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import combined_levers as cl  # noqa: E402


def main(sources_dir, source_id):
    spec = [s for s in cl.old.load_specs(sources_dir) if s["id"] == source_id][0]
    base = cl._original_get_source(spec)
    got = {}
    for mode in cl.MODES:
        cl._MODE["value"] = mode
        got[mode] = cl.old.get_source(spec)
    o, r, d = got["official"], got["relabel"], got["drop"]
    assert o.rows == base.rows and np.array_equal(o.val_y, base.val_y) and np.array_equal(o.val_g, base.val_g)
    assert np.array_equal(o._val, base._val)
    mask = cl.affected([x["id"] for x in base.rows])
    assert mask.sum() == 18 and (base.val_y[mask] == 1).all()
    assert (r.val_y != base.val_y).sum() == 18 and (r.val_y[mask] == 0).all()
    assert (r.val_g[mask] == np.array(["0" + g[1] for g in base.val_g[mask]])).all()
    assert np.array_equal(r.val_y[~mask], base.val_y[~mask]) and r._val is base._val
    assert len(d.rows) == 1181 and d._val.shape == (1181, 2048)
    assert np.array_equal(d._val, base._val[~mask]) and [x["id"] for x in d.rows] == [x["id"] for x, m in zip(base.rows, mask) if not m]
    assert base.rows == [dict(x) for x in cl._original_get_source(spec).rows]   # base not mutated
    views = cl.test_views(base)
    assert len(views["official"][1]) == 5794 and views["removed"][0].sum() == 5704
    assert (views["corrected"][1] != views["official"][1]).sum() == 90
    print("label-mode checks passed for", source_id)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
