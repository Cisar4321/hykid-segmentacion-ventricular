import json

import nibabel as nib
import numpy as np

from .paths import SUBJECTS, LABELS


def load(path):
    img = nib.as_closest_canonical(nib.load(path))
    return img.get_fdata(), img.header.get_zooms()[:3]


def load_seg(path):
    seg, zooms = load(path)
    return seg.astype(int), zooms


def list_scans(require_gt=True):
    scans = sorted(p for p in SUBJECTS.rglob("scan_*") if p.is_dir())
    if require_gt:
        scans = [s for s in scans if (s / "segmentation" / "gt.nii.gz").exists()]
    return scans


def read_info(scan):
    p = scan.parent / "Info.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def parse_labels(raw):
    out = {}
    items = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for k, v in items:
        if isinstance(v, dict):
            name = v.get("name") or v.get("label") or str(v)
            idx = v.get("id", v.get("value", k))
        else:
            name, idx = v, k
        try:
            out[int(idx)] = str(name)
        except (TypeError, ValueError):
            try:
                out[int(name)] = str(idx)
            except (TypeError, ValueError):
                pass
    return out


def read_labels():
    return parse_labels(json.loads(LABELS.read_text(encoding="utf-8"))) if LABELS.exists() else {}


def find_ids(labels, *keys):
    return [i for i, n in labels.items() if any(k in n.lower() for k in keys)]


def volumes_ml(seg, zooms):
    ml = float(np.prod(zooms)) / 1000
    ids, counts = np.unique(seg[seg > 0], return_counts=True)
    return {int(i): c * ml for i, c in zip(ids, counts)}
