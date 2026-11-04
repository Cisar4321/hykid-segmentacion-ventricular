import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from .io import find_ids, list_scans, load, load_seg, read_info, read_labels, volumes_ml
from .paths import OUTPUTS

OUT = OUTPUTS / "hykid_revision.pdf"
N_CASOS = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def norm_vmax(v):
    return np.percentile(v[v > 0], 99.5) if (v > 0).any() else 1


labels = read_labels()
vent_ids = find_ids(labels, "ventric")
print("labels:", labels)
print("ventrículo:", vent_ids)

scans = list_scans()
print(len(scans), "scans con gt")

cases = []
for s in scans:
    gt, z = load_seg(s / "segmentation" / "gt.nii.gz")
    vols = volumes_ml(gt, z)
    vv = sum(vols.get(i, 0) for i in vent_ids)
    cases.append({"scan": s, "name": f"{s.parent.name}/{s.name}", "info": read_info(s), "vols": vols, "vv": vv})

all_ids = sorted({i for c in cases for i in c["vols"]})
colors = plt.cm.tab10.colors
cmap = ListedColormap([colors[i % 10] for i in range(max(all_ids) + 1)]) if all_ids else None
legend = [Patch(color=colors[i % 10], label=labels.get(i, f"label {i}")) for i in all_ids]


def overlay(ax, vol, seg, axis, idx, title, vmax):
    take = lambda a: np.rot90(np.take(a, idx, axis=axis))
    ax.imshow(take(vol), cmap="gray", vmax=vmax)
    ax.imshow(np.ma.masked_equal(take(seg), 0), cmap=cmap, alpha=0.45, vmin=0, vmax=max(all_ids))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plain(ax, sl, title, vmax):
    ax.imshow(np.rot90(sl), cmap="gray", vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def vent_slice(seg, axis):
    m = np.isin(seg, vent_ids) if vent_ids else seg > 0
    counts = m.sum(axis=tuple(i for i in range(3) if i != axis))
    return int(np.argmax(counts)) if counts.any() else seg.shape[axis] // 2


def info_text(info, max_lines=14):
    lines = []
    for k, v in info.items():
        txt = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
        lines.append(f"{k}: {txt[:90]}")
    return "\n".join(lines[:max_lines]) or "sin Info.json"


ordered = sorted(cases, key=lambda c: c["vv"])
if len(ordered) > N_CASOS:
    idx = np.linspace(0, len(ordered) - 1, N_CASOS).astype(int)
    selected = [ordered[i] for i in idx]
else:
    selected = ordered

OUTPUTS.mkdir(exist_ok=True)
with PdfPages(OUT) as pdf:
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(0.05, 0.9, "HyKid: revisión de imágenes", fontsize=22, weight="bold")
    vvs = [c["vv"] for c in cases]
    resumen = [
        f"Scans con segmentación experta: {len(cases)}",
        f"Sujetos: {len({c['scan'].parent.name for c in cases})}",
        "Imagen: T1W FFE, cortes originales ~7 mm (AX + SAG), reconstrucción SVR 1 mm (NiftyMIC)",
        "Etiquetas gt: " + ", ".join(labels.get(i, str(i)) for i in all_ids),
    ]
    if vent_ids:
        resumen.append(f"Volumen ventricular (mL): mediana {np.median(vvs):.1f}, rango {min(vvs):.1f} a {max(vvs):.1f}")
    resumen.append(f"Casos mostrados: {len(selected)}, ordenados de menor a mayor volumen ventricular")
    fig.text(0.05, 0.8, "\n".join(resumen), fontsize=12, va="top", linespacing=1.8)
    if vent_ids:
        ax = fig.add_axes([0.08, 0.08, 0.84, 0.32])
        ax.bar(range(len(ordered)), [c["vv"] for c in ordered], color="steelblue")
        ax.set_ylabel("Volumen ventricular (mL)")
        ax.set_xlabel("Casos ordenados")
        ax.set_title("Variabilidad de ventriculomegalia en la cohorte", fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)

    for c in selected:
        s = c["scan"]
        ax_v, _ = load(s / "eT1W_FFE_AX.nii.gz")
        sag_v, _ = load(s / "eT1W_FFE_SAG.nii.gz")
        svr, _ = load(s / "eT1W_FFE_SVR.nii.gz")
        gt, _ = load(s / "segmentation" / "gt.nii.gz")
        gt = gt.astype(int)
        vmax = norm_vmax(svr)

        fig = plt.figure(figsize=(11.7, 8.3))
        fig.suptitle(f"{c['name']}   |   volumen ventricular: {c['vv']:.1f} mL", fontsize=14, weight="bold")
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])

        plain(fig.add_subplot(gs[0, 0]), ax_v[:, :, ax_v.shape[2] // 2], "Original AX (~7 mm)", norm_vmax(ax_v))
        plain(fig.add_subplot(gs[0, 1]), sag_v[sag_v.shape[0] // 2], "Original SAG (~7 mm)", norm_vmax(sag_v))
        z = vent_slice(gt, 2)
        plain(fig.add_subplot(gs[0, 2]), svr[:, :, z], "SVR 1 mm axial", vmax)
        overlay(fig.add_subplot(gs[0, 3]), svr, gt, 2, z, "SVR + gt axial", vmax)

        overlay(fig.add_subplot(gs[1, 0]), svr, gt, 1, vent_slice(gt, 1), "SVR + gt coronal", vmax)
        overlay(fig.add_subplot(gs[1, 1]), svr, gt, 0, svr.shape[0] // 2, "SVR + gt sagital", vmax)
        tx = fig.add_subplot(gs[1, 2:])
        tx.axis("off")
        vol_txt = "\n".join(f"{labels.get(i, i)}: {v:.1f} mL" for i, v in sorted(c["vols"].items()))
        tx.text(0, 1, "Volúmenes gt\n" + vol_txt + "\n\nInfo.json\n" + info_text(c["info"]),
                fontsize=8, va="top", family="monospace")
        fig.legend(handles=legend, loc="lower center", ncol=len(legend), fontsize=8, frameon=False)
        pdf.savefig(fig)
        plt.close(fig)

    cols = 6
    rows = -(-len(ordered) // cols)
    per_page = 4
    for start in range(0, rows, per_page):
        chunk = ordered[start * cols:(start + per_page) * cols]
        fig, axes = plt.subplots(per_page, cols, figsize=(11.7, 8.3))
        fig.suptitle("Cohorte completa: corte axial con mayor área ventricular", fontsize=12)
        for ax in axes.flat:
            ax.axis("off")
        for ax, c in zip(axes.flat, chunk):
            svr, _ = load(c["scan"] / "eT1W_FFE_SVR.nii.gz")
            gt, _ = load(c["scan"] / "segmentation" / "gt.nii.gz")
            gt = gt.astype(int)
            overlay(ax, svr, gt, 2, vent_slice(gt, 2), f"{c['scan'].parent.name} {c['vv']:.0f} mL", norm_vmax(svr))
        pdf.savefig(fig)
        plt.close(fig)

print("Listo:", OUT.resolve())
