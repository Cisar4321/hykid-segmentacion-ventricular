import sys

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from .io import list_scans, load_seg
from .paths import OUTPUTS

GT_VENT = [4]
SS_VENT = [4, 5, 43, 44]
EROSION_MM = 10
MIN_ML = 0.5


def otsu(x, bins=256):
    hist, edges = np.histogram(x, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    return centers[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def skull_strip(img, zooms, erode_mm=4):
    fg = img > otsu(img[img > 0])
    r = max(1, int(round(erode_mm / min(zooms))))
    ball = np.ones((3, 3, 3), bool)
    core = ndi.binary_erosion(fg, ball, iterations=r)
    lab, n = ndi.label(core)
    if not n:
        return fg
    big = lab == (np.argmax(ndi.sum(core, lab, range(1, n + 1))) + 1)
    brain = ndi.binary_dilation(big, ball, iterations=r) & fg
    brain = ndi.binary_closing(brain, ball, iterations=3)
    for k in range(brain.shape[2]):
        brain[:, :, k] = ndi.binary_fill_holes(brain[:, :, k])
    return brain


def segment(img, brain, zooms):
    vals = img[brain]
    lo, hi = np.percentile(vals, [1, 99])
    norm = np.clip((img - lo) / (hi - lo), 0, 1)
    t = otsu(norm[brain])
    dark = (norm < t) & brain

    r = max(1, int(round(EROSION_MM / min(zooms))))
    ball = np.ones((3, 3, 3), bool)
    inner = ndi.binary_erosion(brain, ball, iterations=r)
    cand = ndi.binary_opening(dark & inner, ball, iterations=1)

    lab, n = ndi.label(cand)
    if not n:
        return cand
    ml = float(np.prod(zooms)) / 1000
    sizes = ndi.sum(cand, lab, range(1, n + 1)) * ml
    center = np.array(ndi.center_of_mass(brain))
    coms = np.array(ndi.center_of_mass(cand, lab, range(1, n + 1)))
    dist = np.linalg.norm((coms - center) * np.array(zooms), axis=1)
    keep = [i + 1 for i in range(n) if sizes[i] >= MIN_ML and dist[i] < 45]
    if not keep:
        keep = [int(np.argmax(sizes)) + 1]
    out = np.isin(lab, keep)
    shell = ndi.binary_erosion(brain, ball, iterations=max(1, int(round(3 / min(zooms)))))
    out = ndi.binary_propagation(out, mask=dark & shell)
    out = ndi.binary_closing(out, ball, iterations=2)
    for k in range(out.shape[2]):
        out[:, :, k] = ndi.binary_fill_holes(out[:, :, k])
    return out & brain


def dice(a, b):
    d = a.sum() + b.sum()
    return 2 * (a & b).sum() / d if d else np.nan


def main():
    n_show = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out_dir = OUTPUTS / "seg_clasica"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, store = [], []
    for s in list_scans():
        ss_path = s / "segmentation" / "synthseg.nii.gz"
        if not ss_path.exists():
            continue
        img_nib = nib.as_closest_canonical(nib.load(s / "eT1W_FFE_SVR.nii.gz"))
        img = img_nib.get_fdata()
        zooms = img_nib.header.get_zooms()[:3]
        gt, _ = load_seg(s / "segmentation" / "gt.nii.gz")
        ss, _ = load_seg(ss_path)
        if not (img.shape == gt.shape == ss.shape):
            print("tamaños distintos, se omite", s)
            continue

        brain = ndi.binary_fill_holes(ss > 0)
        pred = segment(img, brain, zooms)
        g = np.isin(gt, GT_VENT)
        a = np.isin(ss, SS_VENT)
        ml = float(np.prod(zooms)) / 1000

        nib.save(nib.Nifti1Image(pred.astype(np.uint8), img_nib.affine), out_dir / f"{s.parent.name}_{s.name}.nii.gz")
        rows.append({"subject": s.parent.name, "scan": s.name, "vol_gt": g.sum() * ml,
                     "vol_clasico": pred.sum() * ml, "vol_synthseg": a.sum() * ml,
                     "dice_clasico": dice(g, pred), "dice_synthseg": dice(g, a)})
        store.append((s, img, g, a, pred))
        print(f"{s.parent.name}: gt {g.sum() * ml:6.1f} mL | clásico {dice(g, pred):.2f} | synthseg {dice(g, a):.2f}")

    df = pd.DataFrame(rows).sort_values("vol_gt")
    df.to_csv(OUTPUTS / "seg_clasica_resultados.csv", index=False)
    print("\nDice medio clásico :", round(df.dice_clasico.mean(), 3), "+/-", round(df.dice_clasico.std(), 3))
    print("Dice medio SynthSeg:", round(df.dice_synthseg.mean(), 3), "+/-", round(df.dice_synthseg.std(), 3))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(df.vol_gt, df.dice_synthseg, label="SynthSeg", color="darkorange")
    ax.scatter(df.vol_gt, df.dice_clasico, label="Clásico (umbral + morfología)", color="teal")
    ax.set_xlabel("Volumen ventricular experto (mL)")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Desempeño según severidad")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "dice_vs_volumen.png", dpi=150)
    plt.close(fig)

    by_name = {(s.parent.name, s.name): (s, img, g, a, p) for s, img, g, a, p in store}
    order = [(r.subject, r.scan) for r in df.itertuples()]
    picks = [order[i] for i in np.linspace(0, len(order) - 1, min(n_show, len(order))).astype(int)]

    titles = ["RM", "Experto", "SynthSeg", "Clásico"]
    colors = ["purple", "darkorange", "teal"]
    fig, axes = plt.subplots(len(picks), 4, figsize=(14, 3.6 * len(picks)), squeeze=False)
    for row, key in zip(axes, picks):
        s, img, g, a, p = by_name[key]
        k = int(np.argmax(g.sum(axis=(0, 1))))
        sl = np.rot90(img[:, :, k])
        vmax = np.percentile(sl[sl > 0], 99.5) if (sl > 0).any() else 1
        r = df[(df.subject == key[0]) & (df.scan == key[1])].iloc[0]
        info = ["", "", f"Dice {r.dice_synthseg:.2f}", f"Dice {r.dice_clasico:.2f}"]
        for j, ax in enumerate(row):
            ax.imshow(sl, cmap="gray", vmax=vmax)
            ax.axis("off")
            if j:
                m = np.rot90([g, a, p][j - 1][:, :, k])
                ax.imshow(np.ma.masked_where(~m, m), cmap="Greys", alpha=0.0)
                ax.contourf(m, levels=[0.5, 1.5], colors=[colors[j - 1]], alpha=0.45)
                ax.contour(m, levels=[0.5], colors=[colors[j - 1]], linewidths=1)
            ax.set_title(f"{titles[j]} {info[j]}", fontsize=10)
        row[0].set_title(f"{key[0]} | {r.vol_gt:.0f} mL", fontsize=10)
    plt.suptitle("Segmentación del ventrículo lateral: experto vs SynthSeg vs método clásico", fontsize=13, weight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(OUTPUTS / "comparacion_metodos.png", dpi=130)
    print("\nListo:", OUTPUTS)


if __name__ == "__main__":
    main()
