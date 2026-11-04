import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .io import list_scans, load, load_seg
from .paths import OUTPUTS

GT_VENT = [4]
SS_VENT = [4, 5, 43, 44]
N = 3


def dice(a, b):
    d = a.sum() + b.sum()
    return 2 * (a & b).sum() / d if d else np.nan


def best_slice(mask):
    return int(np.argmax(mask.sum(axis=(0, 1))))


def crop_box(mask2d, pad=25):
    ys, xs = np.where(mask2d > 0)
    if not len(ys):
        return slice(None), slice(None)
    return (slice(max(ys.min() - pad, 0), ys.max() + pad),
            slice(max(xs.min() - pad, 0), xs.max() + pad))


def main():
    cases = []
    for s in list_scans():
        gt, z = load_seg(s / "segmentation" / "gt.nii.gz")
        v = np.isin(gt, GT_VENT)
        cases.append((v.sum() * np.prod(z) / 1000, s))
    cases.sort(key=lambda c: c[0])
    picks = [cases[i] for i in np.linspace(0, len(cases) - 1, N).astype(int)]

    cols = ["1. Entrada (RM)", "2. Objetivo (experto)", "3. Automático (SynthSeg)", "4. Comparación"]
    fig, axes = plt.subplots(N, 4, figsize=(15, 4 * N), squeeze=False)

    for row, (vol_ml, s) in zip(axes, picks):
        svr, _ = load(s / "eT1W_FFE_SVR.nii.gz")
        gt, _ = load_seg(s / "segmentation" / "gt.nii.gz")
        g = np.isin(gt, GT_VENT)
        ss_path = s / "segmentation" / "synthseg.nii.gz"
        a = None
        if ss_path.exists():
            ss, _ = load_seg(ss_path)
            if ss.shape == g.shape:
                a = np.isin(ss, SS_VENT)

        k = best_slice(g)
        img = np.rot90(svr[:, :, k])
        gk = np.rot90(g[:, :, k])
        ak = np.rot90(a[:, :, k]) if a is not None else None
        brain = np.rot90(gt[:, :, k] > 0)
        cy, cx = crop_box(brain)
        img, gk, brain = img[cy, cx], gk[cy, cx], brain[cy, cx]
        ak = ak[cy, cx] if ak is not None else None
        vmax = np.percentile(img[img > 0], 99.5) if (img > 0).any() else 1

        for ax in row:
            ax.imshow(img, cmap="gray", vmax=vmax)
            ax.axis("off")

        row[0].set_title(f"{s.parent.name}\nventrículo {vol_ml:.0f} mL", fontsize=10)
        row[1].imshow(np.ma.masked_where(~gk, gk), cmap="Purples", vmin=0, vmax=1.2, alpha=0.7)
        row[1].contour(gk, levels=[0.5], colors="purple", linewidths=1)

        if ak is None:
            row[2].text(0.5, 0.5, "sin synthseg", color="w", ha="center", transform=row[2].transAxes)
            continue
        row[2].imshow(np.ma.masked_where(~ak, ak), cmap="Oranges", vmin=0, vmax=1.2, alpha=0.7)
        row[2].contour(ak, levels=[0.5], colors="darkorange", linewidths=1)

        comp = np.zeros(gk.shape + (4,))
        comp[gk & ak] = (0.1, 0.8, 0.2, 0.75)
        comp[~gk & ak] = (0.9, 0.1, 0.1, 0.75)
        comp[gk & ~ak] = (0.1, 0.4, 1.0, 0.75)
        row[3].imshow(comp)
        d3 = dice(g, a)
        err = (a.sum() - g.sum()) / g.sum() * 100 if g.sum() else np.nan
        row[3].set_title(f"Dice 3D {d3:.2f} | error vol {err:+.0f}%", fontsize=10)

    for ax, t in zip(axes[0], cols):
        ax.set_title((ax.get_title() + "\n\n" if ax.get_title() else "") + t, fontsize=11, weight="bold")

    fig.legend(handles=[Patch(color=(0.1, 0.8, 0.2), label="acierto"),
                        Patch(color=(0.9, 0.1, 0.1), label="sobra (falso positivo)"),
                        Patch(color=(0.1, 0.4, 1.0), label="falta (falso negativo)")],
               loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Objetivo: segmentar automáticamente el ventrículo lateral y medir su volumen", fontsize=14, weight="bold")
    plt.tight_layout(rect=(0, 0.04, 1, 0.96))
    OUTPUTS.mkdir(exist_ok=True)
    out = OUTPUTS / "objetivo_segmentacion.png"
    plt.savefig(out, dpi=150)
    print("Listo:", out)


if __name__ == "__main__":
    main()
