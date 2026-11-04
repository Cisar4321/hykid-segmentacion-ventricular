import pandas as pd

from .io import list_scans, load_seg, read_labels, volumes_ml
from .paths import OUTPUTS


def main():
    labels = read_labels()
    rows = []
    for s in list_scans():
        seg, z = load_seg(s / "segmentation" / "gt.nii.gz")
        vols = volumes_ml(seg, z)
        rows.append({"subject": s.parent.name, "scan": s.name,
                     **{labels.get(i, f"label_{i}"): v for i, v in vols.items()}})
    df = pd.DataFrame(rows)
    OUTPUTS.mkdir(exist_ok=True)
    df.to_csv(OUTPUTS / "volumes_gt.csv", index=False)
    print(df.describe().round(1))


if __name__ == "__main__":
    main()
