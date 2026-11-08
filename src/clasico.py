import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.segmentation import watershed

ball = ndi.generate_binary_structure(3, 1)

EROSION = 8
BIAS = True
BIAS_SIGMA_MM = 25
MEDIAN = 3

RIM = 3
OPEN_ENC = 1
MIN_ML = 0.5
BIG_ML = 40
MAX_DIST = 45
RATIO = 3

ROI_X_MM, ROI_Y_MM = 35, 45
ROI_ZDOWN_MM, ROI_ZUP_MM = 10, 40
MID_MM = 2
TOPHAT_MM = 4

MIN_FINAL_ML = 0.5
MIN_FRAC = 0.05
PAIR_SIM = 0.4

T_MID = 0.3
WS_FAR_MM = 10
SIGMA_WS = 1.0
CLOSE_ITER = 2

EROS_REC = 2
MID_REC_MM = 5
Z_OFF_MM = 0
MAX_LOSS = 0.7


def crop_box(img, pad=6):
    head = img > threshold_otsu(img[img > 0])
    idx = np.where(head)
    return tuple(slice(max(i.min() - pad, 0), i.max() + pad + 1) for i in idx)


def rellenar_3planos(m):
    m = m.copy()
    for axis in range(3):
        for i in range(m.shape[axis]):
            sl = [slice(None)] * 3
            sl[axis] = i
            m[tuple(sl)] = ndi.binary_fill_holes(m[tuple(sl)])
    return m


def brain_mask(img):
    head = img > threshold_otsu(img[img > 0])
    eroded = ndi.binary_erosion(head, ball, iterations=EROSION)
    lab, n = ndi.label(eroded)
    largest = lab == (np.argmax(ndi.sum(eroded, lab, range(1, n + 1))) + 1)
    brain = ndi.binary_dilation(largest, ball, iterations=EROSION) & head
    brain = ndi.binary_closing(brain, ball, iterations=3)
    return rellenar_3planos(ndi.binary_fill_holes(brain))


def corregir_sesgo(img, brain, zooms):
    tejido = brain & (img > threshold_otsu(img[brain]))
    logi = np.log1p(np.clip(img, 0, None))
    s = [BIAS_SIGMA_MM / z for z in zooms]
    num = ndi.gaussian_filter(logi * tejido, s)
    den = ndi.gaussian_filter(tejido.astype(float), s)
    bias = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0)
    ref = logi[tejido].mean()
    corr = np.expm1(logi - bias + ref) * brain
    return corr, np.exp(bias - ref) * brain


def preprocesar_sin_sesgo(img, zooms):
    global BIAS
    prev, BIAS = BIAS, False
    out = preprocesar(img, zooms)
    BIAS = prev
    return out


def preprocesar(img, zooms):
    brain = brain_mask(img)
    if BIAS:
        corr, campo = corregir_sesgo(img, brain, zooms)
    else:
        corr, campo = img * brain, np.ones_like(img) * brain
    lo, hi = np.percentile(corr[brain], [1, 99])
    norm = np.clip((corr - lo) / (hi - lo), 0, 1) * brain
    if MEDIAN:
        norm = ndi.median_filter(norm, size=MEDIAN) * brain
    t3 = threshold_multiotsu(norm[brain], classes=3)
    csf = (norm < t3[0]) & brain
    return {"brain": brain, "campo": campo, "norm": norm, "t3": t3, "csf": csf}


def regla_areas(v, zooms, frac_min=0.0):
    ml = float(np.prod(zooms)) / 1000
    lab, n = ndi.label(v)
    if n == 0:
        return v
    sizes = ndi.sum(v, lab, range(1, n + 1)) * ml
    valid = [i for i in range(n) if sizes[i] >= MIN_FINAL_ML and sizes[i] >= frac_min * sizes.max()]
    if len(valid) <= 1:
        return np.isin(lab, [i + 1 for i in valid]) if valid else v
    order = sorted(valid, key=lambda i: sizes[i], reverse=True)
    a, b = order[0], order[1]
    return np.isin(lab, [a + 1, b + 1] if sizes[b] >= PAIR_SIM * sizes[a] else [a + 1])


def roi_central(brain, zooms):
    c = np.array(ndi.center_of_mass(brain))
    X, Y, Z = np.ogrid[:brain.shape[0], :brain.shape[1], :brain.shape[2]]
    dx, dy, dz = (X - c[0]) * zooms[0], (Y - c[1]) * zooms[1], (Z - c[2]) * zooms[2]
    box = (np.abs(dx) < ROI_X_MM) & (np.abs(dy) < ROI_Y_MM) & (dz > -ROI_ZDOWN_MM) & (dz < ROI_ZUP_MM)
    return box & brain, np.abs(dx) < MID_MM, c


def lcr_encerrado(csf, brain, zooms):
    ml = float(np.prod(zooms)) / 1000
    c = ndi.binary_opening(csf, ball, iterations=OPEN_ENC)
    rim = brain & ~ndi.binary_erosion(brain, ball, iterations=RIM)
    lab, n = ndi.label(c)
    if n == 0:
        return c
    touching = set(np.unique(lab[rim])) - {0}
    sizes = ndi.sum(c, lab, range(1, n + 1)) * ml
    center = np.array(ndi.center_of_mass(brain))
    coms = np.array(ndi.center_of_mass(c, lab, range(1, n + 1)))
    dist = np.linalg.norm((coms - center) * np.array(zooms), axis=1)
    keep = [i + 1 for i in range(n) if sizes[i] >= MIN_ML and
            (i + 1 not in touching or (sizes[i] > BIG_ML and dist[i] < MAX_DIST))]
    return ndi.binary_dilation(np.isin(lab, keep), ball, iterations=OPEN_ENC) & csf


def ventriculos_pequenos(norm, t3, brain, zooms):
    ml = float(np.prod(zooms)) / 1000
    roi, mid, c = roi_central(brain, zooms)
    r = max(1, int(round(TOPHAT_MM / min(zooms))))
    th = ndi.grey_closing(norm, footprint=ndi.iterate_structure(ball, r)) - norm
    t = threshold_otsu(th[roi]) if roi.any() else 1
    dark = (norm < t3[1]) & brain
    cand = ndi.binary_opening((th > t) & dark & roi & ~mid, ball)
    lab, n = ndi.label(cand)
    if n == 0:
        return cand
    sizes = ndi.sum(cand, lab, range(1, n + 1)) * ml
    coms = np.array(ndi.center_of_mass(cand, lab, range(1, n + 1)))
    left = [i + 1 for i in range(n) if sizes[i] >= MIN_ML and coms[i][0] < c[0]]
    right = [i + 1 for i in range(n) if sizes[i] >= MIN_ML and coms[i][0] >= c[0]]
    if left and right:
        keep = left + right
    elif left or right:
        keep = [max(left or right, key=lambda l: sizes[l - 1])]
    else:
        return np.zeros_like(cand)
    return ndi.binary_dilation(np.isin(lab, keep), ball) & dark & ~mid


def gradiente(norm):
    s = ndi.gaussian_filter(norm, SIGMA_WS)
    return np.sqrt(sum(ndi.sobel(s, axis=a) ** 2 for a in range(3)))


def watershed_relleno(v, norm, t3, brain, zooms):
    if v.sum() == 0:
        return v
    seed = ndi.binary_erosion(v, ball) if ndi.binary_erosion(v, ball).any() else v
    far = ~ndi.binary_dilation(v, ball, iterations=max(1, int(round(WS_FAR_MM / min(zooms)))))
    markers = np.zeros(norm.shape, np.int32)
    markers[~brain | far | (norm > t3[0] + T_MID * (t3[1] - t3[0]))] = 1
    markers[seed] = 2
    return (watershed(gradiente(norm), markers) == 2) & brain


def quitar_media_inferior(v, brain, zooms):
    if v.sum() == 0:
        return v
    c = ndi.center_of_mass(brain)
    X = np.arange(v.shape[0])[:, None, None]
    banda = np.abs((X - c[0]) * zooms[0]) < MID_REC_MM
    lat = v & ~banda
    if lat.sum() == 0:
        return v
    z_corte = ndi.center_of_mass(lat)[2] + Z_OFF_MM / zooms[2]
    Z = np.arange(v.shape[2])[None, None, :]
    bloqueo = banda & (Z < z_corte)
    semilla = regla_areas(ndi.binary_erosion(v, ball, iterations=EROS_REC) & ~banda, zooms)
    if semilla.sum() == 0:
        return v
    rec = regla_areas(ndi.binary_propagation(semilla, structure=ball, mask=v & ~bloqueo), zooms)
    return v if rec.sum() < (1 - MAX_LOSS) * v.sum() else rec


def segmentar(img, zooms):
    p = preprocesar(img, zooms)
    brain, norm, t3, csf = p["brain"], p["norm"], p["t3"], p["csf"]
    ml = float(np.prod(zooms)) / 1000
    pasos = {}

    enc = lcr_encerrado(csf, brain, zooms)
    roi, mid, _ = roi_central(brain, zooms)
    est = ndi.binary_opening(csf & roi & ~mid, ball).sum() * ml
    if enc.sum() * ml < MIN_FINAL_ML or enc.sum() * ml > RATIO * max(est, 1):
        v, rama = ventriculos_pequenos(norm, t3, brain, zooms), "pequeño"
    else:
        v, rama = enc, "encerrado"
    if v.sum() * ml < MIN_FINAL_ML:
        BIAS_OFF = preprocesar_sin_sesgo(img, zooms)
        enc2 = lcr_encerrado(BIAS_OFF["csf"], brain, zooms)
        if enc2.sum() * ml >= MIN_FINAL_ML:
            v, rama = enc2, "encerrado sin sesgo"
            norm, t3 = BIAS_OFF["norm"], BIAS_OFF["t3"]
    pasos["inicial"] = v

    v = ndi.binary_opening(v, ball)
    v = regla_areas(v, zooms, MIN_FRAC)
    pasos["areas"] = v
    v = watershed_relleno(v, norm, t3, brain, zooms)
    pasos["watershed"] = v
    v = rellenar_3planos(ndi.binary_closing(v, ball, iterations=CLOSE_ITER))
    v = regla_areas(v, zooms)
    pasos["forma"] = v
    v = quitar_media_inferior(v, brain, zooms)
    pasos["final"] = v
    return v, rama, est, p, pasos
