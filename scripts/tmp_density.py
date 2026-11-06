"""Анализ: можно ли выделить отдельные деревья по профилю плотности яблок.

Для прогона VFC: мировые координаты яблок (tracks.csv) -> проекция на ось ряда ->
профиль плотности -> автокорреляция (поиск периода = расстояние между деревьями).
"""
import numpy as np
import pandas as pd
from pathlib import Path

run = Path(r"G:\opencode\soft\applecounter3\outputs\vfc_k_r1_e")
df = pd.read_csv(run / "tracks.csv")
df = df[df.tree_id > 0]  # только привязанные к stretch
print("всего яблок:", len(df), " stretch:", df.tree_id.nunique())

# PCA по (x, y) — ось ряда
pts = df[["x", "y"]].to_numpy(float)
origin = pts.mean(axis=0)
_, _, vt = np.linalg.svd(pts - origin, full_matrices=False)
d = vt[0]
s = (pts - origin) @ d
print(f"длина ряда в мировых px: {s.max()-s.min():.0f}; ось: {d.round(3)}")

# профиль плотности вдоль оси
lo, hi = s.min(), s.max()
bw = 20.0
bins = int(np.ceil((hi - lo) / bw))
counts, edges = np.histogram(s, bins=bins, range=(lo, hi))
centers = 0.5 * (edges[1:] + edges[:-1])

# сглаживание
def smooth(a, sigma):
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2); k /= k.sum()
    return np.convolve(np.pad(a.astype(float), r, mode="edge"), k, mode="valid")

sm = smooth(counts, 2.0)
print(f"бины: {bins} по {bw}px; пики плотности: min={sm.min():.0f} max={sm.max():.0f} mean={sm.mean():.0f}")

# автокорреляция для поиска периода (расстояние между деревьями)
sm0 = sm - sm.mean()
ac = np.correlate(sm0, sm0, mode="full")[len(sm0)-1:]
ac /= ac[0]
# ищем лаги 40..400 px (период дерева)
lag_min, lag_max = int(40 / bw), int(400 / bw)
seg = ac[lag_min:lag_max]
best = lag_min + int(np.argmax(seg))
print(f"автокорреляция: лучший период ~{best*bw:.0f}px (corr={seg.max():.3f})")
# топ-5 пиков
order = np.argsort(seg)[::-1]
seen, tops = [], []
for i in order:
    lag = lag_min + i
    if any(abs(lag - x) < 3 for x in seen):
        continue
    seen.append(lag); tops.append((lag * bw, seg[i]))
    if len(tops) == 5:
        break
print("топ периодов (px, corr):", [(round(a), round(b, 3)) for a, b in tops])

# также число stretch-границ: 21 stretch -> средняя длина
print(f"\nсредняя длина stretch: {(hi-lo)/21:.0f}px; средняя длина дерева при N деревьев/stretch:")
for n in (3, 4, 5, 6, 8):
    print(f"  {n} деревьев/stretch -> {((hi-lo)/21)/n:.0f}px на дерево")
