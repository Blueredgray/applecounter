"""Оценка масштаба (px на метр) по среднему размеру бокса яблока.

Яблоко спелое ~70-80 мм в диаметре. Экранный размер бокса / физический диаметр
= px/м на плоскости ряда (камера смотрит перпендикулярно ряду).
"""
import numpy as np
import pandas as pd
from pathlib import Path

run = Path(r"G:\opencode\soft\applecounter3\outputs\vfc_k_r1_e")
tf = pd.read_csv(run / "tracks_frame.csv")
print("колонки tracks_frame:", list(tf.columns))
tf["w"] = tf.x2 - tf.x1
tf["h"] = tf.y2 - tf.y1
tf["d"] = np.sqrt(tf.w * tf.h)  # эквивалентный диаметр бокса
print(f"боксов: {len(tf)}; медианный диаметр бокса: {tf.d.median():.1f}px "
      f"(p25 {tf.d.quantile(.25):.1f}, p75 {tf.d.quantile(.75):.1f})")

for phys_m in (0.070, 0.075, 0.080):
    px_per_m = tf.d.median() / phys_m
    stretch_len_m = 2095 / px_per_m  # средняя длина stretch в px
    print(f"  яблоко {phys_m*1000:.0f}мм -> {px_per_m:.1f}px/м; "
          f"stretch 2095px = {stretch_len_m:.1f}м; "
          f"при 1.2м/дерево -> {stretch_len_m/1.2:.1f} деревьев/stretch")
