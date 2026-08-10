"""Figura 15 — Campo esparso de velocidade 2018–2019 (feature tracking) + perfil.

Analogo honesto da Fig. S1 de Duan et al. 2025: como 79% da superficie
decorrelaciona no intervalo anual, mostramos os vetores QC-aprovados (setas
coloridas por magnitude, viridis) sobre hillshade — nao um raster interpolado.
Painel (b): velocidade vs posicao ao longo do eixo de fluxo (proxy: UTM Leste;
a frente/lago fica a oeste), evidenciando a lingua terminal mais lenta.
"""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling

REPO = '/home/matheus/Documents/Github/LACRIO_IC'

# grade 2 m (transform)
ref = rasterio.open(f'{REPO}/results/2016/crevasses_mask_2016_8cm.tif')
t2 = ref.transform * ref.transform.scale(25, 25)

rows = [r for r in csv.DictReader(open(f'{REPO}/results/velocity_2018_2019.csv'))
        if r['qc_ok'] == 'True' and r['on_ice'] == 'True']
px = np.array([float(r['px_x']) for r in rows])
py = np.array([float(r['px_y']) for r in rows])
u = np.array([float(r['u_m']) for r in rows])
v = np.array([float(r['v_m']) for r in rows])          # +v = sul (imagem)
X = t2.c + px * t2.a
Y = t2.f + py * t2.e
sp = np.hypot(u, v)

# hillshade decimado
hsrc = rasterio.open(f'{REPO}/results/qgis/hillshade_2016.tif')
f = 8
hs = hsrc.read(1, out_shape=(hsrc.height // f, hsrc.width // f),
               resampling=Resampling.average).astype(float)
hs[hs <= 1] = 255  # nodata -> branco
ht = hsrc.transform * hsrc.transform.scale(f, f)
ext = [ht.c, ht.c + ht.a * hs.shape[1], ht.f + ht.e * hs.shape[0], ht.f]

outline = json.load(open(f'{REPO}/results/qgis/glacier_outline.geojson'))
ring = np.array(outline['features'][0]['geometry']['coordinates'][0])

fig = plt.figure(figsize=(7.48, 4.3), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], wspace=0.24)

ax = fig.add_subplot(gs[0])
ax.imshow(hs, extent=ext, cmap='gray', vmin=30, vmax=230, zorder=0)
ax.plot(ring[:, 0], ring[:, 1], 'k-', lw=0.9, zorder=2)
q = ax.quiver(X, Y, u, -v, sp, cmap='viridis', clim=(0, 150),
              scale=2600, width=0.004, zorder=3)
ax.quiverkey(q, 0.86, -0.13, 100, '100 m a⁻¹', labelpos='E', fontproperties={'size': 7})
cb = fig.colorbar(q, ax=ax, shrink=0.55, pad=0.012)
cb.set_label('Velocidade (m a⁻¹)', fontsize=8); cb.ax.tick_params(labelsize=7)
ax.set_xlim(376900, 379300); ax.set_ylim(3969800, 3971700)
ax.set_xlabel('Leste UTM 19S (m)', fontsize=8)
ax.set_ylabel('Norte UTM 19S (m)', fontsize=8)
ax.tick_params(labelsize=7)
ax.ticklabel_format(style='plain')
ax.annotate('N', xy=(0.95, 0.93), xycoords='axes fraction', fontsize=11,
            fontweight='bold', ha='center')
ax.annotate('▲', xy=(0.95, 0.885), xycoords='axes fraction', fontsize=9, ha='center')
ax.plot([377050, 377550], [3969900, 3969900], 'k-', lw=2.5)
ax.text(377300, 3969935, '500 m', ha='center', fontsize=7)
ax.text(377280, 3970490, 'lago\nproglacial', fontsize=7, style='italic', color='#333')
ax.set_title(f'(a) Vetores de velocidade 2018→2019\n(n = {len(u)} no QC; decorrelação 79%)', fontsize=8, pad=3)

ax2 = fig.add_subplot(gs[1])
dist = (X - 377250)  # proxy: distancia ao longo do eixo E-W a partir da frente
sc = ax2.scatter(dist, sp, c=sp, cmap='viridis', vmin=0, vmax=150, s=14,
                 edgecolors='k', linewidths=0.2)
med_lo = np.median(sp[dist < np.median(dist)])
med_hi = np.median(sp[dist >= np.median(dist)])
ax2.axhline(med_lo, color='#888', lw=0.8, ls='--')
ax2.axhline(med_hi, color='#888', lw=0.8, ls=':')
ax2.text(0.02, med_lo + 4, f'mediana metade O (frente): {med_lo:.0f}', fontsize=6.5,
         transform=ax2.get_yaxis_transform())
ax2.text(0.02, med_hi + 4, f'mediana metade L (montante): {med_hi:.0f}', fontsize=6.5,
         transform=ax2.get_yaxis_transform())
ax2.set_xlabel('Distância da frente ao longo do\neixo de fluxo O→L (m, proxy)', fontsize=8)
ax2.set_ylabel('Velocidade (m a⁻¹)', fontsize=8)
ax2.tick_params(labelsize=7)
ax2.set_title('(b) Velocidade × posição no fluxo', fontsize=8.5)
ax2.grid(alpha=0.25, lw=0.4)

fig.savefig(f'{REPO}/results/figures/relatorio/fig15_velocidade.png',
            bbox_inches='tight', dpi=300)
print('OK -> fig15_velocidade.png')
