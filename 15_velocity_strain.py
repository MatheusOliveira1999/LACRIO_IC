"""
15_velocity_strain.py - Teste de viabilidade: velocidade por feature tracking
(estilo ImGRAFT/PIV) entre ortomosaicos anuais + tensor de deformacao (Nye).

Receita de Duan et al. 2025 (secao 3.5) adaptada: correlacao cruzada normalizada
por chips (template matching NCC) em grade regular, QC por pico de correlacao,
validacao de co-registro em terreno estavel, suavizacao e gradientes -> taxas de
deformacao principais (Nye, 1959) -> comparacao da orientacao de epsilon1 com a
orientacao das fendas (fendas abrem perpendiculares a epsilon1; Harper et al.,
1998).

RISCO CONHECIDO (documentado na pesquisa de jul/2026): intervalo ANUAL com
ablacao de metros/ano pode decorrelacionar a superficie. Este script MEDE a taxa
de decorrelacao — o resultado pode ser "inviavel", e isso e um resultado.

Uso: python 15_velocity_strain.py [ano1 ano2]   (default: 2018 2019)
"""

import sys
import csv
import numpy as np
import cv2
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import median_filter, gaussian_filter

from config import Config

RES = 2.0        # m/px de trabalho
CHIP = 64        # px (128 m)
SEARCH = 80      # px de busca para cada lado (±160 m)
STRIDE = 25      # px (50 m)
CORR_MIN = 0.45


def load_gray(year, shape_ref=None):
    p = Config.PROJECT_DIR / 'Schiaparelli_glacier' / 'normalized_8cm' / f'Schiaparelli_mosaic_{year}_8cm.tif'
    with rasterio.open(p) as src:
        f = int(round(RES / 0.08))
        h, w = src.height // f, src.width // f
        rgb = src.read([1, 2, 3], out_shape=(3, h, w),
                       resampling=Resampling.average).astype(np.float32)
        t = src.transform * src.transform.scale(f, f)
    g = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return g, t


def main():
    y1, y2 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2018, 2019)
    print(f"Feature tracking {y1} -> {y2} @ {RES} m/px | chip {CHIP*RES:.0f} m | busca ±{SEARCH*RES:.0f} m")
    g1, t = load_gray(y1)
    g2, _ = load_gray(y2)
    H, W = g1.shape

    # dominio de gelo (proxy de afinamento) e estavel, na mesma grade 2 m
    dem = {}
    from rasterio.vrt import WarpedVRT
    ref = rasterio.open(Config.RESULTS_DIR / '2016' / 'crevasses_mask_2016_8cm.tif')
    for y in [2016, 2019]:
        src = rasterio.open(Config.get_dem_path(y))
        with WarpedVRT(src, crs=ref.crs, transform=t, width=W, height=H,
                       resampling=Resampling.bilinear, nodata=np.nan) as v:
            z = v.read(1).astype(np.float64)
        z[z < -1000] = np.nan
        dem[y] = z
    ice = (dem[2019] - dem[2016]) < -0.5
    stable = (np.abs(dem[2019] - dem[2016]) < 0.3) & np.isfinite(dem[2016])

    rows = []
    half = CHIP // 2
    for cy in range(SEARCH + half, H - SEARCH - half, STRIDE):
        for cx in range(SEARCH + half, W - SEARCH - half, STRIDE):
            tpl = g1[cy - half:cy + half, cx - half:cx + half]
            if tpl.std() < 2.0:
                continue
            sr = g2[cy - half - SEARCH:cy + half + SEARCH,
                    cx - half - SEARCH:cx + half + SEARCH]
            res = cv2.matchTemplate(sr.astype(np.float32), tpl.astype(np.float32),
                                    cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            du = (loc[0] - SEARCH) * RES
            dv = (loc[1] - SEARCH) * RES
            # segundo pico fora de 5 px do primeiro
            r2 = res.copy()
            x0, y0 = loc
            r2[max(0, y0-5):y0+6, max(0, x0-5):x0+6] = -1
            ratio = mx / max(r2.max(), 1e-6)
            rows.append([cx, cy, du, dv, mx, ratio,
                         bool(ice[cy, cx]), bool(stable[cy, cx])])

    arr = np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows])
    on_ice = np.array([r[6] for r in rows])
    on_stb = np.array([r[7] for r in rows])
    ok = (arr[:, 4] >= CORR_MIN) & (arr[:, 5] >= 1.05) \
         & (np.abs(arr[:, 2]) < SEARCH * RES * 0.9) & (np.abs(arr[:, 3]) < SEARCH * RES * 0.9)

    print(f"\nchips: {len(rows)} | QC-aprovados: {ok.sum()} ({100*ok.mean():.1f}%)")
    st = ok & on_stb
    if st.sum() > 20:
        sp = np.hypot(arr[st, 2], arr[st, 3])
        print(f"TERRENO ESTAVEL (n={st.sum()}): deslocamento mediano = {np.median(sp):.2f} m "
              f"(co-registro; esperado ~0)")
    gl = ok & on_ice
    print(f"GELO: aprovados {gl.sum()}/{on_ice.sum()} ({100*gl.sum()/max(on_ice.sum(),1):.1f}% — "
          f"taxa de DECORRELACAO = {100*(1-gl.sum()/max(on_ice.sum(),1)):.1f}%)")
    if gl.sum() > 30:
        sp = np.hypot(arr[gl, 2], arr[gl, 3])
        ang = np.degrees(np.arctan2(-arr[gl, 3], arr[gl, 2]))
        print(f"   velocidade: mediana={np.median(sp):.1f} m/ano | p90={np.percentile(sp, 90):.1f} "
              f"| max={sp.max():.1f} m/ano")

        # coerencia direcional local (vetores vizinhos apontam juntos?)
        # grade regular para strain
        xs = sorted(set(arr[:, 0])); ys = sorted(set(arr[:, 1]))
        gw, gh = len(xs), len(ys)
        U = np.full((gh, gw), np.nan); V = np.full((gh, gw), np.nan)
        xi = {v: i for i, v in enumerate(xs)}; yi = {v: i for i, v in enumerate(ys)}
        for k in np.nonzero(gl)[0]:
            U[yi[arr[k, 1]], xi[arr[k, 0]]] = arr[k, 2]
            V[yi[arr[k, 1]], xi[arr[k, 0]]] = arr[k, 3]
        Uf = median_filter(np.nan_to_num(U), 3); Vf = median_filter(np.nan_to_num(V), 3)
        valid = np.isfinite(U)
        resid = np.hypot(U - Uf, V - Vf)[valid]
        coer = float((resid < 10).mean())
        print(f"   coerencia local (resíduo <10 m vs mediana 3x3): {100*coer:.1f}%")

        if coer > 0.5:
            Us = gaussian_filter(Uf, 2); Vs = gaussian_filter(Vf, 2)
            d = STRIDE * RES
            dudy, dudx = np.gradient(Us, d); dvdy, dvdx = np.gradient(Vs, d)
            exx, eyy, exy = dudx, dvdy, 0.5 * (dudy + dvdx)
            e1 = 0.5 * (exx + eyy) + np.sqrt((0.5 * (exx - eyy)) ** 2 + exy ** 2)
            th1 = np.degrees(0.5 * np.arctan2(2 * exy, exx - eyy))  # eixo de e1
            np.save(str(Config.RESULTS_DIR / f'strain_e1_{y1}_{y2}.npy'), e1)
            np.save(str(Config.RESULTS_DIR / f'strain_theta1_{y1}_{y2}.npy'), th1)
            print(f"   strain: e1 mediano no gelo = {np.nanmedian(e1[median_filter(valid.astype(float),3)>0.5]):.2e} /ano "
                  f"(campos salvos em results/strain_*.npy)")

    out = Config.RESULTS_DIR / f'velocity_{y1}_{y2}.csv'
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['px_x', 'px_y', 'u_m', 'v_m', 'corr', 'peak_ratio', 'on_ice', 'on_stable', 'qc_ok'])
        for r, o in zip(rows, ok):
            w.writerow(r + [bool(o)])
    print(f"\nCSV: {out}")
    print("VEREDITO: viavel se decorrelacao no gelo < ~50% e terreno estavel ~0 m; "
          "senao, fallback = tracking manual de feicoes persistentes (Belvedere).")


if __name__ == '__main__':
    main()
