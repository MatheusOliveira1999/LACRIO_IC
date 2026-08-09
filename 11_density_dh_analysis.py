"""
11_density_dh_analysis.py - Correlacao densidade de fendas x dH (atividade 3 do edital)

Metodo (validado adversarialmente em 09/08/2026; ver vault "Conclusoes - Achados
relevantes (2026-08)"):
  1. Grade comum de 2 m: densidade de fendas = media da mascara 8cm em blocos 25x25 px;
     DEMs (~31cm) reamostrados bilinearmente para a mesma grade (WarpedVRT).
  2. Dominio "gelo" SEM circularidade: proxy por afinamento < -0.5 m em intervalos de
     anos COMPLEMENTARES ao par analisado (ex.: para 2016-17 usa o afinamento 2017-19).
  3. Blocos de 100 m (50x50 celulas de 2 m): dH mediano, densidade media do ano inicial,
     elevacao media. Filtros: cobertura valida >50%, elevacao 25-250 m.
  4. Estatistica por par de anos: Spearman bruto, Spearman PARCIAL controlando elevacao,
     Mann-Whitney gelo-sem-fenda (<0.1%) vs top-quartil de densidade.
  5. Tabela por bandas de 50 m (mostra que o sinal e lateral, nao altitudinal).
  6. Validacao de co-registro: dH mediano em blocos estaveis fora do gelo.

Saidas: results/density_dh_blocks.csv | results/density_dh_bands.csv |
        results/density_dh_summary.json

Uso: python 11_density_dh_analysis.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy.stats import spearmanr, rankdata, t as tdist, mannwhitneyu

from config import Config

warnings.simplefilter("ignore")

YEARS = [2016, 2017, 2018, 2019]
PAIRS = [(2016, 2017), (2017, 2018), (2018, 2019)]
AGG = 25          # 25 px de 8 cm -> celula de 2 m
BLOCK = 50        # 50 celulas de 2 m -> bloco de 100 m
ELEV_LO, ELEV_HI = 25.0, 250.0
ICE_THIN_THR = -0.5   # m: afinamento que define "gelo" no proxy complementar
BAND = 50.0


def load_grids():
    """Carrega densidade (2 m) e DEM (2 m) de cada ano na grade da mascara 8cm de 2016."""
    ref_path = Config.RESULTS_DIR / "2016" / "crevasses_mask_2016_8cm.tif"
    with rasterio.open(ref_path) as ref:
        h2, w2 = ref.height // AGG, ref.width // AGG
        transform2m = ref.transform * ref.transform.scale(AGG, AGG)
        crs = ref.crs

    dens, dem = {}, {}
    for y in YEARS:
        mpath = Config.RESULTS_DIR / str(y) / f"crevasses_mask_{y}_8cm.tif"
        with rasterio.open(mpath) as src:
            a = src.read(1, out_shape=(h2 * AGG, w2 * AGG))[: h2 * AGG, : w2 * AGG]
        a = (a > 0).astype(np.float32).reshape(h2, AGG, w2, AGG).mean(axis=(1, 3))
        dens[y] = a

        dpath = Config.get_dem_path(y)
        with rasterio.open(dpath) as src:
            with WarpedVRT(src, crs=crs, transform=transform2m, width=w2, height=h2,
                           resampling=Resampling.bilinear, nodata=np.nan) as vrt:
                z = vrt.read(1).astype(np.float64)
        z[z < -1000] = np.nan
        dem[y] = z
        print(f"  {y}: grade 2m ok ({w2}x{h2})")
    return dens, dem


def blocks(a, f, B=BLOCK):
    H, W = a.shape
    h, w = H // B * B, W // B * B
    return f(a[:h, :w].reshape(h // B, B, w // B, B), axis=(1, 3))


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a, b):
        A = np.vstack([b, np.ones_like(b)]).T
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]

    r = np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1]
    n = len(x)
    tt = r * np.sqrt((n - 3) / (1 - r * r))
    return r, 2 * tdist.sf(abs(tt), n - 3)


def main():
    print("=" * 70)
    print("DENSIDADE DE FENDAS x dH — blocos de 100 m, dominio gelo (proxy)")
    print("=" * 70)
    dens, dem = load_grids()

    ice_proxy = {
        (2016, 2017): (dem[2019] - dem[2017]) < ICE_THIN_THR,
        (2017, 2018): ((dem[2017] - dem[2016]) < ICE_THIN_THR)
                      | ((dem[2019] - dem[2018]) < ICE_THIN_THR),
        (2018, 2019): (dem[2018] - dem[2016]) < ICE_THIN_THR,
    }

    summary = {"config": {"block_m": 100, "cell_m": 2, "elev_range_m": [ELEV_LO, ELEV_HI],
                          "ice_proxy_thr_m": ICE_THIN_THR, "density_year": "inicial",
                          "threshold_mask": 0.6, "modelo": "ep.107 TTA blend"}}
    block_rows = ["pair,block_row,block_col,elev_m,dh_median_m,dens_pct,ice"]
    band_rows = ["pair,band_lo_m,band_hi_m,n_cells,dh_median_m,dens_t1_pct,ddens_pp"]

    for (t1, t2) in PAIRS:
        ice = ice_proxy[(t1, t2)]
        both = np.isfinite(dem[t1]) & np.isfinite(dem[t2]) & ice
        dh_b = blocks(np.where(both, dem[t2] - dem[t1], np.nan), np.nanmedian)
        z_b = blocks(np.where(both, dem[t1], np.nan), np.nanmean)
        d_b = blocks(np.where(both, dens[t1], np.nan), np.nanmean) * 100
        v_b = blocks(both.astype(float), np.nanmean)

        ok = (v_b > 0.5) & np.isfinite(dh_b) & (z_b >= ELEV_LO) & (z_b < ELEV_HI)
        x, h, e = d_b[ok], dh_b[ok], z_b[ok]
        n = int(ok.sum())

        rho, p = spearmanr(x, h)
        prho, pp = partial_spearman(x, h, e)
        lo = h[x < 0.1]
        q75 = float(np.percentile(x, 75))
        hi = h[x >= q75]
        mwu = mannwhitneyu(hi, lo, alternative="two-sided") if len(lo) > 5 else None

        pair_key = f"{t1}-{t2}"
        summary[pair_key] = {
            "n_blocks": n, "rho": round(float(rho), 4), "p": float(p),
            "rho_partial_elev": round(float(prho), 4), "p_partial": float(pp),
            "dH_med_sem_fenda_m": round(float(np.median(lo)), 3) if len(lo) else None,
            "n_sem_fenda": int(len(lo)),
            "dens_q75_pct": round(q75, 3),
            "dH_med_top_quartil_m": round(float(np.median(hi)), 3),
            "n_top_quartil": int(len(hi)),
            "efeito_extra_m": round(float(np.median(hi) - np.median(lo)), 3) if len(lo) else None,
            "p_mwu": float(mwu.pvalue) if mwu else None,
        }
        s = summary[pair_key]
        print(f"\n{t1}->{t2}: n={n} blocos | rho={rho:+.3f} (p={p:.1e}) | "
              f"parcial|elev rho={prho:+.3f} (p={pp:.1e})")
        print(f"  gelo sem fenda: dH={s['dH_med_sem_fenda_m']} m (n={s['n_sem_fenda']}) | "
              f"top-quartil (>={q75:.2f}%): dH={s['dH_med_top_quartil_m']} m "
              f"(n={s['n_top_quartil']}) | efeito extra={s['efeito_extra_m']} m | "
              f"MWU p={s['p_mwu']:.1e}" if mwu else "  (poucos blocos sem fenda)")

        rr, cc = np.where(ok)
        for r_, c_ in zip(rr, cc):
            block_rows.append(f"{pair_key},{r_},{c_},{z_b[r_, c_]:.1f},"
                              f"{dh_b[r_, c_]:.3f},{d_b[r_, c_]:.4f},1")

        # bandas de 50 m (mostra ausencia de sinal altitudinal)
        z = dem[t1]
        dh = dem[t2] - dem[t1]
        for b0 in np.arange(ELEV_LO, ELEV_HI, BAND):
            sel = both & (z >= b0) & (z < b0 + BAND)
            if sel.sum() < 500:
                continue
            band_rows.append(
                f"{pair_key},{b0:.0f},{b0 + BAND:.0f},{int(sel.sum())},"
                f"{np.nanmedian(dh[sel]):.3f},{np.nanmean(dens[t1][sel]) * 100:.4f},"
                f"{(np.nanmean(dens[t2][sel]) - np.nanmean(dens[t1][sel])) * 100:.4f}")

    # validacao de co-registro: blocos fora do gelo com densidade ~0
    print("\n" + "-" * 70)
    print("Validacao de co-registro (dH em blocos estaveis fora do gelo):")
    stable = {}
    for (t1, t2) in PAIRS:
        ice = ice_proxy[(t1, t2)]
        both = np.isfinite(dem[t1]) & np.isfinite(dem[t2]) & (~ice)
        dh_b = blocks(np.where(both, dem[t2] - dem[t1], np.nan), np.nanmedian)
        d_b = blocks(np.where(both, dens[t1], np.nan), np.nanmean) * 100
        v_b = blocks(both.astype(float), np.nanmean)
        ok = (v_b > 0.5) & np.isfinite(dh_b) & (d_b < 0.05)
        med = float(np.nanmedian(dh_b[ok]))
        stable[f"{t1}-{t2}"] = {"n_blocks": int(ok.sum()), "dh_median_m": round(med, 3)}
        print(f"  {t1}->{t2}: dH mediano = {med:+.3f} m (n={int(ok.sum())} blocos)")
    summary["coregistro_terreno_estavel"] = stable

    out = Config.RESULTS_DIR
    (out / "density_dh_blocks.csv").write_text("\n".join(block_rows) + "\n")
    (out / "density_dh_bands.csv").write_text("\n".join(band_rows) + "\n")
    with open(out / "density_dh_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaidas: {out}/density_dh_{{blocks,bands}}.csv + density_dh_summary.json")


if __name__ == "__main__":
    main()
