"""
19_teste_deslocamento_dh.py - As faixas de "ganho de altura" no dH vêm do relevo das
fendas sendo carregado pelo gelo?

dH comum (euleriano) = MDE(t2, x) - MDE(t1, x): mesmo ponto do chão.
dH deslocado (lagrangiano) = MDE(t2, x + d) - MDE(t1, x): mesmo pedaço de gelo, onde d é
o deslocamento anual do gelo.

Se o sobe-e-desce de pequena escala vem das fendas/cristas passando pelo pixel, ele deve
diminuir quando o MDE de t2 é deslocado por d, e o deslocamento que mais "alisa" o dH
(busca independente, só com MDEs) deve coincidir com a velocidade medida nas fotos
(15_velocity_strain.py, feature tracking no ortomosaico).

A comparação usa o dH "passa-alta" (dH menos sua média móvel de ~100 m): o termo de
declividade (o pedaço de gelo desce a encosta ao andar) muda a média do dH deslocado,
mas não a textura de pequena escala, que é o que está em teste.

Região avaliada: gelo a > 150 m do limite do glaciar e a < 150 m de algum ponto de
velocidade do interior (onde a velocidade medida é representativa).

Saídas: results/teste_deslocamento_dh_{t1}_{t2}.json e
        results/figures/relatorio/teste_deslocamento_dh_{t1}_{t2}.png
Uso: python 19_teste_deslocamento_dh.py [t1 t2]   (padrão: 2018 2019)
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy.ndimage import distance_transform_edt, uniform_filter
from shapely.geometry import shape
from rasterio.features import geometry_mask

ROOT = Path(__file__).resolve().parent
Q = ROOT / "results/qgis"
SUAV = 51          # px de 2 m -> média móvel de ~100 m para o passa-alta
BORDA_M = 150
PERTO_M = 150
BUSCA = 70         # px de busca em cada eixo (±140 m)


def carrega_mde(ano, T, shape_, crs):
    nome = f"Schiaparelli_DEM_{ano}.tif" if ano < 2020 else f"schiaparelli_DEM_{ano}.tif"
    with rasterio.open(ROOT / "Schiaparelli_glacier" / nome) as src:
        with WarpedVRT(src, crs=crs, transform=T, width=shape_[1], height=shape_[0],
                       resampling=Resampling.bilinear, nodata=np.nan) as vrt:
            z = vrt.read(1).astype(np.float64)
    z[z < -1000] = np.nan
    return z


def passa_alta(a):
    ok = np.isfinite(a)
    num = uniform_filter(np.where(ok, a, 0.0), SUAV)
    den = uniform_filter(ok.astype(float), SUAV)
    media = np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)
    return a - media


def desloca(z, dx, dy):
    """Retorna z2 tal que z2[i, j] = z[i + dy, j + dx] (NaN fora)."""
    out = np.full_like(z, np.nan)
    H, W = z.shape
    i0, i1 = max(0, -dy), min(H, H - dy)
    j0, j1 = max(0, -dx), min(W, W - dx)
    out[i0:i1, j0:j1] = z[i0 + dy:i1 + dy, j0 + dx:j1 + dx]
    return out


def main():
    t1, t2 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2018, 2019)
    with rasterio.open(Q / "dh_2018_2019.tif") as r:
        T, shp, crs = r.transform, r.shape, r.crs
        gelo = r.read(1) != r.nodata
    z1, z2 = carrega_mde(t1, T, shp, crs), carrega_mde(t2, T, shp, crs)
    res = T.a

    # velocidade (grade de 2 m idêntica; +v = sul)
    rows = [x for x in csv.DictReader(open(ROOT / "results/velocity_2018_2019.csv"))
            if x["qc_ok"] == "True" and x["on_ice"] == "True"]
    px = np.array([int(float(x["px_x"])) for x in rows])
    py = np.array([int(float(x["px_y"])) for x in rows])
    u = np.array([float(x["u_m"]) for x in rows])
    v = np.array([float(x["v_m"]) for x in rows])
    sp = np.hypot(u, v)

    geo = shape(json.load(open(Q / "glacier_outline.geojson"))["features"][0]["geometry"])
    dentro = ~geometry_mask([geo], out_shape=shp, transform=T)
    dist_borda = distance_transform_edt(dentro) * res
    interior_pt = dist_borda[py, px] > 100
    u_med, v_med = np.median(u[interior_pt]), np.median(v[interior_pt])

    perto = np.zeros(shp, bool)
    perto[py[interior_pt], px[interior_pt]] = True
    perto = distance_transform_edt(~perto) * res < PERTO_M
    regiao = gelo & dentro & (dist_borda > BORDA_M) & perto & np.isfinite(z1)

    # recorta tudo na caixa da região avaliada (+ margem para a busca e o filtro)
    ys_, xs_ = np.where(regiao)
    mg = BUSCA + SUAV
    ra, rb = max(ys_.min() - mg, 0), min(ys_.max() + mg, shp[0])
    ca, cb = max(xs_.min() - mg, 0), min(xs_.max() + mg, shp[1])
    z1, z2 = z1[ra:rb, ca:cb], z2[ra:rb, ca:cb]
    regiao, gelo, dentro = regiao[ra:rb, ca:cb], gelo[ra:rb, ca:cb], dentro[ra:rb, ca:cb]
    T = T * T.translation(ca, ra)

    def textura(dx, dy):
        d = desloca(z2, dx, dy) - z1
        hp = passa_alta(np.where(regiao, d, np.nan))
        m = regiao & np.isfinite(hp)
        return np.sqrt(np.nanmean(hp[m] ** 2)), m.sum()

    # busca grossa (passo 4 px = 8 m) e refinamento (passo 1 px)
    melhor = (np.inf, 0, 0)
    for dy in range(-BUSCA, BUSCA + 1, 4):
        for dx in range(-BUSCA, BUSCA + 1, 4):
            s, n = textura(dx, dy)
            if n > 0.6 * regiao.sum() and s < melhor[0]:
                melhor = (s, dx, dy)
    for dy in range(melhor[2] - 4, melhor[2] + 5):
        for dx in range(melhor[1] - 4, melhor[1] + 5):
            s, n = textura(dx, dy)
            if n > 0.6 * regiao.sum() and s < melhor[0]:
                melhor = (s, dx, dy)
    s_best, dx_best, dy_best = melhor

    dx_vel, dy_vel = int(round(u_med / res)), int(round(v_med / res))
    s0, _ = textura(0, 0)
    s_vel, _ = textura(dx_vel, dy_vel)
    s_contra, _ = textura(-dx_vel, -dy_vel)
    s_perp, _ = textura(-dy_vel, dx_vel)

    def frac_ganho(dx, dy):
        d = desloca(z2, dx, dy) - z1
        hp = passa_alta(np.where(regiao, d, np.nan))
        m = regiao & np.isfinite(hp)
        return float(np.mean(hp[m] > 2.0)), float(np.mean(hp[m] < -2.0))

    out = {
        "par": f"{t1}-{t2}",
        "pixels_avaliados": int(regiao.sum()),
        "velocidade_fotos_interior_mediana_m": {"u_leste": float(u_med), "v_sul": float(v_med),
                                               "modulo": float(np.hypot(u_med, v_med)),
                                               "n_pontos": int(interior_pt.sum())},
        "deslocamento_que_mais_alisa_mde_m": {"leste": dx_best * res, "sul": dy_best * res,
                                              "modulo": float(np.hypot(dx_best, dy_best) * res)},
        "textura_rms_m": {"sem_deslocar": float(s0), "deslocado_pela_velocidade": float(s_vel),
                          "melhor_deslocamento": float(s_best),
                          "controle_sentido_contrario": float(s_contra),
                          "controle_perpendicular": float(s_perp)},
        "fracao_pixels_passa_alta_>+2m_<-2m": {"sem_deslocar": frac_ganho(0, 0),
                                               "deslocado_pela_velocidade": frac_ganho(dx_vel, dy_vel)},
    }
    # ---- teste LOCAL: em cada ponto do interior, janela de 128 m, deslocamento medido ali
    W = 32  # meia janela (64 px = 128 m, mesmo tamanho do chip do feature tracking)
    pxl, pyl = px - ca, py - ra
    loc = []
    for k in np.where(interior_pt)[0]:
        i, j = pyl[k], pxl[k]
        if i - W < 0 or j - W < 0 or i + W > z1.shape[0] or j + W > z1.shape[1]:
            continue
        a1 = z1[i - W:i + W, j - W:j + W]
        if np.isnan(a1).mean() > 0.2:
            continue

        def sd(dx, dy):
            ii, jj = i + dy, j + dx
            if ii - W < 0 or jj - W < 0 or ii + W > z2.shape[0] or jj + W > z2.shape[1]:
                return np.inf
            d = z2[ii - W:ii + W, jj - W:jj + W] - a1
            return np.nanstd(d) if np.isfinite(d).mean() > 0.8 else np.inf

        dxk, dyk = int(round(u[k] / res)), int(round(v[k] / res))
        melhor_l = min(((sd(dx, dy), dx, dy) for dy in range(-BUSCA, BUSCA + 1, 2)
                        for dx in range(-BUSCA, BUSCA + 1, 2)), key=lambda t: t[0])
        loc.append((sd(0, 0), sd(dxk, dyk), melhor_l[0],
                    np.hypot(melhor_l[1] - dxk, melhor_l[2] - dyk) * res, sp[k]))
    loc = np.array([l for l in loc if np.all(np.isfinite(l))])
    out["teste_local_janela_128m"] = {
        "n_pontos": int(len(loc)),
        "sobe_desce_mediano_m": {"sem_deslocar": float(np.median(loc[:, 0])),
                                 "deslocado_pela_velocidade_local": float(np.median(loc[:, 1])),
                                 "melhor_deslocamento_local": float(np.median(loc[:, 2]))},
        "fracao_pontos_que_alisam_com_velocidade_local": float(np.mean(loc[:, 1] < loc[:, 0])),
        "reducao_mediana_pct": float(np.median(100 * (1 - loc[:, 1] / loc[:, 0]))),
        "distancia_mediana_entre_melhor_deslocamento_mde_e_velocidade_fotos_m": float(np.median(loc[:, 3])),
        "fracao_com_distancia_<=_10m": float(np.mean(loc[:, 3] <= 10)),
    }
    np.save(ROOT / f"results/teste_deslocamento_dh_{t1}_{t2}_local.npy", loc)

    (ROOT / f"results/teste_deslocamento_dh_{t1}_{t2}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))

    # figura: (a) exemplo de janela antes/depois, (b) sobe-e-desce por ponto, (c) concordância MDE x fotos
    fig = plt.figure(figsize=(11.69, 4.4), dpi=300)
    gs = fig.add_gridspec(1, 4, width_ratios=[0.8, 0.8, 1.1, 1.1], wspace=0.35, left=0.03, right=0.98,
                          top=0.80, bottom=0.22)
    # exemplo: ponto com redução próxima da mediana
    red = 1 - loc[:, 1] / loc[:, 0]
    ks = [k for k in np.where(interior_pt)[0]]
    idx_ex = int(np.argsort(np.abs(red - np.median(red)))[0])
    k = ks[idx_ex] if len(ks) == len(loc) else ks[0]
    i, j = pyl[k], pxl[k]
    dxk, dyk = int(round(u[k] / res)), int(round(v[k] / res))
    ex0 = z2[i - W:i + W, j - W:j + W] - z1[i - W:i + W, j - W:j + W]
    ex1 = z2[i + dyk - W:i + dyk + W, j + dxk - W:j + dxk + W] - z1[i - W:i + W, j - W:j + W]
    for c, (ex, tit) in enumerate([(ex0, "(a) Mesmo ponto\ndo chão"),
                                   (ex1, f"(b) Mesmo pedaço de gelo\n(2019 movido {np.hypot(u[k], v[k]):.0f} m)")]):
        ax = fig.add_subplot(gs[c])
        im = ax.imshow(ex - np.nanmedian(ex), cmap="RdBu", vmin=-6, vmax=6, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(tit, fontsize=8.5, loc="left")
        ax.set_xlabel(f"sobe-e-desce: {np.nanstd(ex):.1f} m".replace(".", ","), fontsize=8)
    cax = fig.add_axes([0.05, 0.115, 0.33, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("Mudança de altura em torno da mediana da janela (m); janela de 128 m", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[2])
    ax.scatter(loc[:, 0], loc[:, 1], s=12, color="#0072B2", edgecolors="white", linewidths=0.3)
    lim = [0, np.nanmax(loc[:, :2]) * 1.05]
    ax.plot(lim, lim, color="#555", ls=":", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Sobe-e-desce no mesmo ponto do chão (m)", fontsize=8)
    ax.set_ylabel("Sobe-e-desce no mesmo pedaço de gelo (m)", fontsize=8)
    ax.tick_params(labelsize=7.5); ax.grid(alpha=0.3, lw=0.4)
    ax.set_title(f"(c) {100 * np.mean(loc[:, 1] < loc[:, 0]):.0f}% dos {len(loc)} pontos ficam mais lisos\n"
                 "(abaixo da linha pontilhada)", fontsize=8.5, loc="left")

    ax = fig.add_subplot(gs[3])
    ax.hist(loc[:, 3], bins=np.arange(0, max(60, loc[:, 3].max()) + 5, 5), color="#009E73", edgecolor="white")
    ax.set_xlabel("Distância entre o deslocamento que mais alisa\no mapa de altura e a velocidade das fotos (m)",
                  fontsize=8)
    ax.set_ylabel("Pontos", fontsize=8)
    ax.tick_params(labelsize=7.5); ax.grid(alpha=0.3, lw=0.4, axis="y")
    ax.set_title(f"(d) Mapa de altura e fotos concordam:\nmediana {np.median(loc[:, 3]):.0f} m de diferença "
                 f"(busca de ±{BUSCA * res:.0f} m)", fontsize=8.5, loc="left")
    fig.text(0.03, 0.93, f"Teste: as faixas de ganho e perda vêm do relevo das fendas carregado pelo gelo? "
             f"({t1}→{t2}, {len(loc)} pontos do interior)", fontsize=12, fontweight="bold")
    fig.savefig(ROOT / f"results/figures/relatorio/teste_deslocamento_dh_{t1}_{t2}.png", dpi=300,
                facecolor="white")

if __name__ == "__main__":
    main()
