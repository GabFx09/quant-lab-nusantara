# Quant Lab Nusantara

Riset dan portofolio kuantitatif multi-pasar: **45 saham LQ45, 20 crypto, 12 pair forex**.
Website dibangun ulang otomatis setiap hari oleh GitHub Actions dan di-hosting di GitHub Pages.

## Cara kerja

1. **Data**: harga harian dari Yahoo Finance sejak 2010 (`quant/data.py`).
2. **Sinyal**: 7 sinyal per aset, yaitu momentum 12-1 dan 3 bulan, tren SMA 200, high 52 minggu, reversal 1 minggu, RSI oversold, dan volatilitas rendah (`quant/engine.py`).
3. **Normalisasi**: peringkat cross-sectional di dalam tiap pasar, skala −1..+1.
4. **Bobot walk-forward**: tiap 6 bulan, bobot = 50% sama rata + 50% IC-IR dari 3 tahun terakhir.
5. **Portofolio**: top-N (saham, crypto) atau long/short (forex), bobot inverse-volatility, target volatilitas, buffer peringkat untuk menekan turnover, dan filter tren BTC untuk crypto.
6. **Gabungan**: risk parity antar pasar, dihitung ulang bulanan.
7. **Eksekusi**: sinyal di close t, eksekusi di close t+1, sudah termasuk biaya transaksi.

## Menjalankan lokal

```bash
pip install -r requirements.txt
python build.py          # hasil di folder site/
python build.py --fast   # lewati uji ketahanan
```

## Struktur

| Path | Isi |
|---|---|
| `quant/` | mesin riset multi-faktor (config, data, engine, metrics, report) |
| `site_src/template.html` | template dashboard |
| `build.py` | menjalankan semua riset dan membangun `site/` |
| `run.py`, `seasonality.py`, `export_web.py`, `web/template.html` | riset awal "Buku Pola" (pola individual dan siklus kalender) |

Bersifat edukasi dan riset, bukan nasihat keuangan.
