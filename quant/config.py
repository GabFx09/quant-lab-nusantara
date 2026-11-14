"""Konfigurasi universe, biaya, dan parameter portofolio."""

LQ45 = ["ACES", "ADMR", "ADRO", "AKRA", "AMMN", "AMRT", "ANTM", "ARTO", "ASII", "BBCA", "BBNI", "BBRI",
        "BBTN", "BMRI", "BRIS", "BRPT", "CPIN", "CTRA", "ESSA", "EXCL", "GOTO", "ICBP", "INCO", "INDF",
        "INKP", "ISAT", "ITMG", "JPFA", "JSMR", "KLBF", "MAPA", "MAPI", "MBMA", "MDKA", "MEDC", "PGAS",
        "PGEO", "PTBA", "SIDO", "SMGR", "SMRA", "TLKM", "TOWR", "UNTR", "UNVR"]
CRYPTO = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "TRX", "AVAX", "LINK", "DOT", "LTC", "BCH",
          "XLM", "ATOM", "ETC", "NEAR", "FIL", "HBAR", "XMR"]
FOREX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "EURJPY", "GBPJPY",
         "EURGBP", "AUDJPY", "EURCHF"]

START = "2010-01-01"
REFIT = 126            # optimasi ulang bobot sinyal tiap ~6 bulan (dalam baris kalender kelas)
SHRINK = 0.5           # bobot akhir = 50% bobot IC + 50% bobot sama (prior)

CLASSES = {
    "saham": dict(
        label="Saham IDX", sub="LQ45", tickers=[t + ".JK" for t in LQ45], strip=".JK",
        cost=0.0020, periods=252, long_short=False, top_n=10, target_vol=0.15, max_gross=1.0,
        hold=5, buffer=2, warmup=756, ic_lookback=756, min_assets=15, regime=None, bad_ret=0.5,
        benchmark="^JKSE", benchmark_label="IHSG", currency="IDR"),
    "crypto": dict(
        label="Crypto", sub="20 coin besar", tickers=[t + "-USD" for t in CRYPTO], strip="-USD",
        cost=0.0010, periods=365, long_short=False, top_n=5, target_vol=0.35, max_gross=1.0,
        hold=14, buffer=2, warmup=730, ic_lookback=730, min_assets=8, regime="BTC-USD", bad_ret=1.0,
        benchmark="BTC-USD", benchmark_label="Bitcoin", currency="USD"),
    "forex": dict(
        label="Forex", sub="12 pair", tickers=[t + "=X" for t in FOREX], strip="=X",
        cost=0.0002, periods=260, long_short=True, top_n=3, target_vol=0.08, max_gross=4.0,
        hold=10, buffer=2, warmup=756, ic_lookback=756, min_assets=10, regime=None, bad_ret=0.1,
        benchmark=None, benchmark_label=None, currency="USD"),
}

# nama, label, penjelasan. Semua diorientasikan supaya prior-nya "nilai tinggi = beli".
SIGNALS = {
    "mom_12_1": ("Momentum 12-1", "Return 12 bulan, tidak termasuk bulan terakhir, dibagi volatilitas"),
    "mom_3m": ("Momentum 3 bulan", "Return 63 hari dibagi volatilitas"),
    "trend_200": ("Tren SMA 200", "Jarak harga dari SMA 200, dibagi volatilitas"),
    "high_52w": ("Dekat high 52 minggu", "Jarak harga dari harga tertinggi 252 hari"),
    "rev_1w": ("Reversal 1 minggu", "Kebalikan return 5 hari: yang baru turun diharapkan memantul"),
    "rsi_rev": ("RSI oversold", "50 − RSI(5): makin oversold makin tinggi"),
    "low_vol": ("Volatilitas rendah", "Kebalikan volatilitas 60 hari"),
}
