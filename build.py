"""Bangun seluruh website ke folder site/.

  python build.py           # riset lengkap + website
  python build.py --fast    # tanpa uji ketahanan (lebih cepat, untuk cek tampilan)
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
os.chdir(ROOT)


def step(msg):
    print(f"\n== {msg}", flush=True)


def main():
    fast = "--fast" in sys.argv
    SITE.mkdir(exist_ok=True)

    step("Riset kuantitatif (quant/)")
    from quant.report import run_all
    data = run_all(full=not fast)
    tpl = (ROOT / "site_src" / "template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    (SITE / "index.html").write_text(tpl.replace("__DATA__", payload), encoding="utf-8")

    step("Buku Pola (riset awal)")
    for script in ("run.py", "seasonality.py", "export_web.py"):
        subprocess.run([sys.executable, script], check=True, stdout=subprocess.DEVNULL)
    body = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    body = body.replace('<span class="brand">Buku Pola</span>',
                        '<a class="brand" href="index.html" style="text-decoration:none;color:inherit">← Quant Lab</a>'
                        '<span class="brand">Buku Pola</span>')
    page = ('<!doctype html>\n<html lang="id">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n'
            '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
    (SITE / "pola.html").write_text(page, encoding="utf-8")
    (SITE / ".nojekyll").write_text("")
    for f in ("hasil_backtest.csv", "hasil_musiman.json"):
        shutil.copy(ROOT / f, SITE / f)
    print(f"\nSelesai: {SITE}  ({(SITE / 'index.html').stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
