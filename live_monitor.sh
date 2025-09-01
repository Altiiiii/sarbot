#!/usr/bin/env bash
cd ~/Documents/scalper
while true; do
  clear
  date -u
  echo "── last signals (son 10) ──"; tail -n 10 logs/incr_signals.csv
  echo
  echo "── last outcomes (son 10) ──"; tail -n 10 logs/incr_outcomes.csv
  echo
  echo "── disjoint TP1 stats ──"
  python3 - <<'PY'
import json, os
p="ptp1_stats.json"
if not os.path.exists(p):
    print("ptp1_stats.json yok (engine henüz yazmadı)")
else:
    d=json.load(open(p))
    print("updated_at:", d.get("updated_at"))
    b=d.get("buckets",{})
    order=["0-15m","15-60m","1-2h","2-4h","4-8h","8-12h","12-24h"]
    for k in order:
        v=b.get(k,{"hits":0,"trials":0})
        n=v.get("trials",0); h=v.get("hits",0)
        pct=f"{(100*h/n):.1f}%" if n else "-"
        print(f"{k:>6}  {h}/{n}  {pct}")
PY
  sleep 15
done
