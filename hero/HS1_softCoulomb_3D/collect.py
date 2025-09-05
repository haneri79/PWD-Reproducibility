# -*- coding: utf-8 -*-
import re, pandas as pd, pathlib as P
pat = re.compile(r"(\w+)=([^\s]+)")
NEEDED = ["wall_time_s","n_cores","peak_ram_gb","energy_wh","error_mha","system_size"]
def parse_log(p):
    kv={}
    for line in P.Path(p).read_text().splitlines():
        for k,v in pat.findall(line):
            kv[k]=v
    for k in NEEDED:
        kv[k]=float(kv.get(k,"nan"))
    kv["offline_included"]=kv.get("offline_included","full")
    return kv
rows=[]
logdir=P.Path(__file__).parent / "logs"
for lp in sorted(logdir.glob("base_*.log")):
    tag=lp.stem.replace("base_","")
    b=parse_log(lp)
    pp=logdir/f"pwd_{tag}.log"
    if not pp.exists():
        print(f"[warn] missing PWD log for case {tag}, skip"); continue
    p=parse_log(pp)
    rows.append({
      "case_id":tag,
      "time_base":b["wall_time_s"], "time_pwd":p["wall_time_s"],
      "core_hours_base":b["wall_time_s"]*b["n_cores"]/3600.0,
      "core_hours_pwd":p["wall_time_s"]*p["n_cores"]/3600.0,
      "peak_ram_gb_base":b["peak_ram_gb"], "peak_ram_gb_pwd":p["peak_ram_gb"],
      "energy_wh_base":b["energy_wh"], "energy_wh_pwd":p["energy_wh"],
      "error_mha_base":b["error_mha"], "error_mha_pwd":p["error_mha"],
      "system_size":b["system_size"],
      "offline_included_base":b["offline_included"],
      "offline_included_pwd":p["offline_included"],
    })
if not rows:
    raise SystemExit("no paired logs found")
df=pd.DataFrame(rows)
out=P.Path(__file__).parent/"bench_hero.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"wrote {out}")
