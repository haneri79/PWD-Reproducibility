## 🚀 Hero Suite (HS1) — “hundreds×” measured
Reproduce and verify large-scale advantage fairly and reproducibly.

**Reproduce**
```bash
bash hero/HS1_softCoulomb_3D/run_base.sh
bash hero/HS1_softCoulomb_3D/run_pwd.sh
python hero/HS1_softCoulomb_3D/collect.py
python tools/verify_pwd.py --input hero/HS1_softCoulomb_3D/bench_hero.csv   --metrics time core_hours peak_ram_gb energy_wh   --primary-metric time   --delta-mha 0.5 --alpha 0.05   --speedup-threshold 100 --ci-lower-threshold 50   --bootstrap 2000 --seed 1234   --outdir results --figdir docs/img
```
**Pass criterion**: `GM×(time) ≥ 100` **AND** `95%CI lower ≥ 50` **AND** `TOST == True`
