"""Reproduce the combined analysis. Never launches experiments or writes raw data.

Use a Python environment with numpy and matplotlib; Node.js is also required.
Run from any directory: python -B 3e_results/analysis/reproduce.py
The frozen run ledger is verified, not regenerated. Each step has a log.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parent
STEPS=[
 ('verify_inputs','scope.py'),
 ('collection_provenance','check_collection_provenance.py'),
 ('analyze_3e','analyze_3e.py'),
 ('divergence','analyze_divergence.py'),
 ('skip_bias','analyze_skip_bias.py'),
 ('signals','analyze_signals.py'),
 ('sq1','analyze_sq1.py'),
 ('sq2','sq2/analyze_sq2.py'),
 ('overhead_by_arm','sq2/sq2_overhead_by_arm.py'),
 ('load_profile','sq2/cell_load_profile.py'),
 ('rungs','compare_state_aware_rungs.py'),
 ('ttft_decode','compare_ttft_decode_split.py'),
 ('sq3a','sq3/a/analyze_sq3a.py'),
 ('failure_taxonomy','sq3/a/analyze_sq3a_failure_taxonomy.py'),
 ('sq3b','sq3/b/analyze_sq3b.py'),
 ('bh','sq3/b/with_correction/apply_bh_correction.py'),
 ('routing_share','sq3/b/routing_share_vs_tail.py'),
 ('tail_nodes','sq3/b/tail_node_composition.py'),
 ('sensitivity_inputs','sq3/b/prepare_sensitivity.py'),
 ('sensitivity','sq3/b/reproduce_sensitivity.mjs'),
 ('figure_sq1','../../figures/analysis/sq1/plot_divergence_by_workload.py'),
 ('figure_overhead','../../figures/analysis/sq2/p1/plot_overhead_candidates.py'),
 ('figure_staleness','../../figures/analysis/sq2/p2/plot_staleness.py'),
 ('figure_prefill','../../figures/analysis/sq2/p3/plot_stale_vs_prefill.py'),
 ('figure_ladder','../../figures/analysis/sq3/a/ladder/plot_ladder_forest.py'),
 ('figure_dispatch','../../figures/analysis/sq3/b/dispatch/plot_dispatch_share.py'),
 ('report_tables','build_report_tables.py'),
 ('validate','validate.py'),
]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--from-step',choices=[s[0] for s in STEPS],help='Resume after a failed step; dependencies must already exist.')
    a=ap.parse_args(); started=a.from_step is None
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    env.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'thesis-combined-mpl'))
    env.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'thesis-combined-cache'))
    (ROOT/'logs').mkdir(exist_ok=True)
    for name,script in STEPS:
        if name==a.from_step: started=True
        if not started:continue
        cmd=(['node'] if script.endswith('.mjs') else [sys.executable,'-B'])+[str(ROOT/script)]
        t=time.monotonic(); print(f'{name}: starting',flush=True)
        with (ROOT/'logs'/f'{name}.log').open('w') as log:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f'{name} failed ({result.returncode}); inspect logs/{name}.log')
        print(f'{name}: passed ({time.monotonic()-t:.1f}s)',flush=True)

if __name__=='__main__':main()
