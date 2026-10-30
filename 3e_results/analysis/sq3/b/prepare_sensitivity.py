"""Extracts run-level values for the sensitivity and attainable-p calculations."""
import csv
import json
import math
import statistics
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scope import included_paths, CELLS
from analyze_3e import pct, truthy_trunc, col_label, metric_ms

METRICS = ['e2el_p50','e2el_p95','tpot_p50','tpot_p95','throughput_req_s','completion_rate_pp']
BASES = {'wlc':'weighted_least_connections','wrr':'weighted_round_robin'}

def main():
    values = {}
    for p in included_paths():
        m=json.loads(p.read_text()); arm=m.get('arm') or m['policy']
        if arm not in {*BASES.values(),'state_aware'}: continue
        b=json.loads((p.parent/m['bench_result_file']).read_text())
        clean=m['verdict']=='clean'
        cell=col_label(m['workload'],m['concurrency'])
        for metric in METRICS:
            if metric=='completion_rate_pp': v=100*sum(bool(r['success']) for r in b['raw'])/b['num_prompts']
            elif not clean: continue
            elif metric=='throughput_req_s': v=b[metric]
            else:
                stem,q=metric.split('_p')
                xs=[metric_ms(r,stem) for r in b['raw'] if r['success'] and not truthy_trunc(r.get('truncated')) and isinstance(metric_ms(r,stem),(float,int))]
                v=pct(xs,int(q))/(1000 if stem=='e2el' else 1)
            if v is not None: values.setdefault((arm,cell,metric),[]).append(v)
    out=[]
    for comparison,base in BASES.items():
        source={(r['cell'],r['metric']):r for r in csv.DictReader((HERE/f'sq3b_vs_{comparison}.csv').open()) if r['arm']=='state_aware'}
        for w,c in CELLS:
            cell=col_label(w,c)
            for metric in METRICS:
                a=values[(base,cell,metric)];b=values[('state_aware',cell,metric)]
                ma,mb=statistics.median(a),statistics.median(b)
                sa,sb=statistics.stdev(a),statistics.stdev(b)
                src=source.get((cell,metric));scale=1000 if metric.startswith('e2el') else 1
                if src:
                    assert len(a)==int(src['n_base']) and len(b)==int(src['n_cand'])
                    assert math.isclose(ma,float(src['base_median'])/scale,abs_tol=1e-9)
                    assert math.isclose(mb,float(src['cand_median'])/scale,abs_tol=1e-9)
                out.append(dict(comparison=comparison,source_file=f'sq3b_vs_{comparison}.csv',
                    source_row_present=bool(src),base_arm=base,cand_arm='state_aware',cell=cell,
                    workload=w,concurrency=c,metric=metric,
                    unit='s' if metric.startswith('e2el') else 'ms' if metric.startswith('tpot') else 'pp' if metric=='completion_rate_pp' else 'req/s',
                    run_scope='all' if metric=='completion_rate_pp' else 'clean',
                    n_base=len(a),n_cand=len(b),sd_base=sa,sd_cand=sb,
                    pooled_sd=math.sqrt(((len(a)-1)*sa*sa+(len(b)-1)*sb*sb)/(len(a)+len(b)-2)),
                    base_median=ma,cand_median=mb,observed_diff=mb-ma,base_values=a,cand_values=b))
    (HERE/'sensitivity_inputs.json').write_text(json.dumps(out,indent=2)+'\n')
    print(f'Extracted {len(out)} comparisons from the shared inclusion ledger.')

if __name__=='__main__': main()
