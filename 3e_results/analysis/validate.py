"""Fail-fast acceptance checks for the merged analysis and frozen inputs."""
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from scope import ROOT, DATA, included_paths, verify

def rows(name):
    with (ROOT/name).open() as f:return list(csv.DictReader(f))

def main():
    checks=[]
    def check(ok, name):
        if not ok:raise AssertionError(name)
        checks.append(name)
    verify()
    ledger=rows('run_inclusion.csv');inc=[r for r in ledger if r['included']=='True']
    check(len(inc)==508 and sum(int(r['measured_requests']) for r in inc)==50800,'508 runs / 50,800 measured requests')
    check(Counter(r['verdict'] for r in inc)=={'clean':416,'suspect':92},'416 clean / 92 suspect runs')
    check(len(ledger)-len(inc)==1 and next(r for r in ledger if r['included']=='False')['metadata_path']=='least_connections/c4_sharegpt/rep1/run_meta.json','Only the historical LC ShareGPT c4 pilot is excluded')
    spec=importlib.util.spec_from_file_location('recorded_driver',DATA/'run_3e.py');driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    outcomes=Counter();bench_failed=0;truncated=0
    from analyze_3e import truthy_trunc
    for p in included_paths():
        m=json.loads(p.read_text());b=json.loads((p.parent/m['bench_result_file']).read_text())
        assert len(b['raw'])==b['num_prompts']==100
        assert sum(bool(r['success']) for r in b['raw'])==b['completed']
        bench_failed+=b['failed'];truncated+=sum(truthy_trunc(r.get('truncated')) for r in b['raw'])
        verdict,*_=driver.compute_verdict(p.parent/m['router_log_file'])
        assert verdict==m['verdict'],f'Verdict differs: {p}'
        logs=[json.loads(line) for line in (p.parent/m['router_log_file']).open() if line.strip()]
        measured=[r for r in logs if isinstance(r.get('request_id'),int) and r['request_id']>=b['num_warmup']]
        assert len(measured)==100
        outcomes.update(r['outcome'] for r in measured)
        assert sum(r['outcome']=='success' for r in measured)+sum(r['outcome']=='client_disconnect' for r in measured)==b['completed']
    check(bench_failed==2630 and truncated==0,'Request counts, success/disconnect reconciliation and zero truncation verified for every run')
    check(True,'Stored clean/suspect verdict reproduced from router logs for all 508 runs')
    check(outcomes['client_disconnect']==25,'25 client disconnects reported separately')
    for file,scope_field,scope,n in [('sq1/sq1_divergence.csv','clean','True',416),('sq1/sq1_divergence.csv','clean','False',92),('sq2/sq2_routing_overhead.csv','scope','all',508),('sq2/sq2_routing_overhead.csv','scope','clean',416)]:
        check(sum(int(r['n_runs']) for r in rows(file) if r[scope_field]==scope)==n,f'{file} {scope}: run count reconciles')
    completion=rows('sq3/a/sq3a_completion_rate.csv');latency=rows('sq3/b/sq3b_latency_throughput.csv')
    check(sum(int(r['n_runs']) for r in completion)==508 and sum(int(r['offered']) for r in completion)==50800,'SQ3a scope reconciles')
    check(sum(int(r['n_all']) for r in latency)==508 and sum(int(r['n_runs']) for r in latency)==416,'SQ3b scope reconciles')
    check(not any(r['arm']=='least_connections' and r['cell']=='c4_sgpt' for r in completion),'Pilot absent from completion tables')
    tax=[r for r in rows('sq3/a/sq3a_failure_taxonomy.csv') if r['block']=='grid_total']
    check(all(r['n_runs']=='508' and r['bench_offered']=='50800' for r in tax),'Taxonomy uses correct denominator')
    check(sum(int(r['count']) for r in tax if r['counted_as_bench_failure']=='True' and r['category']!='bench_failures')==2630,'Taxonomy categories reconcile with driver failures')
    profile=list(csv.reader((ROOT/'sq2/cell_load_profile.csv').open()))
    cells={r[1] for r in profile[1:] if r[0]=='throughput'}
    check(len(cells)==9 and 'c12_sgpt' in cells,'SQ2 load profile covers all nine cells')
    from scipy.stats import false_discovery_control
    bh=rows('sq3/b/with_correction/sq3b_bh_corrected.csv')
    families={r['family_B'] for r in bh}
    check(len(bh)==337 and len(families)==16,'337 comparisons and 16 definition-B families')
    for fam in families:
        group=[r for r in bh if r['family_B']==fam]
        expected=false_discovery_control([float(r['p_raw']) for r in group],method='bh')
        assert all(math.isclose(float(r['p_adj_B']),float(a),abs_tol=1e-12) for r,a in zip(group,expected))
    check(True,'Definition-B adjusted p-values independently reproduce with SciPy BH')
    share=[r for r in rows('sq3/b/sq3b_routing_share_vs_tail.csv') if r['metric']=='e2el_p95' and r['counted']=='True' and r['significant']=='True']
    check((sum(r['agrees']=='True' for r in share),len(share))==(13,14),'Routing share: 13 of 14 significant p95 differences agree (paper, supporting variants)')
    tail={(r['arm'],r['node']):r for r in rows('sq3/b/sq3b_tail_node_composition.csv') if r['cell']=='c12_sgpt'}
    disp={(r['arm'],r['node']):float(r['share_pooled']) for r in rows('sq3/b/sq3b_dispatch_share.csv') if r['cell']=='c12_sgpt'}
    check(all(abs(float(tail[k]['share'])-disp[k])<1e-6 for k in tail)
          and all(float(tail[(a,'pi-node-1')]['sent_to_full_node'])>0.5 for a in ('weighted_least_connections','state_aware'))
          and all(int(tail[(a,'pi-node-2')]['tail_requests'])+int(tail[(a,'pi-node-3')]['tail_requests'])>=0.9*int(tail[(a,'pi-node-1')]['tail_total'])
                  for a in ('weighted_least_connections','state_aware')),
          'Request logs: at ShareGPT c12 most Pi 5 dispatches found all slots busy and the slowest 5% ran almost entirely on the Pi 4Bs')
    # Historical outputs computed TPOT per streamed text chunk; analyze_3e.tpot_ms now
    # uses the server token count. Chunks and tokens differ only for ShareGPT, so those
    # TPOT rows are expected to change and every other historical row must still match.
    def historical(old):
        return [r for r in old if not (r['metric'].startswith('tpot') and r['cell'].endswith('_sgpt'))]
    for c in ('wlc','wrr'):
        report=rows(f'sq3/b/sq3b_vs_{c}_state_aware_report_table.csv')
        check(len(report)==54 and len({(r['cell'],r['metric']) for r in report})==54,f'{c}: report table contains six metrics across nine cells')
        for r in report:
            if r['metric']=='completion_rate_pp':assert r['run_scope']=='all'
            else:assert r['run_scope']=='clean'
        f=f'sq3b_vs_{c}_state_aware_e2el_p50_p95_tpot_p50_multipower.csv'
        new={(r['cell'],r['metric']):r for r in rows('sq3/b/'+f)}
        check(len(new)==27,f'{c}: 27 multi-power rows')
        old=historical(list(csv.DictReader((DATA/'sq3/b'/f).open())))
        check(len(old)==20,f'{c}: 20 historical multi-power rows outside ShareGPT TPOT')
        for r in old:
            for k in ['detectable_diff_80','detectable_diff_90','detectable_diff_95']:
                a,b=r[k],new[(r['cell'],r['metric'])][k]
                assert (not a and not b) or (a and b and math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-9))
        check(True,f'{c}: all 20 historical multi-power rows outside ShareGPT TPOT reproduce')
        f=f'sq3b_vs_{c}_state_aware_sensitivity.csv'
        new={(r['cell'],r['metric']):r for r in rows('sq3/b/'+f)}
        old=historical(list(csv.DictReader((DATA/'sq3/b'/f).open())))
        check(len(old)==40,f'{c}: 40 historical sensitivity rows outside ShareGPT TPOT')
        for r in old:
            for k in ['observed_pool_min_p','detectable_diff_80','pooled_sd','sd_base','sd_cand']:
                a,b=r[k],new[(r['cell'],r['metric'])][k]
                assert (not a and not b) or (a and b and math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-10))
        check(True,f'{c}: all 40 historical six-metric D80/SD/observed-pool values outside ShareGPT TPOT reproduce')
    figs=ROOT.parents[1]/'figures'/'analysis'
    for glob in ['sq1/*.png','sq2/p1/*.png','sq2/p2/*.png','sq2/p3/*.png','sq3/a/ladder/*.png','sq3/b/dispatch/*.png']:
        check(any(figs.glob(glob)),f'Figure artifacts exist: figures/analysis/{glob}')
    artifacts=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.suffix in ('.csv','.json','.pdf','.png','.py','.mjs','.md') and '__pycache__' not in p.parts and p.name not in ('validation.json','artifact_manifest.csv'):
            artifacts.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    with (ROOT/'artifact_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(artifacts)
    (ROOT/'validation.json').write_text(json.dumps(dict(status='passed',checks=checks,router_outcomes=dict(outcomes),artifacts=len(artifacts)),indent=2)+'\n')
    print(f'PASS: {len(checks)} acceptance checks; {len(artifacts)} artifact hashes written.')

if __name__=='__main__':main()
