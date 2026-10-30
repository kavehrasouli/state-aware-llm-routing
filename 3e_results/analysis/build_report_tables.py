"""Assemble report-ready tables without recomputing or selecting test families."""
import csv
import json
import statistics
from pathlib import Path
from scope import ROOT, included_paths

def read(name):
    with (ROOT/name).open() as f:return list(csv.DictReader(f))

def write(name,rows):
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    bh=read('sq3/b/with_correction/sq3b_bh_corrected.csv')
    corrected={(r['base_arm'],r['arm'],r['cell'],r['metric']):r for r in bh}
    ladder=read('sq3/a/sq3a_ladder_completion.csv')
    for c in ('wlc','wrr'):
        source=read(f'sq3/b/sq3b_vs_{c}.csv')
        sensitivity={(r['cell'],r['metric']):r for r in read(f'sq3/b/sq3b_vs_{c}_state_aware_sensitivity.csv')}
        out=[]
        for inp in json.loads((ROOT/'sq3/b/sensitivity_inputs.json').read_text()):
            if inp['comparison']!=c:continue
            row=next((r for r in source if r['arm']=='state_aware' and r['cell']==inp['cell'] and r['metric']==inp['metric']),None)
            is_ladder=row is None
            if is_ladder:
                row=next(r for r in ladder if r.get('base_arm')==inp['base_arm'] and r.get('cand_arm',r.get('arm'))=='state_aware' and r['cell']==inp['cell'])
            corr=corrected.get((inp['base_arm'],'state_aware',inp['cell'],inp['metric']))
            sens=sensitivity[(inp['cell'],inp['metric'])]
            scale=1000 if inp['metric'].startswith('e2el') else 1
            raw=row['p_permutation'];lo=row.get('ci_low_pp') if is_ladder else row['ci_low'];hi=row.get('ci_high_pp') if is_ladder else row['ci_high']
            eligible=sens['observed_pool_can_reach_alpha'].lower()=='true'
            out.append(dict(cell=inp['cell'],metric=inp['metric'],unit=inp['unit'],base_arm=inp['base_arm'],cand_arm='state_aware',run_scope=inp['run_scope'],n_base=inp['n_base'],n_cand=inp['n_cand'],base_median=inp['base_median'],cand_median=inp['cand_median'],difference=inp['observed_diff'],ci95_low=float(lo)/scale,ci95_high=float(hi)/scale,p_raw=raw,p_adj_B=corr['p_adj_B'] if corr else '',family_B=corr['family_B'] if corr else 'SQ3a completion; uncorrected',family_B_size=corr['family_B_size'] if corr else '',observed_pool_min_p=sens['observed_pool_min_p'],observed_pool_can_reach_alpha=eligible,p_display=raw if eligible else 'U',p_adj_display=(corr['p_adj_B'] if corr else '') if eligible else 'U',significant_B=bool(corr and eligible and float(corr['p_adj_B'])<=.05)))
        write(f'sq3/b/sq3b_vs_{c}_state_aware_report_table.csv',out)

    ledger=read('run_inclusion.csv');inc=[r for r in ledger if r['included']=='True']
    skip=[r for r in read('sq1/sq1_skip_accounting.csv') if r['clean']=='True' and int(r['n_runs'])]
    shifts=[abs(float(r['delta_div_router_relaxed_minus_strict'])) for r in skip if r['delta_div_router_relaxed_minus_strict']]
    distribution=[r for r in read('sq1/sq1_b_metrics_distribution.csv') if r['clean']=='True']
    # Recompute stale ages from measured records; percentiles cannot be pooled from cell medians.
    ages={n:[] for n in ('pi-node-1','pi-node-2','pi-node-3')}
    from sq2.analyze_sq2 import scan
    from analyze_sq1 import resolve_num_warmup
    for p in included_paths():
        m=json.loads(p.read_text())
        if m['verdict']!='clean':continue
        nw,_=resolve_num_warmup(m,str(p.parent))
        result=scan(m,str(p.parent/m['router_log_file']),nw,'measured')
        for n in ages:ages[n].extend(result['node_age'][n])
    write('sq2/sq2_stale_ages_by_node.csv',[dict(node=n,scope='clean',n_stale_observations=len(v),median_age_ms=statistics.median(v),max_age_ms=max(v)) for n,v in ages.items()])
    stats=dict(runs=len(inc),measured_requests=sum(int(r['measured_requests']) for r in inc),clean_runs=sum(r['verdict']=='clean' for r in inc),excluded_pilot_runs=sum(r['included']=='False' for r in ledger),clean_nonempty_arm_cells=len(skip),clean_measured_decisions=sum(int(r['total_decisions']) for r in skip),strict_filter_excluded_decisions=sum(int(r['skip_total']) for r in skip),freshness_filter_mean_absolute_divergence_shift=statistics.mean(shifts),freshness_filter_max_absolute_divergence_shift=max(shifts),clean_b_observations=sum(int(r['count']) for r in distribution),max_b_observed=max(float(r['b_metrics_value']) for r in distribution),sq3b_tests=len(bh),bh_definition_B_families=len({r['family_B'] for r in bh}),ladder_rows=len(ladder),ladder_tested=sum(r['tested']=='True' for r in ladder),ladder_raw_p_below_005=sum(r['p_permutation']!='' and float(r['p_permutation'])<.05 for r in ladder))
    (ROOT/'report_summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    lines=['# Combined results: numbers to update when writing', '',
           'Generated from the combined outputs; inference uses BH definition B.', '',
           f"- Scope: {stats['runs']} runs, {stats['measured_requests']:,} measured requests; {stats['clean_runs']} clean runs.",
           f"- SQ1: {stats['clean_nonempty_arm_cells']} nonempty clean arm–cells; {stats['clean_measured_decisions']:,} decisions, with {stats['strict_filter_excluded_decisions']:,} excluded by the strict comparison filter.",
           f"- Removing the freshness filter shifts divergence by {statistics.mean(shifts):.6f} on average in absolute terms, at most {max(shifts):.6f}.",
           f"- Count observations: {stats['clean_b_observations']:,}; maximum recorded processing count {stats['max_b_observed']:g}.",
           f"- SQ3b: {stats['sq3b_tests']} tests in {stats['bh_definition_B_families']} reported families.",
           f"- Completion ladder: {stats['ladder_tested']} computable tests out of {stats['ladder_rows']} planned rows; {stats['ladder_raw_p_below_005']} raw p-values below .05. Missing arms account for the uncomputed rows.", '',
           '## Primary comparisons', '',
           '| Baseline | Table rows | Raw p < .05 | Significant under B | Observed-pool U rows |',
           '|---|---:|---:|---:|---:|']
    for c in ('wlc','wrr'):
        table=read(f'sq3/b/sq3b_vs_{c}_state_aware_report_table.csv')
        lines.append(f"| {c} | {len(table)} | {sum(float(r['p_raw'])<.05 for r in table)} | {sum(r['significant_B']=='True' for r in table)} | {sum(r['p_display']=='U' for r in table)} |")
    lines += ['', 'WLC completion rows are corrected within block2|completion_rate_pp under B; each equals rung E of the SQ3a ladder. U is exact observed-pool inability to attain p ≤ .05, not evidence of equivalence.', '',
              '## New ShareGPT latency contrasts: state_aware minus WLC', '',
              '| Cell | Metric | Difference (s) | 95% CI (s) | p adjusted B |',
              '|---|---|---:|---|---:|']
    for r in read('sq3/b/sq3b_vs_wlc_state_aware_report_table.csv'):
        if r['cell'] in ('c4_sgpt','c12_sgpt') and r['metric'].startswith('e2el'):
            lines.append(f"| {r['cell']} | {r['metric']} | {float(r['difference']):+.3f} | [{float(r['ci95_low']):+.3f}, {float(r['ci95_high']):+.3f}] | {float(r['p_adj_B']):.5f} |")
    lines += ['', 'At c12, lower median latency coincides with higher p95 and fewer completions (592/600 versus WLC 599/600). This is a trade-off, not a general policy win. Revise the original claim of no corrected latency advantage anywhere.', '',
              '## Failure totals', '', '| Category | Count | % of bench failures | % of requests |', '|---|---:|---:|---:|']
    for r in read('sq3/a/sq3a_failure_taxonomy.csv'):
        if r['block']=='grid_total':
            share=f"{float(r['pct_of_bench_failures']):.2f}" if r['pct_of_bench_failures'] else '—'
            lines.append(f"| {r['category']} | {r['count']} | {share} | {float(r['pct_of_offered']):.3f} |")
    lines += ['', 'The bench_failures row is a subtotal, not another category. Client disconnects are excluded from bench failures.', '',
              '## Descriptive staleness associations', '', '| Predictor | Cells | Spearman rho |', '|---|---:|---:|']
    for r in read('../../figures/analysis/sq2/p3/staleness_spearman.csv'):
        if r['predictor'] in ('prompt_tok_s','inflight_mean','resident_mean'):
            lines.append(f"| {r['predictor']} | {r['n_cells']} | {float(r['spearman_rho']):.4f} |")
    lines += ['', 'These nine-cell associations are descriptive, not evidence of a causal mechanism. Different arm coverage between the original and extension cells also limits pooled cross-cell interpretation.', '',
              'See the top-level README.md for how each result is reproduced.']
    (ROOT/'RESULTS_UPDATE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
