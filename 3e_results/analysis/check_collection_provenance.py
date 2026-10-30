"""Record compatibility checks, without inferring binary or prompt identity."""
import hashlib
import json
from scope import ROOT, included_paths

def main():
    config={};profiles={};commits={}
    arms={'weighted_least_connections','weighted_round_robin','state_aware','state_aware_v2','wlc_conc','cost_predict_v3'}
    for p in included_paths():
        m=json.loads(p.read_text());arm=m.get('arm') or m['policy']
        if m['workload']!='sharegpt' or arm not in arms:continue
        b=json.loads((p.parent/m['bench_result_file']).read_text())
        phase='original_c8' if m['concurrency']==8 else 'extension_c4_c12';key=(arm,phase)
        config.setdefault(key,[]).append(m['frozen_config']);commits.setdefault(phase,set()).add(m['router_git_commit'])
        h=hashlib.sha256(json.dumps([(v.get('prompt_len'),v.get('requested_output_tokens')) for v in b['raw']]).encode()).hexdigest()
        profiles.setdefault(phase,{});profiles[phase][h]=profiles[phase].get(h,0)+1
    findings=[]
    for arm in sorted(arms):
        a=config[(arm,'original_c8')];b=config[(arm,'extension_c4_c12')]
        common=set.intersection(*(set(x) for x in a+b));different=[]
        for k in sorted(common):
            if {json.dumps(x[k],sort_keys=True) for x in a}!={json.dumps(x[k],sort_keys=True) for x in b}:different.append(k)
        findings.append(dict(arm=arm,original_c8_runs=len(a),extension_runs=len(b),shared_fields=sorted(common),different_shared_fields=different))
    result=dict(recorded_config_comparison=findings,ordered_request_length_fingerprints=profiles,
        recorded_commits={k:sorted(v) for k,v in commits.items()},
        limitations=['Length fingerprints do not establish identical prompt text.',
                     'Recorded git commits do not uniquely establish executed binaries or uncommitted source state.',
                     'This check does not establish phase equivalence experimentally.'])
    (ROOT/'collection_provenance.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Wrote collection_provenance.json; recorded differences:',[(x['arm'],x['different_shared_fields']) for x in findings])

if __name__=='__main__':main()
