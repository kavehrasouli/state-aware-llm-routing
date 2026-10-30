"""Frozen, shared run selection for the combined analysis. Raw data stay read-only."""
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
REPO = DATA.parent
LEDGER = ROOT / 'run_inclusion.csv'
GROUPS = {'sharegpt_c4_c12_v1', 'sharegpt_ext_c4_c12_v1'}
CELLS = [(w, c) for w in ('uniform', 'bimodal', 'sharegpt') for c in (4, 8, 12)]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

@lru_cache(None)
def included_paths():
    if not LEDGER.exists():
        raise RuntimeError('Run scope.py --freeze first; no implicit directory selection.')
    with LEDGER.open() as f:
        return tuple(DATA / r['metadata_path'] for r in csv.DictReader(f) if r['included'] == 'True')

def freeze():
    """Explicitly rebuild the ledger; ordinary analysis never adds newly found runs."""
    rows = []
    for p in sorted(DATA.glob('*/*/rep*/run_meta.json')):
        m = json.loads(p.read_text())
        w, c = m['workload'], m['concurrency']
        original = (w in ('uniform','bimodal') and c in (4,8,12)) or (w=='sharegpt' and c==8)
        extension = w=='sharegpt' and c in (4,12) and m.get('extension_group') in GROUPS
        included = original or extension
        bpath = p.parent / m['bench_result_file']
        lpath = p.parent / m['router_log_file']
        b = json.loads(bpath.read_text())
        rows.append(dict(metadata_path=str(p.relative_to(DATA)), included=included,
            reason='original_grid' if original else 'sharegpt_extension' if extension else 'off_grid',
            arm=m.get('arm') or m['policy'], workload=w, concurrency=c, rep=m['rep'],
            extension_group=m.get('extension_group') or '', verdict=m['verdict'],
            measured_requests=b['num_prompts'], completed=b['completed'], failed=b['failed'],
            start_ts=m.get('start_ts',''), router_git_commit=m.get('router_git_commit',''),
            metadata_sha256=digest(p), bench_file=bpath.name, bench_sha256=digest(bpath),
            router_file=lpath.name, router_sha256=digest(lpath)))
    assert sum(r['included'] for r in rows)==508, 'Review scope before changing frozen expected count'
    assert sum(r['included'] and r['reason']=='original_grid' for r in rows)==436
    with LEDGER.open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
    print('Frozen 508 included runs; exclusions remain visible in run_inclusion.csv.')

def verify():
    with LEDGER.open() as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        p=DATA/r['metadata_path']
        for path,key in [(p,'metadata_sha256'),(p.parent/r['bench_file'],'bench_sha256'),
                         (p.parent/r['router_file'],'router_sha256')]:
            if digest(path)!=r[key]: raise RuntimeError(f'Frozen input changed: {path}')
    print(f'Input hashes verified for all {len(rows)} ledger entries.')

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');a=ap.parse_args()
    freeze() if a.freeze else verify()
