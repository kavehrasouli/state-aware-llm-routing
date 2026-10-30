/** Power and attainable-p analysis; run prepare_sensitivity.py first.
 * 50,000 simulated normal datasets per comparison with exact median-statistic
 * permutation tests at unadjusted alpha = 0.05; observed ties are enumerated separately.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const DIR = path.dirname(fileURLToPath(import.meta.url));
const N = 50000, ALPHA = 0.05, SEED_BASE = 20260906, STEP = 1 / 256;
const METRICS = new Set(['e2el_p50', 'e2el_p95', 'tpot_p50']);
const STEM = 'state_aware_e2el_p50_p95_tpot_p50';

function readCSV(filename) {
  const text = fs.readFileSync(path.join(DIR, filename), 'utf8');
  const rows = []; let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') { field += '"'; i++; }
      else quoted = !quoted;
    } else if (c === ',' && !quoted) { row.push(field); field = ''; }
    else if (c === '\n' && !quoted) {
      row.push(field.replace(/\r$/, '')); rows.push(row); row = []; field = '';
    } else field += c;
  }
  if (field || row.length) { row.push(field.replace(/\r$/, '')); rows.push(row); }
  assert(!quoted, `Unterminated quoted CSV field in ${filename}`);
  const headers = rows.shift();
  return rows.filter(r => r.some(Boolean)).map(r => {
    assert.equal(r.length, headers.length, `CSV width in ${filename}`);
    return Object.fromEntries(headers.map((h, i) => [h, r[i]]));
  });
}

function csvText(rows) {
  const columns = Object.keys(rows[0]);
  const escape = v => /[",\n\r]/.test(String(v ?? ''))
    ? '"' + String(v).replaceAll('"', '""') + '"' : String(v ?? '');
  return [columns, ...rows.map(r => columns.map(c => r[c]))]
    .map(r => r.map(escape).join(',')).join('\n') + '\n';
}

function close(actual, expected, label) {
  assert(Number.isFinite(actual) && Number.isFinite(expected), label);
  assert(Math.abs(actual - expected) <= 1e-10 * Math.max(1, Math.abs(expected)),
    `${label}: ${actual} != ${expected}`);
}

const signatureCache = new Map();
function rankSignatures(na, nb) {
  const cacheKey = `${na}/${nb}`;
  if (signatureCache.has(cacheKey)) return signatureCache.get(cacheKey);
  const n = na + nb, k = Math.min(na, nb), chooseA = na <= nb, map = new Map();
  assert(n < 31, 'Bit-mask enumeration requires fewer than 31 total runs');
  let total = 0;
  function rec(start, left, mask) {
    if (!left) {
      const a = [], b = [];
      for (let i = 0; i < n; i++) {
        ((chooseA ? Boolean(mask & (1 << i)) : !Boolean(mask & (1 << i))) ? a : b).push(i);
      }
      const key = [a[Math.floor((na - 1) / 2)], a[Math.floor(na / 2)],
        b[Math.floor((nb - 1) / 2)], b[Math.floor(nb / 2)]].join(',');
      map.set(key, (map.get(key) || 0) + 1); total++; return;
    }
    for (let i = start; i <= n - left; i++) rec(i + 1, left - 1, mask | (1 << i));
  }
  rec(0, k, 0);
  const result = { total, sigs: [...map].map(([k, w]) => [...k.split(',').map(Number), w]) };
  signatureCache.set(cacheKey, result); return result;
}

function rng(seed) {
  return () => {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function normal(r) { return Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()); }
function median(a) { const m = a.length >> 1; return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2; }

function study(na, nb) {
  const { total, sigs } = rankSignatures(na, nb), r = rng(SEED_BASE + na);
  const cutoff = Math.floor(ALPHA * total + 1e-9), samples = [], cache = new Map();
  for (let i = 0; i < N; i++) {
    const a = Array.from({ length: na }, () => normal(r)).sort((x, y) => x - y);
    const b = Array.from({ length: nb }, () => normal(r)).sort((x, y) => x - y);
    samples.push([a, b, median(a), median(b)]);
  }
  const nullRejections = new Uint8Array(N);
  function power(d) {
    if (cache.has(d)) return cache.get(d);
    let reject = 0; const pool = new Array(na + nb);
    for (let s = 0; s < N; s++) {
      const [a, b, ma, mb] = samples[s]; let ia = 0, ib = 0;
      for (let i = 0; i < pool.length; i++) {
        pool[i] = ib >= nb || (ia < na && a[ia] <= b[ib] + d) ? a[ia++] : b[ib++] + d;
      }
      const obs = Math.abs(mb + d - ma); let hits = 0;
      for (const [a0, a1, b0, b1, w] of sigs) {
        if (Math.abs((pool[b0] + pool[b1]) / 2 - (pool[a0] + pool[a1]) / 2) >= obs - 1e-12) hits += w;
        if (hits > cutoff) break;
      }
      if (hits <= cutoff) { reject++; if (d === 0) nullRejections[s] = 1; }
    }
    cache.set(d, reject / N); return reject / N;
  }
  const nullPower = power(0);
  const limitingSigs = sigs.map(([a0, a1, b0, b1, w]) => [a0, a1, b0, b1, w,
    ((b0 >= na) + (b1 >= na) - (a0 >= na) - (a1 >= na)) / 2])
    .filter(s => Math.abs(s[5]) === 1);
  let limitingRejections = 0, unionRejections = 0;
  for (let s = 0; s < N; s++) {
    const [a, b, ma, mb] = samples[s], pool = a.concat(b); let hits = 0;
    for (const [a0, a1, b0, b1, w, slope] of limitingSigs) {
      if (slope * ((pool[b0] + pool[b1]) / 2 - (pool[a0] + pool[a1]) / 2) >= mb - ma - 1e-12) hits += w;
    }
    const rejected = hits <= cutoff;
    limitingRejections += Number(rejected);
    unionRejections += Number(rejected || nullRejections[s]);
  }
  const limitingPower = limitingRejections / N, powerUpperBound = unionRejections / N;
  close(power(32), limitingPower, 'Large-shift and limiting power');

  function threshold(target) {
    if (powerUpperBound < target) return {
      lower: null, upper: null, achieved: null, status: 'unattainable_in_simulated_datasets'
    };
    // Integer bisection preserves the earlier [0,4], ten-step D80 resolution.
    let lo = 0, hi = 1024;
    while (power(hi * STEP) < target && hi < 8192) hi *= 2;
    assert(power(hi * STEP) >= target, `Target ${target} not bracketed for ${na}/${nb}`);
    while (hi - lo > 1) {
      const mid = Math.floor((lo + hi) / 2);
      if (power(mid * STEP) >= target) hi = mid; else lo = mid;
    }
    assert(power(lo * STEP) < target && power(hi * STEP) >= target);
    return { lower: lo * STEP, upper: hi * STEP, achieved: power(hi * STEP), status: 'reached' };
  }
  const thresholds = [0.8, 0.9, 0.95].map(threshold);
  console.log(JSON.stringify({ n_base: na, n_cand: nb, total, nullPower,
    limitingPower, powerUpperBound, thresholds }));
  return { na, nb, total, nullPower, limitingPower, powerUpperBound, thresholds, power };
}


const inputs = JSON.parse(fs.readFileSync(path.join(DIR, 'sensitivity_inputs.json')));
const studies = new Map(), full = {wlc:[],wrr:[]}, multi={wlc:[],wrr:[]};
function observedFloor(row) {
  const {total,sigs}=rankSignatures(row.n_base,row.n_cand);
  const p=row.base_values.concat(row.cand_values).sort((a,b)=>a-b);
  const vals=sigs.map(([a0,a1,b0,b1,w])=>[Math.abs((p[b0]+p[b1]-p[a0]-p[a1])/2),w]);
  const max=Math.max(...vals.map(x=>x[0]));
  const hits=vals.filter(x=>x[0]>=max-1e-12).reduce((s,x)=>s+x[1],0);
  return {min:hits/total,hits,total,identical:p[0]===p.at(-1)};
}
for (const row of inputs) {
  const floor=observedFloor(row), sd=row.pooled_sd, key=`${row.n_base}/${row.n_cand}`;
  let studyResult=null;
  if (sd>0) {
    if (!studies.has(key)) studies.set(key,study(row.n_base,row.n_cand));
    studyResult=studies.get(key);
  }
  const t=studyResult?.thresholds[0];
  const d=sd>0 && t?.upper!==null ? t.upper*sd : null;
  const caveat=row.metric==='completion_rate_pp' ? 'Discrete bounded completion; normal location model is a poor fit. Observed-pool attainability is reported separately.' : row.metric==='throughput_req_s' ? 'Nonnegative capacity-constrained throughput; unbounded normal location shifts may be unrealistic.' : row.metric==='tpot_p95' ? 'Wide intervals and potentially skewed run-level tail percentiles; normal common-variance model is a poor fit.' : 'Run-level percentile approximation; nonnegative support, skew, unequal variances and uncertainty in pooled SD are not modelled.';
  const {base_values,cand_values,comparison,...observed}=row;
  full[comparison].push({...observed,detectable_diff_80:d,
    detectable_diff_to_base_ratio:d!==null && row.base_median!==0 ? d/row.base_median:null,
    detectable_diff_to_base_pct:d!==null && row.base_median!==0 ? 100*d/row.base_median:null,
    observed_pool_min_p:floor.min, observed_pool_min_p_hits:floor.hits,
    permutation_partitions:floor.total,observed_pool_can_reach_alpha:floor.min<=ALPHA,
    all_observed_values_identical:floor.identical,
    power_estimate_status:sd===0?'zero_observed_sd_no_location_power_estimate':t.status,
    alpha:ALPHA,target_power:.8,simulated_datasets:sd>0?N:0,simulation_seed:SEED_BASE+row.n_base,
    standardized_d80_lower:t?.lower??null,standardized_d80_upper:t?.upper??null,
    simulated_power_at_upper:t?.achieved??null,simulated_null_rejection_rate:studyResult?.nullPower??null,
    simulation_model:'independent normal run-level values, common variance fixed at pooled observed SD',
    test_statistic:'absolute difference of group medians; candidate minus baseline',
    permutation_method:'exhaustive labelled partitions aggregated by median rank signatures; unadjusted alpha',model_caveat:caveat});
  if (METRICS.has(row.metric)) {
    assert(studyResult, 'Positive SD required for selected latency metrics');
    const s=studyResult;
    const result={...observed,detectable_diff_80:d,
      detectable_diff_90:s.thresholds[1].upper===null?null:s.thresholds[1].upper*sd,
      detectable_diff_95:s.thresholds[2].upper===null?null:s.thresholds[2].upper*sd,
      alpha:ALPHA,simulated_datasets:N,simulation_seed:SEED_BASE+s.na,
      permutation_partitions:s.total,simulated_null_rejection_rate:s.nullPower,
      simulated_limiting_power:s.limitingPower,simulated_power_upper_bound:s.powerUpperBound,
      standardized_search_resolution:STEP};
    for(let i=0;i<3;i++) {
      const label=[80,90,95][i],t=s.thresholds[i];
      result[`d${label}_status`]=t.status;result[`standardized_d${label}_lower`]=t.lower;
      result[`standardized_d${label}_upper`]=t.upper;result[`simulated_power_at_d${label}`]=t.achieved;
    }
    result.simulation_model='independent normal run-level statistics; common variance fixed at pooled observed sample SD';
    result.test_statistic='absolute difference of group medians; observed_diff = cand - base';
    result.permutation_method='exhaustive labelled partitions aggregated by median rank indices; two-sided unadjusted alpha';
    multi[comparison].push(result);
  }
}
for(const c of ['wlc','wrr']) {
  assert.equal(full[c].length,54);assert.equal(multi[c].length,27);
  fs.writeFileSync(path.join(DIR,`sq3b_vs_${c}_state_aware_sensitivity.csv`),csvText(full[c]));
  fs.writeFileSync(path.join(DIR,`sq3b_vs_${c}_${STEM}_multipower.csv`),csvText(multi[c]));
}
const curves=[];
for(const cell of ['c8_bim','c12_bim']) {
  const row=inputs.find(r=>r.comparison==='wlc' && r.cell===cell && r.metric==='e2el_p50');
  const s=studies.get(`${row.n_base}/${row.n_cand}`);
  for(let i=0;i<=80;i++) curves.push({cell,imposed_difference_s:i/10*row.pooled_sd,power:s.power(i/10)});
  assert(curves.at(-1).power>=.99);
}
fs.writeFileSync(path.join(DIR,'sq3b_vs_wlc_state_aware_e2el_p50_c8_c12_bimodal_power_curves.csv'),csvText(curves));
console.log('Wrote two 54-row sensitivity tables, two 27-row multi-power tables and 162 curve points.');
