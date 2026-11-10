#!/usr/bin/env python3
"""Build frozen, contamination-screened benchmark subsets. No inference or judges."""
from __future__ import annotations

import argparse
import base64
import collections
import csv
import hashlib
import inspect
import json
import math
import re
import shutil
import sys
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'stratified-suite-v1'


def encode_binary(value):
    if isinstance(value,bytes):return {'__bytes_base64__':base64.b64encode(value).decode()}
    raise TypeError(type(value).__name__)


def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2,default=encode_binary) + '\n')


def jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'),default=encode_binary) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def canonical(text):
    # Keep letters, numbers and order, including dates and numerical constraints.
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def shingles(text):
    text = canonical(text)
    return {text[i:i+13] for i in range(max(1, len(text)-12))}


def signatures(grams):
    return sorted(grams, key=lambda x: hashlib.blake2b(x.encode(), digest_size=8).digest())[:8]


def read(path):
    path = Path(path)
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        yield from pq.read_table(path).to_pylist()
    elif path.suffix == '.csv':
        with path.open() as f: yield from csv.DictReader(f)
    else:
        with path.open() as f:
            for line in f:
                if line.strip(): yield json.loads(line)


def decrypt(value, canary):
    encrypted = base64.b64decode(value)
    key = hashlib.sha256(canary.encode()).digest()
    return bytes(x ^ key[i % len(key)] for i, x in enumerate(encrypted)).decode()


def normalize_parent(name, spec):
    """Return whole tasks; never sample checklist/answer rows as independent tasks."""
    files = [Path(p) for p in spec['files']]
    for p in files:
        if not p.is_file(): raise FileNotFoundError(p)
    rows = []
    excluded_view = 0
    if name == 'browsecomp':
        for i, r in enumerate(read(files[0])):
            rows.append(dict(id=str(i), question=decrypt(r['problem'], r['canary']),
                             answer=decrypt(r['answer'], r['canary']), stratum=r['problem_topic'], reference=r))
    elif name == 'browsecomp_plus':
        # Stream the complete 2GB author-decrypted release; retain task + all qrels,
        # not copied document bodies. The full source file hash remains in the lock.
        for r in read(files[0]):
            ref = {k:v for k,v in r.items() if k not in ('gold_docs','negative_docs','evidence_docs')}
            for key in ('gold_docs','evidence_docs'):
                ref[key] = [str(d.get('docid', d.get('id', ''))) if isinstance(d, dict) else str(d)
                            for d in r.get(key, [])]
            rows.append(dict(id=str(r['query_id']), question=r['query'], answer=r['answer'],
                             stratum='pending_topic_join', reference=ref))
    elif name == 'gaia':
        for r in read(files[0]):
            if r.get('file_name'):
                excluded_view += 1; continue
            rows.append(dict(id=r['task_id'], question=r['Question'], answer=r['Final answer'],
                             stratum=str(r['Level']), reference=r))
    elif name == 'hle':
        for r in read(files[0]):
            if r.get('image'):
                excluded_view += 1; continue
            rows.append(dict(id=r['id'], question=r['question'], answer=r['answer'],
                             stratum=str(r.get('category', r.get('raw_subject', 'unknown'))), reference=r))
    elif name == 'drbench':
        for r in read(files[0]):
            rows.append(dict(id=str(r['id']), question=r['prompt'], answer='',
                             stratum=r['language'], reference=r))
    elif name == 'liveresearchbench':
        grouped = collections.defaultdict(list)
        for r in read(files[0]): grouped[str(r['qid'])].append(r)
        for key, group in grouped.items():
            # Official non-realtime version: frozen authored dates in question and
            # checklist, rather than replacing dates using the runtime wall clock.
            questions = {r['question_no_placeholder'] for r in group}
            if len(questions) != 1: raise ValueError('inconsistent LRB grouped question')
            rows.append(dict(id=key, question=questions.pop(), answer='',
                             stratum=group[0]['category'], reference={'qid':key,'checklists':group}))
    elif name == 'mind2web2':
        for p in files:
            for r in read(p):
                rows.append(dict(id=r['task_id'], question=r['task_description'], answer='',
                                 stratum=p.stem, reference={**r,'split':p.stem}))
    elif name == 'widesearch':
        for r in read(files[0]):
            rows.append(dict(id=r['instance_id'], question=r['query'], answer='',
                             stratum=r['language'], reference=r))
    else: raise ValueError(name)
    if len(rows) != spec['expected_population']:
        raise ValueError(f'{name}: expected {spec["expected_population"]} tasks; got {len(rows)}')
    if len({r['id'] for r in rows}) != len(rows): raise ValueError('duplicate official IDs')
    if any(not r['question'].strip() for r in rows): raise ValueError('empty task')
    return rows, excluded_view


def apportion(counts, total):
    """Hamilton allocation. Stable ties; no minimum-one distortion of a 5% sample."""
    n = sum(counts.values())
    if not n or total > n or total < 0: raise ValueError('infeasible quota')
    raw = {k: total*v/n for k,v in counts.items()}
    out = {k: math.floor(v) for k,v in raw.items()}
    for key in sorted(raw, key=lambda k: (-(raw[k]-out[k]), k))[:total-sum(out.values())]: out[key] += 1
    return out


def select(rows, count, seed, name):
    groups = collections.defaultdict(list)
    for row in rows: groups[row['stratum']].append(row)
    quota = apportion({k:len(v) for k,v in groups.items()}, count)
    selected = []
    for group, values in sorted(groups.items()):
        values = sorted(values, key=lambda r: digest(f'{seed}\0{name}\0{r["id"]}'))
        selected.extend(values[:quota[group]])
    return sorted(selected, key=lambda r: digest(f'{seed}\0{name}\0{r["id"]}')), quota


class Overlap:
    """Exact checks + deterministic shingle candidate retrieval, confirmed by Jaccard.

    Eight bottom-hash anchors retrieve candidates. Exact Jaccard >= .90 confirms
    near matches. Candidate recall is probabilistic, not exhaustive semantic or
    translation detection. No model answers/scores enter this decision.
    """
    def __init__(self, parents):
        self.rows = {}; self.exact = collections.defaultdict(set)
        self.anchors = collections.defaultdict(set); self.grams = {}; self.hits = collections.defaultdict(list)
        self.flags = collections.defaultdict(set)
        for name, rows in parents.items():
            for r in rows:
                key = name+'::'+r['id']; q = canonical(r['question'])
                self.rows[key] = q; self.exact[q].add(key)
                grams = shingles(q); self.grams[key] = grams
                if len(q) >= 80:
                    for g in signatures(grams): self.anchors[g].add(key)

    def add(self, question, provenance):
        q = canonical(question)
        keys = self.exact.get(q, ())
        for key in keys: self.record(key, provenance, 'exact', 1.0)
        if len(q) < 80: return
        grams = shingles(q); candidates = set()
        for g in grams:
            candidates.update(self.anchors.get(g, ()))
        for key in candidates:
            if key in keys: continue
            eq = self.rows[key]
            # Training user messages may wrap the task in harness instructions.
            # Compare the whole benchmark question, not the wrapper length.
            if len(eq)>=80 and eq in q:
                start=q.index(eq);end=start+len(eq)
                if (start==0 or q[start-1]==' ') and (end==len(q) or q[end]==' '):
                    self.record(key,provenance,'contained_question',1.0);continue
            if min(len(q), len(eq))/max(len(q), len(eq)) < .85: continue
            eg = self.grams[key]
            similarity = len(eg & grams)/len(eg | grams)
            if similarity >= .90: self.record(key, provenance, 'near_jaccard', similarity)

    def record(self, key, provenance, method, score):
        # Bounded evidence per task, while retaining distinct source/file examples.
        hit = dict(provenance, method=method, similarity=round(score,6))
        self.flags[key].add(provenance['source']+':'+method)
        if len(self.hits[key]) < 8 and hit not in self.hits[key]: self.hits[key].append(hit)


def capabilities(name, row):
    declared = {
        'browsecomp':['web_retrieval','multi_hop','precise_answer'],
        'browsecomp_plus':['fixed_corpus_retrieval','multi_hop','precise_answer'],
        'gaia':['web_retrieval','reasoning','precise_answer'],
        'hle':['expert_reasoning','precise_answer'],
        'drbench':['web_retrieval','report_synthesis','citations'],
        'liveresearchbench':['web_retrieval','report_synthesis','citations'],
        'mind2web2':['web_retrieval','constraint_aggregation','attribution'],
        'widesearch':['web_retrieval','wide_enumeration','structured_table'],
    }[name]
    q = row['question']
    # These tags are audit hints, not ground-truth capabilities or selection rules.
    hints = []
    if re.search(r'compare|comparison|比较|对比',q,re.I): hints.append('comparison')
    if re.search(r'calculate|percentage|sum of|计算|增长率',q,re.I): hints.append('numerical')
    if re.search(r'latest|current|today|last year|目前|最新',q,re.I): hints.append('time_sensitive')
    if len(q) >= 1500: hints.append('long_question')
    return {'benchmark_declared':declared,'lexical_hints_unverified':hints,
            'language_hint':'zh' if len(re.findall(r'[\u4e00-\u9fff]',q)) > len(q)*.15 else 'non_zh'}


def annotation_audit(name, rows):
    result={'question_length_characters':dict(collections.Counter(
        '<256' if len(r['question'])<256 else '256-1023' if len(r['question'])<1024 else '1024+' for r in rows))}
    if name=='hle':result['official_answer_types']=dict(collections.Counter(str(r['reference']['answer_type']) for r in rows))
    if name=='gaia':
        counts=collections.Counter()
        for r in rows:
            tools=str((r['reference'].get('Annotator Metadata') or {}).get('Tools','')).lower()
            # Report author annotations; tools used by an annotator do not prove
            # every alternative solution requires those tools.
            for label,pattern in [('visual_or_audio',r'image|vision|video|audio|speech'),
                                  ('code',r'python|code|programming'),('calculation',r'calculator|calculation'),
                                  ('office_or_pdf',r'pdf|spreadsheet|excel|word processor')]:
                if re.search(pattern,tools):counts[label]+=1
        result['annotator_reported_tool_hints']=dict(counts)
        result['tool_hints_caveat']='author-reported tool usage, not proof of tool necessity; visual/audio/office may exceed the text-web harness'
    if name=='widesearch':
        counts=collections.Counter()
        for r in rows:
            spec=r['reference']['evaluation'];spec=json.loads(spec) if isinstance(spec,str) else spec
            width=len(spec['required']);counts['1-3' if width<=3 else '4-6' if width<=6 else '7+']+=1
        result['official_required_table_columns']=dict(counts)
    return result


def sample_target(config, name, population):
    """Allow explicit per-dataset top-up quotas without changing old releases."""
    fixed = config.get('per_benchmark')
    if isinstance(fixed, dict):
        if name not in fixed:
            raise ValueError('missing per_benchmark quota: ' + name)
        fixed = fixed[name]
    if fixed is not None:
        if isinstance(fixed, bool) or not isinstance(fixed, int) or fixed < 0:
            raise ValueError('per_benchmark quota must be a nonnegative integer')
        return fixed
    return math.floor(population * config['fraction'] + .5)


def build(config_path, output):
    config = json.loads(Path(config_path).read_text()); output = Path(output)
    if output.exists(): raise FileExistsError(f'immutable release exists: {output}')
    parents = {}; views = {}; inputs = {}
    for name, spec in config['benchmarks'].items():
        parents[name], views[name] = normalize_parent(name, spec)
        inputs[name] = [{'path':str(Path(p).resolve()),'sha256':sha(p)} for p in spec['files']]
    topics = {canonical(r['question']):r['stratum'] for r in parents['browsecomp']}
    for r in parents['browsecomp_plus']:
        r['stratum'] = topics.get(canonical(r['question']), 'unknown')
    if any(r['stratum']=='unknown' for r in parents['browsecomp_plus']): raise ValueError('BC+ topic join incomplete')
    overlap = Overlap(parents)
    training_manifest = json.loads(Path(config['training_manifest']).read_text())
    if training_manifest['errors'] or training_manifest['unextractable_rows']:
        raise ValueError('training overlap inventory incomplete; inspect training manifest')
    train_index = Path(config['training_questions'])
    if sha(train_index) != training_manifest['questions_sha256']: raise ValueError('training index changed')
    overlap_algorithm_sha256=digest(''.join(inspect.getsource(obj) for obj in (normalize_parent,canonical,shingles,signatures,Overlap)))
    cache_key=digest(overlap_algorithm_sha256+json.dumps(inputs,sort_keys=True)+sha(train_index)+sha(config['exclude_current34']))
    cache=train_index.parent/('overlap-'+cache_key+'.json')
    if cache.exists():
        cached=json.loads(cache.read_text());overlap.hits.update(cached['hits'])
        overlap.flags.update({k:set(v) for k,v in cached['flags'].items()})
    else:
        for r in read(config['exclude_current34']): overlap.add(r['question'], {'source':'current34'})
        for i,r in enumerate(read(train_index)):
            overlap.add(r['question'], {'source':'training','file':r['path'],'row':r['row']})
            if i % 20000 == 0: print(json.dumps({'phase':'overlap','unique_train_questions':i}),flush=True)
        dump(cache,{'cache_key':cache_key,'algorithm_sha256':overlap_algorithm_sha256,'hits':overlap.hits,
                    'flags':{k:sorted(v) for k,v in overlap.flags.items()}})
    previous_suites = {}
    for previous in config.get('exclude_suites', []):
        previous = Path(previous).resolve();verify(previous)
        previous_suites[str(previous)] = sha(previous/'manifest.json')
        for row in read(previous/'private/references.jsonl'):
            overlap.add(row['question'], {'source':'previous_eval','suite':str(previous),'id':row['id']})
    final_output=output
    output=Path(tempfile.mkdtemp(prefix='.'+output.name+'.building-',dir=output.parent))
    (output/'private').mkdir(mode=0o700)
    audit = {}; summary = []; exclusions = []; membership = []; refs = []; prompts = {}
    for name, rows in parents.items():
        if name not in config.get('include_benchmarks', config['benchmarks']):continue
        eligible = [r for r in rows if name+'::'+r['id'] not in overlap.hits]
        target = sample_target(config, name, len(rows))
        # Insufficient uncontaminated population is a real result, never refill
        # with excluded tasks. Report any shortfall explicitly.
        picked, quota = select(eligible, min(target,len(eligible)), config['seed'], name) if eligible else ([],{})
        for r in rows:
            key = name+'::'+r['id']
            if key in overlap.hits: exclusions.append({'benchmark':name,'id':r['id'],'match_flags':sorted(overlap.flags[key]),'evidence':overlap.hits[key]})
        for i, r in enumerate(picked, 1):
            opaque = 't'+digest(name+'\0'+r['id'])[:16]
            caps = capabilities(name,r)
            membership.append({'benchmark':name,'id':r['id'],'opaque_id':opaque,'stratum':r['stratum'],
                               'question_sha256':digest(canonical(r['question'])),'capabilities':caps})
            refs.append({'benchmark':name,'id':r['id'],'opaque_id':opaque,'question':r['question'],
                         'answer':r['answer'],'reference':r['reference']})
        # A dataset name cannot select a benchmark-specific prompt accidentally.
        jsonl(output/'inference'/f'{name}.jsonl', [
            {'id':'t'+digest(name+'\0'+r['id'])[:16], 'question':r['question'],'answer':'','type':'research'} for r in picked])
        counts = lambda records: dict(sorted(collections.Counter(r['stratum'] for r in records).items()))
        parent_counts = counts(rows); eligible_counts = counts(eligible); sample_counts = counts(picked)
        cap_counts = lambda records: dict(sorted(collections.Counter(c for r in records
                       for c in capabilities(name,r)['benchmark_declared']).items()))
        secondary = lambda records: dict(sorted(collections.Counter(
            str(r['reference'].get('topic',r['reference'].get('domain',r['reference'].get('raw_subject',r['stratum']))))
            for r in records).items()))
        hints = lambda records: dict(sorted(collections.Counter(c for r in records
                     for c in capabilities(name,r)['lexical_hints_unverified']).items()))
        audit[name] = {'parent_strata':parent_counts,'eligible_strata':eligible_counts,'selected_strata':sample_counts,
            'quota':quota,'uncovered_parent_strata':[s for s in parent_counts if sample_counts.get(s,0)==0],
            'capabilities_parent':cap_counts(rows),'capabilities_selected':cap_counts(picked),
            'secondary_domains_parent':secondary(rows),'secondary_domains_selected':secondary(picked),
            'lexical_hints_parent':hints(rows),'lexical_hints_selected':hints(picked),
            'annotation_audit_parent':annotation_audit(name,rows),'annotation_audit_selected':annotation_audit(name,picked),
            'uncovered_secondary_domains':[k for k in secondary(rows) if k not in secondary(picked)],
            'audit_method':'official metadata for strata; benchmark-level capabilities; unverified lexical hints separately',
            'small_sample_warning':target<30}
        summary.append({'benchmark':name,'full_view':len(rows),'outside_text_view':views[name],
                        'excluded':len(rows)-len(eligible),'eligible':len(eligible),'target':target,'selected':len(picked),
                        'current34_overlap':sum(any(f.startswith('current34:') for f in overlap.flags[name+'::'+r['id']]) for r in rows),
                        'training_pool_exact':sum('training:exact' in overlap.flags[name+'::'+r['id']] for r in rows),
                        'training_pool_near_only':sum('training:near_jaccard' in overlap.flags[name+'::'+r['id']] and 'training:exact' not in overlap.flags[name+'::'+r['id']] for r in rows),
                        'training_pool_contained':sum('training:contained_question' in overlap.flags[name+'::'+r['id']] for r in rows),
                        'previous_eval_overlap':sum(any(f.startswith('previous_eval:') for f in overlap.flags[name+'::'+r['id']]) for r in rows),
                        'shortfall':target-len(picked),'actual_fraction':len(picked)/len(rows),'protocol':config['benchmarks'][name]['protocol']})
        prompts[name] = config['benchmarks'][name]['answer_prompt']
    jsonl(output/'private/references.jsonl',refs)
    jsonl(output/'private/membership.jsonl', membership)
    jsonl(output/'private/exclusions.jsonl',exclusions)
    with (output/'private/coverage_review.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['benchmark','official_id','opaque_id','stratum','declared_capabilities','unverified_lexical_hints','human_review'])
        writer.writeheader()
        for row in membership:writer.writerow({'benchmark':row['benchmark'],'official_id':row['id'],'opaque_id':row['opaque_id'],
            'stratum':row['stratum'],'declared_capabilities':','.join(row['capabilities']['benchmark_declared']),
            'unverified_lexical_hints':','.join(row['capabilities']['lexical_hints_unverified']),'human_review':''})
    dump(output/'coverage.json',audit); dump(output/'summary.json',summary)
    with (output/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    question_groups = collections.defaultdict(list)
    for r in membership: question_groups[r['question_sha256']].append(r['opaque_id'])
    dump(output/'private/cross_benchmark_duplicates.json', [v for v in question_groups.values() if len(v)>1])
    dump(output/'coverage_summary.json',{'tasks':len(membership),'unique_question_texts':len(question_groups),
         'duplicate_question_groups':sum(len(v)>1 for v in question_groups.values()),
         'benchmark_declared_capabilities':dict(collections.Counter(c for r in membership for c in r['capabilities']['benchmark_declared'])),
         'limits':'benchmark-level labels and official strata are audited programmatically; lexical/tool-use hints are not human-verified task-level capability labels'})
    promptdir = output/'prompts'; promptdir.mkdir()
    for name,text in prompts.items(): (promptdir/(name+'.txt')).write_text(text+'\n')
    dump(promptdir/'registry.json',{'default':'default.txt','datasets':{f'suite_{k}':k+'.txt' for k in prompts}})
    (promptdir/'default.txt').write_text('Answer the question and follow its requested format.\n')
    dump(output/'config.json',config)
    shutil.copyfile(config['training_manifest'], output/'private/training_manifest.json')
    assets=final_output.parent/'assets.lock.json'
    if not assets.exists():raise ValueError('acquisition not complete: assets.lock.json missing')
    shutil.copyfile(assets,output/'assets.lock.json')
    shutil.copytree(Path(__file__).parent,output/'code',ignore=shutil.ignore_patterns('__pycache__'))
    files = {str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    manifest = {'version':VERSION,'seed':config['seed'],'fraction':config['fraction'],
        'per_benchmark':config.get('per_benchmark'),'excluded_suites':previous_suites,
        'quota_basis':('fixed count from full eligible population; Hamilton apportionment over eligible strata' if config.get('per_benchmark') else 'round-half-up(full text-view population * fraction); Hamilton apportionment over eligible strata'),
        'selection':'SHA256(seed,NUL,benchmark,NUL,official_id), no model outcomes',
        'source_files':inputs,'source_versions':{k:s['version'] for k,s in config['benchmarks'].items()},
        'current34_sha256':sha(config['exclude_current34']), 'training_index_sha256':sha(train_index),
        'overlap_algorithm_sha256':overlap_algorithm_sha256,
        'overlap_method':'NFKC/casefold word exact at all lengths; complete question containment (>=80 chars); near checks >=80 chars, 8 bottom-hash char13-gram anchors + verified Jaccard>=0.90, length ratio>=0.85',
        'overlap_limits':'does not certify absence of paraphrase, translation or pretraining contamination',
        'scoring':'retain separate native metrics; no aggregate average across incomparable metric scales',
        'cross_benchmark_duplicate_groups':sum(len(v)>1 for v in question_groups.values()),
        'source_code_sha256':sha(__file__),'files':files}
    import platform
    import subprocess
    manifest['python_version']=platform.python_version()
    manifest['glasseek_git_head']=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    dump(output/'manifest.json',manifest)
    output.rename(final_output)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def verify(output):
    output=Path(output); m=json.loads((output/'manifest.json').read_text())
    for rel,h in m['files'].items():
        if sha(output/rel)!=h: raise ValueError('release changed: '+rel)
    asset_lock=json.loads((output/'assets.lock.json').read_text())
    for rel,h in asset_lock['files'].items():
        if sha(output.parent/rel)!=h:raise ValueError('frozen source/evaluator changed: '+rel)
    print(json.dumps({'verified_files':len(m['files']),'manifest_sha256':sha(output/'manifest.json')}))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    b=sub.add_parser('build'); b.add_argument('--config',required=True,type=Path);b.add_argument('--output',required=True,type=Path)
    v=sub.add_parser('verify');v.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.action=='build': build(a.config,a.output)
    else: verify(a.output)


if __name__=='__main__': main()
