"""Build a standalone, auditable HTML/CSV report from recorded results."""
import csv,html,json
from pathlib import Path
OUT=Path('/root/autodl-tmp/Glasseek/reports/trajectory-stratified-qwen38-20261107')
def esc(x):return html.escape(str(x))
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def main():
    summary=json.loads((OUT/'run_summary.json').read_text());backend=json.loads((OUT/'backend_selection.json').read_text());spot=json.loads((OUT/'spotcheck.json').read_text())
    cols=['source','selected','normalized','eligible','completed','scored','candidates']
    with (OUT/'dataset_rates.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=cols+['inventory','audit','review','behaviors']);w.writeheader()
        for r in summary['sources']:w.writerow({**{k:r[k] for k in cols},**r['stages']})
    artifacts={a['sample']['id']:a['result'] for a in (json.loads(p.read_text()) for p in (OUT/'results').glob('*.json'))}
    with (OUT/'review_triage.csv').open('w') as file:
        fields=['id','source','normalization_status','eligible','pre_llm_hold','llm_status','candidate','quality_issues','evidence_gaps','annotation_issues','error','raw_path','sample_path']
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for row in (json.loads(line) for line in (OUT/'manifest.jsonl').open()):
            result=artifacts.get(row['id'],{})
            writer.writerow({'id':row['id'],'source':row['source'],'normalization_status':row['normalization_status'],'eligible':row['eligible'],'pre_llm_hold':row['hold_category'],'llm_status':result.get('status','not_attempted' if not row['eligible'] else 'pending'),'candidate':result.get('offline_quality_candidate',False),'quality_issues':len(result.get('quality_issues',[])),'evidence_gaps':len(result.get('evidence_gaps',[])),'annotation_issues':len(result.get('annotation_issues',[])),'error':result.get('error',row.get('normalization_error','')),'raw_path':row['raw_path'],'sample_path':row.get('sample_path','')})
    parts=['<!doctype html><meta charset="utf-8"><title>轨迹预处理小样本评测</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#172033}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #ddd;padding:7px;text-align:left;vertical-align:top}th{background:#f1f5f9}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f7f7;padding:16px}small{color:#555}</style>',
      '<h1>轨迹预处理：每数据集 8 条</h1><p>9 个公开数据集，共 72 个抽样单位；SearchSwarm 按 bundle 计数，未展开子代理。MiroVerse 本机不可用，未替换抽样。按长度分层覆盖抽样，以下未经总体加权的通过率不能代表完整数据集质量。</p>',
      '<p>JSON 工具调用统一转换为 Hermes/Qwen XML 文本；原始文件与证据保留。评分规则文件校验和未改变。结构、适配、预算、标注错误与事实质量分开记录。人工复核与实际 harness 重放尚未完成，所有结果 training_ready=false。</p>',
      '<h2>后端对比</h2><p>上下文 131072，CUDA Graph FULL_AND_PIECEWISE，medium，并发上限 16，FP8 KV，batch token budget 4096。相同 16 请求混合长度负载、两轮、不同轮次前缀隔离缓存。微基准输出上限 4096；正式标注按阶段使用 16384/32768。初次编译启动耗时不计入推理基准。</p>',
      table(['后端','正确/请求','两轮耗时（秒）','平均轮次耗时','输出 tokens/s'],[[k,f"{v['correct']}/{v['requests']}",', '.join(f'{n:.2f}' for n in v['round_wall_seconds']),f"{v['mean_round_seconds']:.2f}",f"{v['completion_tokens_per_wall_second']:.2f}"] for k,v in backend['results'].items()]),
      '<p>选用 TRITON_ATTN：两个后端均通过，平均总耗时相差约 1.1%，本次 TRITON 两轮波动更小。该微基准不证明任何后端对所有真实长输出负载都更优。</p>',
      '<h2>总体与成本</h2><pre>'+esc(json.dumps(summary['totals'],ensure_ascii=False,indent=2))+'</pre>',
      '<h2>各数据集漏斗</h2>'+table(['数据集','抽样','格式转换','可进入 LLM','已完成','完成评分','离线候选'],[[r[k] for k in cols] for r in summary['sources']]),
      '<h2>模型阶段</h2>'+table(['阶段','原始样本尝试','校验通过','含重试/变体调用','失败调用','输出截断','输入 tokens','输出 tokens','思考 tokens','单调用中位秒'],[[r['stage'],r['original_samples_attempted'],r['original_samples_validated'],r['calls_including_variants'],r['call_failures_including_retries'],r['length_finishes'],r['prompt_tokens'],r['completion_tokens'],r['reasoning_tokens'],round(r['request_seconds_median'],2)] for r in summary['stages']]),
      '<p>模型输出通过格式与程序校验，不等于轨迹质量通过。调用失败包含重试，分母不同于样本失败。外部 API 账单为 0 美元。本地 GPU 按用户提供的 2.79 元/小时计价，只对实际墙钟区间计费一次，不累加并发请求耗时。编译、后端基准和额外待机另计。</p>',
      '<h2>暂缓原因</h2><pre>'+esc(json.dumps([{k:r[k] for k in ['source','pre_llm_holds','scored_blockers_nonexclusive','annotation_failures']} for r in summary['sources']],ensure_ascii=False,indent=2))+'</pre>',
      '<h2>误暂缓与漏放抽查</h2><p>以下为助手逐条核对已保存证据的抽查，不是独立人工金标，不能推算总体误杀/漏放率。</p>']
    for c in spot['cases']:parts.extend(['<h3>'+esc(c['source']+' / '+c['id'])+'</h3><p>'+esc(c['classification'])+'；pipeline candidate='+esc(c['pipeline_candidate'])+'</p><p>'+esc(c['finding'])+'</p><pre>'+esc(json.dumps(c['evidence'],ensure_ascii=False,indent=2))+'</pre>'])
    pricing=OUT/'original_pricing_cny.json'
    if pricing.exists():parts.extend(['<h2>原始评测的实际卡费</h2><p>输入和输出的等效单价均把同一笔总费用全部分摊到相应 token 数上，不能相加使用。缓存命中为 0，无法测得独立缓存输入单价。</p><pre>'+esc(pricing.read_text())+'</pre>'])
    variant_review=OUT/'variant_spotcheck.json'
    if variant_review.exists():parts.extend(['<h2>增强样本抽查</h2><pre>'+esc(variant_review.read_text())+'</pre>'])
    variant=OUT/'variant_probe_summary.json'
    if variant.exists():parts.extend(['<h2>改写与反事实探针</h2><p>每个可标注来源预先固定一个父样本，未按质量结果挑选。改写重新走完整审核；反事实独立核查。生成成功不代表通过，负例不会导出为正向 SFT，父样本也不会自动成为 DPO 正例。</p><pre>'+esc(variant.read_text())+'</pre>'])
    for filename,title in [('variant_analysis.json','变体核验统计：区分生成成功与已证实负例'),('training_readiness.json','接入现有微调的准备状态')]:
        artifact=OUT/filename
        if artifact.exists():parts.extend(['<h2>'+title+'</h2><pre>'+esc(artifact.read_text())+'</pre>'])
    parts.append('<h2>审计文件</h2><p>同目录保存 manifest.jsonl、calls.jsonl、results/、spotcheck.json、backend_selection.json、xml_roundtrip_check.json、quality_rule_integrity.json、human_review_queue.jsonl 与 frozen_source/、xml_source/。</p>')
    state='原始评测与改写/反事实探针均已完成。' if (OUT/'variant_probe_summary.json').exists() else '原始评测已完成；改写/反事实探针仍在运行，相关汇总尚未完成。' if (OUT/'annotation_summary.json').exists() else '评测运行中，以下为当前进度。'
    parts.insert(2,'<p><strong>'+state+'</strong></p>')
    (OUT/'report.html').write_text('\n'.join(parts))
    print(OUT/'report.html')
if __name__=='__main__':main()
