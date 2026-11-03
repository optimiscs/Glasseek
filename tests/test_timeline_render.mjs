import assert from 'node:assert/strict';
import { createTimeline } from '../Evaluation/viewer/timeline.js';

const dom = Object.fromEntries(['chartTitle', 'chartSub', 'chartLegend', 'timelineVisual'].map((key) => [key, { textContent: '', innerHTML: '' }]));
const timeline = createTimeline({
  dom, state: {}, activeCase: () => null, $$: () => [],
  formatAxis: String, escapeAttr: String, formatDuration: String, number: String,
  formatSeconds: String, escapeHTML: String, pad: String,
});
const c = { durationSeconds: 10, timingAvailable: false, stats: { computeCalls: 0, searchCalls: 2, readCalls: 0 }, steps: [
  { n: 1, kind: 'search', label: '搜索', tool: {}, modelStart: 0, model_seconds: 1, toolStart: 1, tool_seconds: 2 },
  { n: 2, kind: 'search', label: '搜索', tool: {}, modelStart: 3, model_seconds: 1, toolStart: null, tool_seconds: null },
] };
timeline.renderDurationChart(c);
assert.match(dom.chartSub.textContent, /部分耗时已记录/);
assert.equal((dom.timelineVisual.innerHTML.match(/class="chart-bar /g) || []).length, 3);
assert.match(dom.timelineVisual.innerHTML, /1 项耗时未记录/);
timeline.renderCallsChart(c);
assert.equal((dom.timelineVisual.innerHTML.match(/class="chart-bar /g) || []).length, 1);
assert.match(dom.chartSub.textContent, /1 次耗时未记录/);
console.log('partial timeline renders recorded spans and excludes missing spans');
