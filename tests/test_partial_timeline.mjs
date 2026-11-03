import assert from 'node:assert/strict';
import { createTimeline } from '../Evaluation/viewer/timeline.js';

const dom = { chartTitle: {}, chartSub: {}, chartLegend: {}, timelineVisual: {} };
const trace = {
  timingAvailable: false, durationSeconds: 20, stats: { searchCalls: 2, readCalls: 0, computeCalls: 0 },
  steps: [
    { n: 1, kind: 'search', label: '搜索', modelStart: 1, model_seconds: 3, toolStart: 4, tool_seconds: 2, tool: {} },
    { n: 2, kind: 'search', label: '搜索', modelStart: 9, model_seconds: 1, toolStart: null, tool_seconds: null, tool: {} },
  ],
};
const context = { state: {}, dom, activeCase: () => trace, $$: () => [], formatAxis: String,
  escapeAttr: String, escapeHTML: String, formatDuration: String, number: String, formatSeconds: String, pad: String };
const timeline = createTimeline(context);
timeline.renderDurationChart(trace);
assert.match(dom.chartSub.textContent, /部分耗时已记录/);
assert.match(dom.timelineVisual.innerHTML, /1 项耗时未记录，未绘制/);
assert.equal((dom.timelineVisual.innerHTML.match(/class="chart-bar/g) || []).length, 3);
timeline.renderCallsChart(trace);
assert.match(dom.chartSub.textContent, /1 次调用已记录 · 1 次耗时未记录/);
assert.equal((dom.timelineVisual.innerHTML.match(/class="chart-bar/g) || []).length, 1);
console.log('partial timeline renders recorded spans without inventing missing times');
