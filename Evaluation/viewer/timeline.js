export function createTimeline(context) {
  const { state, dom, activeCase, $$, formatAxis, escapeAttr, formatDuration, number, formatSeconds, escapeHTML, pad } = context;
  function makeBar(step, start, duration, total, type, role) {
    const safeTotal = Math.max(1, total);
    const left = Math.min(99.6, Math.max(0, (Number(start || 0) / safeTotal) * 100));
    const width = Math.min(100 - left, Math.max(0.45, (Number(duration || 0) / safeTotal) * 100));
    const title = `第 ${step.n} 轮 · ${type === "model" ? "模型" : step.label} · ${formatSeconds(duration)} · ${formatAxis(start)}`;
    return `<button type="button" class="chart-bar ${type}" data-chart-step="${step.n}" data-chart-role="${role}" style="left:${left.toFixed(3)}%;width:${width.toFixed(3)}%" title="${escapeAttr(title)}" aria-label="${escapeAttr(title)}"></button>`;
  }

  function axisHTML(total) {
    const marks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => formatAxis(total * ratio));
    return `<div class="chart-axis"><span class="axis-label">时间轴</span><div class="axis-marks">${marks.map((mark) => `<span>${mark}</span>`).join("")}</div></div>`;
  }

  function laneHTML(label, className, bars) {
    return `<div class="chart-lane"><span class="chart-lane-label">${escapeHTML(label)}</span><div class="chart-track">${bars.join("")}</div></div>`;
  }

  function recordedSpan(start, duration) {
    return typeof start === "number" && Number.isFinite(start) && start >= 0 &&
      typeof duration === "number" && Number.isFinite(duration) && duration >= 0;
  }

  function recordedSpans(c) {
    const model = c.steps.filter((step) => recordedSpan(step.modelStart, step.model_seconds));
    const tools = c.steps.filter((step) => step.tool && recordedSpan(step.toolStart, step.tool_seconds));
    const expected = c.steps.length + c.steps.filter((step) => step.tool).length;
    const missing = Math.max(0, expected - model.length - tools.length);
    const ends = [...model.map((s) => s.modelStart + s.model_seconds), ...tools.map((s) => s.toolStart + s.tool_seconds)];
    const total = Math.max(Number(c.durationSeconds) || 0, ...ends, 1);
    return { model, tools, expected, missing, total };
  }

  function renderDurationChart(c) {
    const spans = recordedSpans(c);
    if (!spans.model.length && !spans.tools.length) {
      dom.chartTitle.textContent = "执行时间线";
      dom.chartSub.textContent = "时间未记录";
      dom.chartLegend.innerHTML = "";
      dom.timelineVisual.innerHTML = "<div class=\"log-empty\">此轨迹未记录完整的步骤耗时，无法绘制时间线。</div>";
      return;
    }
    const total = spans.total;
    const modelBars = spans.model.map((step) => makeBar(step, step.modelStart, step.model_seconds, total, "model", "assistant"));
    const searchBars = spans.tools
      .filter((step) => step.kind === "search")
      .map((step) => makeBar(step, step.toolStart, step.tool_seconds, total, "search", "tool"));
    const readBars = spans.tools
      .filter((step) => step.kind === "read")
      .map((step) => makeBar(step, step.toolStart, step.tool_seconds, total, "read", "tool"));
    const computeBars = spans.tools
      .filter((step) => step.kind === "compute")
      .map((step) => makeBar(step, step.toolStart, step.tool_seconds, total, "compute", "tool"));
    const readLane = laneHTML("读取 / 计算", "read", [...readBars, ...computeBars]);
    dom.chartTitle.textContent = "执行时间线";
    dom.chartSub.textContent = spans.missing ? `${formatDuration(total)} · 部分耗时已记录` : `${formatDuration(total)} · 真实调用跨度`;
    dom.chartLegend.innerHTML = [
      ["model", "模型"],
      ["search", "搜索"],
      ["read", "读取"],
      c.stats.computeCalls ? ["compute", "计算"] : null,
    ]
      .filter(Boolean)
      .map(([type, label]) => `<span><i class="legend-dot ${type}"></i>${label}</span>`)
      .join("");
    dom.timelineVisual.innerHTML = `${axisHTML(total)}<div class="chart-lanes">${laneHTML("模型", "model", modelBars)}${laneHTML("搜索", "search", searchBars)}${readLane}</div><div class="chart-footnote"><span>记录窗口 ${formatDuration(total)}${spans.missing ? ` · ${number(spans.missing)} 项耗时未记录，未绘制` : ""}</span><span>点击色块定位事件</span></div>`;
  }

  function renderRoundsChart(c) {
    const cells = c.steps
      .map((step) => {
        const kind = step.kind === "answer" ? "answer" : step.kind;
        const label = `第 ${step.n} 轮 · ${step.label} · ${formatSeconds(step.model_seconds == null || step.tool_seconds == null ? null : step.model_seconds + step.tool_seconds)}`;
        return `<button type="button" class="round-cell ${kind}" data-chart-step="${step.n}" data-chart-role="assistant" title="${escapeAttr(label)}" aria-label="${escapeAttr(label)}"></button>`;
      })
      .join("");
    dom.chartTitle.textContent = "轮次分布";
    dom.chartSub.textContent = `${c.stats.steps} 轮 · ${number(c.stats.searchCalls + c.stats.readCalls + c.stats.computeCalls)} 次工具调用`;
    dom.chartLegend.innerHTML = [
      ["search", "搜索"],
      ["read", "读取"],
      ["compute", "计算"],
      ["answer", "综合"],
    ]
      .filter(([type]) => type !== "compute" || c.stats.computeCalls)
      .map(([type, label]) => `<span><i class="legend-dot ${type}"></i>${label}</span>`)
      .join("");
    dom.timelineVisual.innerHTML = `<div class="round-chart">
      <div class="round-head"><span>第 1 轮 → 第 ${c.stats.steps} 轮</span><span>点击轮次定位</span></div>
      <div class="round-strip">${cells}</div>
      <div class="round-labels"><span>01</span><span>${pad(c.stats.steps)}</span></div>
      <div class="call-summary"><span>搜索 <b>${c.stats.searchCalls}</b></span><span>读取 <b>${c.stats.readCalls}</b></span><span>计算 <b>${c.stats.computeCalls}</b></span></div>
    </div>`;
  }

  function renderCallsChart(c) {
    const spans = recordedSpans(c);
    if (!spans.tools.length) {
      dom.chartTitle.textContent = "工具调用";
      dom.chartSub.textContent = `${number(c.stats.searchCalls + c.stats.readCalls + c.stats.computeCalls)} 次调用 · 时间未记录`;
      dom.chartLegend.innerHTML = "";
      dom.timelineVisual.innerHTML = "<div class=\"log-empty\">此轨迹未记录工具调用耗时，无法绘制时间线。</div>";
      return;
    }
    const total = spans.total;
    const searchBars = spans.tools
      .filter((step) => step.kind === "search")
      .map((step) => makeBar(step, step.toolStart, step.tool_seconds, total, "search", "tool"));
    const readBars = spans.tools
      .filter((step) => step.kind === "read")
      .map((step) => makeBar(step, step.toolStart, step.tool_seconds, total, "read", "tool"));
    const computeBars = spans.tools
      .filter((step) => step.kind === "compute")
      .map((step) => makeBar(step, step.toolStart, step.tool_seconds, total, "compute", "tool"));
    dom.chartTitle.textContent = "工具调用";
    const missingTools = c.steps.filter((step) => step.tool).length - spans.tools.length;
    dom.chartSub.textContent = `${number(spans.tools.length)} 次调用已记录${missingTools ? ` · ${number(missingTools)} 次耗时未记录` : ""} · ${formatDuration(total)}`;
    dom.chartLegend.innerHTML = [
      ["search", `搜索 ${c.stats.searchCalls}`],
      ["read", `读取 ${c.stats.readCalls}`],
      ["compute", `计算 ${c.stats.computeCalls}`],
    ]
      .map(([type, label]) => `<span><i class="legend-dot ${type}"></i>${label}</span>`)
      .join("");
    dom.timelineVisual.innerHTML = `${axisHTML(total)}<div class="chart-lanes">
      ${laneHTML("搜索", "search", searchBars)}
      ${laneHTML("读取", "read", readBars)}
      ${laneHTML("计算", "compute", computeBars)}
    </div><div class="chart-footnote"><span>调用跨度 ${formatDuration(total)}</span><span>点击色块定位工具事件</span></div>`;
  }

  function renderChart() {
    if (!state.data) return;
    const c = activeCase();
    $$(".metric-button").forEach((button) => {
      const active = button.dataset.chartMode === state.chartMode;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    if (state.chartMode === "rounds") renderRoundsChart(c);
    else if (state.chartMode === "calls") renderCallsChart(c);
    else renderDurationChart(c);
  }


  return { makeBar, axisHTML, laneHTML, renderDurationChart, renderRoundsChart, renderCallsChart, renderChart };
}
