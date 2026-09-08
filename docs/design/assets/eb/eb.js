/* ==========================================================================
   EmbodiedBein 设计文档共享脚本
   职责：统一顶栏/页脚导航注入、自动目录与滚动高亮、Tab 交互、
         以及一个零依赖的 SVG 图表库（供高保真 UI 原型使用）。
   所有图表数据均为确定性伪随机生成的示意数据，非真实实验结果。
   ========================================================================== */
(function () {
  'use strict';

  var PAGES = [
    { file: 'index.html',           nav: '总览',   num: '00', title: '平台总览与设计导航' },
    { file: '01-architecture.html', nav: '架构',   num: '01', title: '系统架构与技术选型' },
    { file: '02-domain-model.html', nav: '领域模型', num: '02', title: '领域模型、本体环境与接口' },
    { file: '03-data.html',         nav: '数据',   num: '03', title: '数据管理模块' },
    { file: '04-model.html',        nav: '模型',   num: '04', title: '模型与本体资产管理' },
    { file: '05-training.html',     nav: '训练',   num: '05', title: '任务编排与训练可视化' },
    { file: '06-evaluation.html',   nav: '评估',   num: '06', title: '评估体系与可视化' },
    { file: '07-deploy.html',       nav: '部署',   num: '07', title: '策略发布与真机部署' },
    { file: '08-agent.html',        nav: 'Agent', num: '08', title: '平台 Agent 与自动闭环' },
    { file: '09-ui-system.html',    nav: 'UI',    num: '09', title: 'UI 设计系统与协作展示' },
    { file: '10-roadmap.html',      nav: '路线图', num: '10', title: '实施路线图与风险登记' },
    { file: '11-workspace-views.html', nav: '视图', num: '11', title: '数据、本体环境、配置工作台视图' },
    { file: '12-embodiment-environment.html', nav: '本体', num: '12', title: '本体与本体环境' },
    { file: '13-embodiment-crud-prototype.html', nav: '原型', num: '13', title: '本体环境 CRUD 原型' }
  ];

  var EB = window.EB = {};
  EB.pages = PAGES;

  /* ---------------- 基础工具 ---------------- */
  function el(tag, attrs, kids) {
    var n = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (var k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) n.setAttribute(k, attrs[k]);
    (kids || []).forEach(function (c) { n.appendChild(c); });
    return n;
  }
  function txt(x, y, s, cls, extra) {
    var a = { x: x, y: y, class: cls || 'svg-t-sm' };
    for (var k in (extra || {})) a[k] = extra[k];
    var n = el('text', a); n.textContent = s; return n;
  }
  function currentFile() {
    var p = location.pathname.split('/').pop();
    return (!p || p === '') ? 'index.html' : p;
  }
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  EB.esc = esc;

  /* 确定性伪随机（mulberry32），保证每次刷新图表一致 */
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  EB.rng = rng;

  /* 曲线生成器：用于示意训练/评估曲线 */
  function gen(spec) {
    var n = spec.n || 100, r = rng(spec.seed || 1), out = [], i, t, base;
    var from = spec.from !== undefined ? spec.from : 0;
    var to = spec.to !== undefined ? spec.to : 1;
    var noise = spec.noise !== undefined ? spec.noise : 0.05;
    var kind = spec.kind || 'rise';
    var drift = 0;
    for (i = 0; i < n; i++) {
      t = n === 1 ? 1 : i / (n - 1);
      if (kind === 'rise') base = from + (to - from) * (1 - Math.exp(-3.1 * t));
      else if (kind === 'fall') base = to + (from - to) * Math.exp(-2.7 * t);
      else if (kind === 'linear') base = from + (to - from) * t;
      else if (kind === 'step') base = from + (to - from) * (t < 0.45 ? t * 0.5 : 0.225 + (t - 0.45) * 1.41);
      else if (kind === 'plateau') base = from + (to - from) * Math.min(1, 1.35 * (1 - Math.exp(-4.2 * t)));
      else if (kind === 'noise') base = from + (to - from) * (0.5 + 0.5 * Math.sin(t * 9.3));
      else if (kind === 'walk') { drift += (r() - 0.5) * (to - from) * 0.16; base = from + (to - from) * t * 0.25 + drift; }
      else base = from + (to - from) * t;
      var amp = (Math.abs(to - from) || 1) * noise;
      out.push(base + (r() - 0.5) * 2 * amp * (kind === 'rise' || kind === 'plateau' ? (1.25 - 0.55 * t) : 1));
    }
    return out;
  }
  EB.gen = gen;

  function extent(arrs) {
    var lo = Infinity, hi = -Infinity;
    arrs.forEach(function (a) { a.forEach(function (v) { if (v < lo) lo = v; if (v > hi) hi = v; }); });
    if (lo === Infinity) { lo = 0; hi = 1; }
    if (lo === hi) { lo -= 1; hi += 1; }
    return [lo, hi];
  }
  function fmt(v) {
    var a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(1) + 'B';
    if (a >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (a >= 1e4) return (v / 1e3).toFixed(0) + 'k';
    if (a >= 1000) return (v / 1000).toFixed(1) + 'k';
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    if (a === 0) return '0';
    return v.toFixed(3);
  }
  EB.fmt = fmt;

  var PALETTE = ['#53d4ff', '#55d6be', '#b28dff', '#ffc857', '#ff884d', '#ff7ac0', '#5b8cff', '#53e5a3'];

  /* ---------------- 图表：折线（可带 ±std 带） ---------------- */
  function drawLine(host, spec) {
    var W = spec.w || 620, H = spec.h || 190;
    var m = spec.margin || { t: 10, r: 12, b: 24, l: 40 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var series = spec.series.map(function (s, i) {
      var data = s.data || gen(s.gen || { n: 90, seed: i + 3 });
      return { name: s.name, color: s.color || PALETTE[i % PALETTE.length], data: data, band: s.band, dash: s.dash };
    });
    var ex = extent(series.map(function (s) { return s.data; }));
    var y0 = spec.ymin !== undefined ? spec.ymin : ex[0] - (ex[1] - ex[0]) * 0.1;
    var y1 = spec.ymax !== undefined ? spec.ymax : ex[1] + (ex[1] - ex[0]) * 0.1;
    var n = Math.max.apply(null, series.map(function (s) { return s.data.length; }));
    var xmax = spec.xmax || (n - 1);
    var X = function (i) { return m.l + (n <= 1 ? 0 : i / (n - 1) * iw); };
    var Y = function (v) { return m.t + ih - (v - y0) / (y1 - y0) * ih; };

    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'chart', role: 'img',
      'aria-label': spec.aria || spec.title || '折线图' });

    var ticks = spec.yticks || 4, i;
    for (i = 0; i <= ticks; i++) {
      var v = y0 + (y1 - y0) * i / ticks, yy = Y(v);
      svg.appendChild(el('line', { x1: m.l, y1: yy, x2: W - m.r, y2: yy,
        stroke: i === 0 ? '#284052' : '#16293a', 'stroke-width': 1 }));
      svg.appendChild(txt(m.l - 6, yy + 3.5, fmt(v), 'svg-t-mono', { 'text-anchor': 'end' }));
    }
    var xt = spec.xticks || 4;
    for (i = 0; i <= xt; i++) {
      var xi = Math.round((n - 1) * i / xt), xx = X(xi);
      svg.appendChild(txt(xx, H - 7, fmt(Math.round(xmax * i / xt)), 'svg-t-mono', { 'text-anchor': 'middle' }));
    }
    if (spec.ylabel) svg.appendChild(txt(4, 10, spec.ylabel, 'svg-t-mono'));
    if (spec.xlabel) svg.appendChild(txt(W - m.r, H - 7, spec.xlabel, 'svg-t-mono', { 'text-anchor': 'end' }));

    (spec.hlines || []).forEach(function (h) {
      var yy = Y(h.v);
      svg.appendChild(el('line', { x1: m.l, y1: yy, x2: W - m.r, y2: yy,
        stroke: h.color || '#ff5d68', 'stroke-width': 1.2, 'stroke-dasharray': '5 4' }));
      if (h.label) svg.appendChild(txt(W - m.r - 3, yy - 4, h.label, 'svg-t-mono',
        { 'text-anchor': 'end', fill: h.color || '#ff5d68' }));
    });

    series.forEach(function (s) {
      if (s.band) {
        var up = [], dn = [];
        s.data.forEach(function (v, k) {
          var sd = (typeof s.band === 'number' ? s.band : 0.08) * (Math.abs(y1 - y0));
          var w = sd * (1.15 - 0.5 * (k / Math.max(1, s.data.length - 1)));
          up.push(X(k) + ',' + Y(Math.min(y1, v + w)));
          dn.unshift(X(k) + ',' + Y(Math.max(y0, v - w)));
        });
        svg.appendChild(el('polygon', { points: up.concat(dn).join(' '), fill: s.color, opacity: 0.13 }));
      }
      var pts = s.data.map(function (v, k) { return X(k) + ',' + Y(v); }).join(' ');
      svg.appendChild(el('polyline', { points: pts, fill: 'none', stroke: s.color,
        'stroke-width': spec.lw || 1.7, 'stroke-linejoin': 'round', 'stroke-linecap': 'round',
        'stroke-dasharray': s.dash || null }));
      if (spec.dot !== false) {
        var last = s.data.length - 1;
        svg.appendChild(el('circle', { cx: X(last), cy: Y(s.data[last]), r: 2.6, fill: s.color }));
      }
    });

    host.appendChild(svg);
    if (spec.legend !== false && series.length > 1) {
      var lg = document.createElement('div');
      lg.className = 'legend';
      lg.innerHTML = series.map(function (s) {
        return '<span><i style="background:' + s.color + '"></i>' + esc(s.name || '') + '</span>';
      }).join('');
      host.appendChild(lg);
    }
  }

  /* ---------------- 图表：柱状 / 堆叠 ---------------- */
  function drawBars(host, spec) {
    var W = spec.w || 620, H = spec.h || 190;
    var m = spec.margin || { t: 10, r: 12, b: 30, l: 40 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var cats = spec.cats, series = spec.series.map(function (s, i) {
      return { name: s.name, color: s.color || PALETTE[i % PALETTE.length], data: s.data };
    });
    var tot = cats.map(function (_, i) {
      return spec.stacked ? series.reduce(function (a, s) { return a + (s.data[i] || 0); }, 0)
                          : Math.max.apply(null, series.map(function (s) { return s.data[i] || 0; }));
    });
    var y1 = spec.ymax !== undefined ? spec.ymax : Math.max.apply(null, tot) * 1.12;
    var Y = function (v) { return m.t + ih - v / y1 * ih; };
    var bw = iw / cats.length;
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'chart', role: 'img',
      'aria-label': spec.aria || spec.title || '柱状图' });

    for (var i = 0; i <= (spec.yticks || 4); i++) {
      var v = y1 * i / (spec.yticks || 4), yy = Y(v);
      svg.appendChild(el('line', { x1: m.l, y1: yy, x2: W - m.r, y2: yy,
        stroke: i === 0 ? '#284052' : '#16293a', 'stroke-width': 1 }));
      svg.appendChild(txt(m.l - 6, yy + 3.5, fmt(v), 'svg-t-mono', { 'text-anchor': 'end' }));
    }
    cats.forEach(function (c, ci) {
      var x0 = m.l + ci * bw, acc = 0;
      var inner = spec.stacked ? 1 : series.length;
      var pad = Math.min(9, bw * 0.22), w = (bw - pad) / inner;
      series.forEach(function (s, si) {
        var v = s.data[ci] || 0;
        var x = spec.stacked ? x0 + pad / 2 : x0 + pad / 2 + si * w;
        var y = spec.stacked ? Y(acc + v) : Y(v);
        var h = Math.max(0, (spec.stacked ? Y(acc) - Y(acc + v) : ih - (Y(v) - m.t)));
        svg.appendChild(el('rect', { x: x, y: y, width: Math.max(1, w - (spec.stacked ? 0 : 1.5)),
          height: Math.max(0.6, h), fill: s.color, rx: 2.5, opacity: 0.92 }));
        acc += v;
      });
      svg.appendChild(txt(x0 + bw / 2, H - 9, c, 'svg-t-mono', { 'text-anchor': 'middle' }));
    });
    host.appendChild(svg);
    if (spec.legend !== false && series.length > 1) {
      var lg = document.createElement('div'); lg.className = 'legend';
      lg.innerHTML = series.map(function (s) {
        return '<span><i style="background:' + s.color + '"></i>' + esc(s.name) + '</span>';
      }).join('');
      host.appendChild(lg);
    }
  }

  /* ---------------- 图表：堆叠面积 ---------------- */
  function drawArea(host, spec) {
    var W = spec.w || 620, H = spec.h || 190;
    var m = { t: 10, r: 12, b: 24, l: 40 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var series = spec.series.map(function (s, i) {
      return { name: s.name, color: s.color || PALETTE[i % PALETTE.length],
               data: s.data || gen(s.gen || { n: 60, seed: i + 11 }) };
    });
    var n = series[0].data.length;
    var totals = [];
    for (var k = 0; k < n; k++) totals.push(series.reduce(function (a, s) { return a + Math.max(0, s.data[k]); }, 0));
    var y1 = spec.ymax !== undefined ? spec.ymax : Math.max.apply(null, totals) * 1.05;
    var X = function (i) { return m.l + i / (n - 1) * iw; };
    var Y = function (v) { return m.t + ih - v / y1 * ih; };
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'chart', role: 'img',
      'aria-label': spec.aria || spec.title || '堆叠面积图' });
    for (var i = 0; i <= 4; i++) {
      var v = y1 * i / 4, yy = Y(v);
      svg.appendChild(el('line', { x1: m.l, y1: yy, x2: W - m.r, y2: yy, stroke: i === 0 ? '#284052' : '#16293a' }));
      svg.appendChild(txt(m.l - 6, yy + 3.5, spec.percent ? Math.round(v * 100) + '%' : fmt(v),
        'svg-t-mono', { 'text-anchor': 'end' }));
    }
    var base = new Array(n).fill(0);
    series.forEach(function (s) {
      var up = [], dn = [];
      for (var j = 0; j < n; j++) {
        var lo = base[j], hi = base[j] + Math.max(0, s.data[j]);
        up.push(X(j) + ',' + Y(hi)); dn.unshift(X(j) + ',' + Y(lo));
        base[j] = hi;
      }
      svg.appendChild(el('polygon', { points: up.concat(dn).join(' '), fill: s.color, opacity: 0.55,
        stroke: s.color, 'stroke-width': 0.8 }));
    });
    for (var t = 0; t <= 4; t++) {
      var xi = Math.round((n - 1) * t / 4);
      svg.appendChild(txt(X(xi), H - 7, fmt(Math.round((spec.xmax || n) * t / 4)), 'svg-t-mono', { 'text-anchor': 'middle' }));
    }
    host.appendChild(svg);
    var lg = document.createElement('div'); lg.className = 'legend';
    lg.innerHTML = series.map(function (s) {
      return '<span><i style="background:' + s.color + '"></i>' + esc(s.name) + '</span>';
    }).join('');
    host.appendChild(lg);
  }

  /* ---------------- 图表：散点（Pareto） ---------------- */
  function drawScatter(host, spec) {
    var W = spec.w || 480, H = spec.h || 260;
    var m = { t: 12, r: 14, b: 32, l: 44 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var pts = spec.points;
    var xs = pts.map(function (p) { return p.x; }), ys = pts.map(function (p) { return p.y; });
    var xe = spec.xrange || [Math.min.apply(null, xs) * 0.85, Math.max.apply(null, xs) * 1.1];
    var ye = spec.yrange || [Math.min.apply(null, ys) * 0.9, Math.max.apply(null, ys) * 1.05];
    var X = function (v) { return m.l + (v - xe[0]) / (xe[1] - xe[0]) * iw; };
    var Y = function (v) { return m.t + ih - (v - ye[0]) / (ye[1] - ye[0]) * ih; };
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'chart', role: 'img',
      'aria-label': spec.aria || 'Pareto 散点图' });
    var i;
    for (i = 0; i <= 4; i++) {
      var yy = m.t + ih * i / 4, vv = ye[1] - (ye[1] - ye[0]) * i / 4;
      svg.appendChild(el('line', { x1: m.l, y1: yy, x2: W - m.r, y2: yy, stroke: '#16293a' }));
      svg.appendChild(txt(m.l - 6, yy + 3.5, fmt(vv), 'svg-t-mono', { 'text-anchor': 'end' }));
      var xx = m.l + iw * i / 4, xv = xe[0] + (xe[1] - xe[0]) * i / 4;
      svg.appendChild(el('line', { x1: xx, y1: m.t, x2: xx, y2: m.t + ih, stroke: '#16293a' }));
      svg.appendChild(txt(xx, H - 16, fmt(xv), 'svg-t-mono', { 'text-anchor': 'middle' }));
    }
    if (spec.front) {
      var f = spec.front.map(function (p) { return X(p.x) + ',' + Y(p.y); }).join(' ');
      svg.appendChild(el('polyline', { points: f, fill: 'none', stroke: '#53e5a3',
        'stroke-width': 1.3, 'stroke-dasharray': '4 3' }));
    }
    pts.forEach(function (p) {
      svg.appendChild(el('circle', { cx: X(p.x), cy: Y(p.y), r: p.r || 5,
        fill: p.color || '#53d4ff', opacity: 0.85, stroke: '#071018', 'stroke-width': 1 }));
      if (p.label) svg.appendChild(txt(X(p.x) + (p.r || 5) + 4, Y(p.y) + 3.5, p.label, 'svg-t-mono'));
    });
    if (spec.xlabel) svg.appendChild(txt(m.l + iw / 2, H - 3, spec.xlabel, 'svg-t-sm', { 'text-anchor': 'middle' }));
    if (spec.ylabel) svg.appendChild(txt(3, 10, spec.ylabel, 'svg-t-sm'));
    host.appendChild(svg);
  }

  /* ---------------- 图表：漏斗 ---------------- */
  function drawFunnel(host, spec) {
    var stages = spec.stages, W = spec.w || 560;
    var rowH = spec.rowH || 34, H = stages.length * rowH + 14;
    var max = stages[0].v;
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'chart', role: 'img',
      'aria-label': spec.aria || '数据处理漏斗' });
    var labelW = spec.labelW || 176;
    stages.forEach(function (s, i) {
      var y = 8 + i * rowH;
      var frac = max > 0 ? s.v / max : 0;
      var w = Math.max(2.5, frac * (W - labelW - 92));
      svg.appendChild(txt(labelW - 8, y + 15, s.name, 'svg-t', { 'text-anchor': 'end' }));
      svg.appendChild(el('rect', { x: labelW, y: y + 4, width: W - labelW - 88, height: rowH - 13,
        fill: '#0b1a26', stroke: '#1d3243', rx: 4 }));
      svg.appendChild(el('rect', { x: labelW, y: y + 4, width: w, height: rowH - 13,
        fill: s.color || (i === stages.length - 1 ? '#53e5a3' : '#53d4ff'), opacity: 0.82, rx: 4 }));
      svg.appendChild(txt(W - 84, y + 15, s.v.toLocaleString('en-US'), 'svg-t-mono',
        { 'text-anchor': 'start', fill: '#edf6fb' }));
      if (i > 0) {
        var keep = stages[i - 1].v > 0 ? (s.v / stages[i - 1].v * 100) : 0;
        svg.appendChild(txt(W - 4, y + 15, (keep < 1 ? keep.toFixed(2) : keep.toFixed(1)) + '%',
          'svg-t-mono', { 'text-anchor': 'end', fill: keep < 5 ? '#ff5d68' : '#9fb4c2' }));
      }
    });
    host.appendChild(svg);
  }

  /* ---------------- 图表：仪表 ---------------- */
  function drawGauge(host, spec) {
    var W = spec.w || 130, H = spec.h || 88, cx = W / 2, cy = H - 12, r = Math.min(W / 2 - 8, H - 22);
    var v = Math.max(0, Math.min(1, spec.value));
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'chart', role: 'img',
      'aria-label': (spec.label || '指标') + ' ' + Math.round(v * 100) + '%' });
    function arc(frac, color, width) {
      var a0 = Math.PI, a1 = Math.PI - Math.PI * frac;
      var x0 = cx + r * Math.cos(a0), y0 = cy + r * Math.sin(a0) * -1;
      var x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1) * -1;
      return el('path', { d: 'M ' + x0 + ' ' + y0 + ' A ' + r + ' ' + r + ' 0 0 1 ' + x1 + ' ' + y1,
        fill: 'none', stroke: color, 'stroke-width': width || 8, 'stroke-linecap': 'round' });
    }
    svg.appendChild(arc(1, '#16293a'));
    svg.appendChild(arc(v, spec.color || (v >= (spec.good || 0.8) ? '#53e5a3' : v >= (spec.warn || 0.5) ? '#ffc857' : '#ff5d68')));
    if (spec.threshold !== undefined) {
      var a = Math.PI - Math.PI * spec.threshold;
      svg.appendChild(el('line', { x1: cx + (r - 7) * Math.cos(a), y1: cy - (r - 7) * Math.sin(a),
        x2: cx + (r + 7) * Math.cos(a), y2: cy - (r + 7) * Math.sin(a), stroke: '#ff5d68', 'stroke-width': 1.6 }));
    }
    svg.appendChild(txt(cx, cy - 4, (v * 100).toFixed(spec.digits === undefined ? 1 : spec.digits) + '%', 'svg-t',
      { 'text-anchor': 'middle', 'font-size': 18, 'font-weight': 800, fill: '#edf6fb' }));
    if (spec.label) svg.appendChild(txt(cx, cy + 9, spec.label, 'svg-t-sm', { 'text-anchor': 'middle' }));
    host.appendChild(svg);
  }

  /* ---------------- 图表：GPU 甘特 ---------------- */
  function drawGantt(host, spec) {
    var rows = spec.rows, W = spec.w || 660, rowH = 26, H = rows.length * rowH + 26;
    var labelW = spec.labelW || 62, tmax = spec.tmax || 24;
    var X = function (t) { return labelW + t / tmax * (W - labelW - 8); };
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'chart', role: 'img',
      'aria-label': spec.aria || 'GPU 占用甘特图' });
    var i;
    for (i = 0; i <= 8; i++) {
      var xx = X(tmax * i / 8);
      svg.appendChild(el('line', { x1: xx, y1: 14, x2: xx, y2: H - 14, stroke: '#16293a' }));
      svg.appendChild(txt(xx, H - 3, '+' + Math.round(tmax * i / 8) + 'h', 'svg-t-mono', { 'text-anchor': 'middle' }));
    }
    rows.forEach(function (row, ri) {
      var y = 14 + ri * rowH;
      svg.appendChild(txt(4, y + 15, row.name, 'svg-t-mono', { fill: '#bdd3e0' }));
      svg.appendChild(el('rect', { x: labelW, y: y + 4, width: W - labelW - 8, height: rowH - 10,
        fill: 'rgba(11,26,38,.7)', stroke: '#16293a', rx: 4 }));
      (row.bars || []).forEach(function (b) {
        var x = X(b.t0), w = Math.max(3, X(b.t1) - X(b.t0));
        svg.appendChild(el('rect', { x: x, y: y + 4, width: w, height: rowH - 10,
          fill: b.color || '#53d4ff', opacity: b.ghost ? 0.3 : 0.85, rx: 4,
          stroke: b.ghost ? '#3d5f78' : 'none', 'stroke-dasharray': b.ghost ? '3 3' : null }));
        if (w > 52 && b.label) {
          var t = txt(x + 5, y + 18, b.label, 'svg-t-mono', { fill: '#04121a', 'font-weight': 700 });
          svg.appendChild(t);
        }
      });
    });
    host.appendChild(svg);
  }

  /* ---------------- 图表：小型 sparkline ---------------- */
  function drawSpark(host, spec) {
    var W = spec.w || 96, H = spec.h || 26;
    var d = spec.data || gen(spec.gen || { n: 30, seed: 5 });
    var ex = extent([d]);
    var X = function (i) { return i / (d.length - 1) * (W - 2) + 1; };
    var Y = function (v) { return H - 2 - (v - ex[0]) / (ex[1] - ex[0]) * (H - 4); };
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'chart', 'aria-hidden': 'true' });
    var pts = d.map(function (v, i) { return X(i) + ',' + Y(v); }).join(' ');
    svg.appendChild(el('polygon', { points: pts + ' ' + X(d.length - 1) + ',' + H + ' ' + X(0) + ',' + H,
      fill: spec.color || '#53d4ff', opacity: 0.14 }));
    svg.appendChild(el('polyline', { points: pts, fill: 'none', stroke: spec.color || '#53d4ff', 'stroke-width': 1.4 }));
    host.appendChild(svg);
  }

  /* ---------------- 热力图（DOM 网格） ---------------- */
  function drawHeat(host, spec) {
    var rows = spec.rows, cols = spec.cols, vals = spec.values;
    var lo = spec.min !== undefined ? spec.min : 0, hi = spec.max !== undefined ? spec.max : 1;
    var wrap = document.createElement('div');
    wrap.className = 'heat';
    wrap.style.gridTemplateColumns = (spec.labelW || 120) + 'px repeat(' + cols.length + ', minmax(0,1fr))';
    var head = ['<div class="heat-lbl"></div>'].concat(cols.map(function (c) {
      return '<div class="heat-lbl">' + esc(c) + '</div>';
    }));
    var html = head.join('');
    rows.forEach(function (r, ri) {
      html += '<div class="heat-lbl" style="justify-content:flex-end;text-align:right;padding-right:4px">' + esc(r) + '</div>';
      cols.forEach(function (_, ci) {
        var v = vals[ri][ci];
        if (v === null || v === undefined) {
          html += '<div class="heat-cell" style="background:#0b1a26;color:#425a6b">·</div>';
          return;
        }
        var t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
        var col = spec.scheme === 'rg'
          ? 'hsl(' + Math.round(4 + t * 138) + ',' + Math.round(52 + t * 18) + '%,' + Math.round(42 + t * 16) + '%)'
          : 'hsl(' + Math.round(196 - t * 26) + ',' + Math.round(30 + t * 62) + '%,' + Math.round(20 + t * 42) + '%)';
        html += '<div class="heat-cell" title="' + esc(r) + ' / ' + esc(cols[ci]) + ' = ' + v + '" style="background:' +
          col + ';color:' + (t > 0.52 ? '#04121a' : '#dbe9f2') + '">' +
          (spec.percent ? Math.round(v * 100) : (typeof v === 'number' ? fmt(v) : v)) + '</div>';
      });
    });
    wrap.innerHTML = html;
    host.appendChild(wrap);
  }

  var RENDERERS = { line: drawLine, bars: drawBars, area: drawArea, scatter: drawScatter,
    funnel: drawFunnel, gauge: drawGauge, gantt: drawGantt, spark: drawSpark, heat: drawHeat };

  EB.render = function (host, spec) {
    var fn = RENDERERS[spec.type];
    if (!fn) { host.innerHTML = '<p class="xs c-red">未知图表类型: ' + esc(spec.type) + '</p>'; return; }
    try { host.innerHTML = ''; fn(host, spec); }
    catch (e) { host.innerHTML = '<p class="xs c-red">图表渲染失败：' + esc(e.message) + '</p>'; }
  };

  function mountCharts(root) {
    (root || document).querySelectorAll('[data-chart]').forEach(function (host) {
      if (host.dataset.mounted) return;
      var spec;
      try { spec = JSON.parse(host.getAttribute('data-chart')); }
      catch (e) { host.innerHTML = '<p class="xs c-red">图表配置 JSON 解析失败</p>'; return; }
      EB.render(host, spec);
      host.dataset.mounted = '1';
    });
  }

  /* ---------------- 顶栏 / 页脚导航 ---------------- */
  function mountChrome() {
    var here = currentFile();
    var slot = document.getElementById('eb-topbar');
    if (slot) {
      var links = PAGES.map(function (p) {
        return '<a href="' + p.file + '"' + (p.file === here ? ' aria-current="page"' : '') +
          ' title="' + esc(p.num + ' · ' + p.title) + '">' + esc(p.nav) + '</a>';
      }).join('');
      slot.outerHTML =
        '<header class="topbar"><div class="topbar-in">' +
        '<a class="brand" href="index.html"><span class="brand-mark">EB</span>' +
        '<span class="brand-name">EmbodiedBein</span>' +
        '<span class="brand-sub">设计文档</span></a>' +
        '<nav class="topnav" aria-label="文档章节导航">' + links + '</nav>' +
        '</div></header>';
    }
    var pg = document.getElementById('eb-pager');
    if (pg) {
      var idx = PAGES.findIndex(function (p) { return p.file === here; });
      var prev = idx > 0 ? PAGES[idx - 1] : null, next = idx >= 0 && idx < PAGES.length - 1 ? PAGES[idx + 1] : null;
      var h = '';
      if (prev) h += '<a href="' + prev.file + '"><span class="dir">← 上一章 ' + prev.num +
        '</span><span class="ttl">' + esc(prev.title) + '</span></a>';
      if (next) h += '<a class="rev" href="' + next.file + '"><span class="dir">下一章 ' + next.num +
        ' →</span><span class="ttl">' + esc(next.title) + '</span></a>';
      pg.outerHTML = '<nav class="pager" aria-label="上下章导航">' + h + '</nav>';
    }
    var ft = document.getElementById('eb-foot');
    if (ft) {
      ft.outerHTML = '<footer class="site-foot"><div class="wrap-wide">' +
        'EmbodiedBein · 具身智能研发管理平台设计文档 · v0.1 草案 ' +
        '<span class="muted-2">｜ 文档内所有图表与界面均为设计示意，数值为占位数据，需以实测校准。</span>' +
        '</div></footer>';
    }
  }

  /* ---------------- 自动目录 ---------------- */
  function mountToc() {
    var toc = document.getElementById('eb-toc');
    if (!toc) return;
    var content = document.querySelector('.content') || document.body;
    var hs = content.querySelectorAll('h2[id], h3[id]');
    if (!hs.length) { toc.remove(); return; }
    var html = '<p class="toc-title">本页目录</p>';
    hs.forEach(function (h) {
      html += '<a class="lvl-' + h.tagName[1] + '" href="#' + h.id + '">' +
        esc(h.textContent.replace(/\s*#\s*$/, '')) + '</a>';
    });
    toc.innerHTML = html;

    var links = Array.prototype.slice.call(toc.querySelectorAll('a'));
    var map = {};
    links.forEach(function (a) { map[a.getAttribute('href').slice(1)] = a; });
    if ('IntersectionObserver' in window) {
      var visible = new Set();
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (en.isIntersecting) visible.add(en.target.id); else visible.delete(en.target.id);
        });
        var first = null;
        hs.forEach(function (h) { if (!first && visible.has(h.id)) first = h.id; });
        links.forEach(function (a) { a.classList.remove('active'); });
        if (first && map[first]) map[first].classList.add('active');
      }, { rootMargin: '-80px 0px -65% 0px' });
      hs.forEach(function (h) { io.observe(h); });
    }
    // 标题锚点
    hs.forEach(function (h) {
      var a = document.createElement('a');
      a.className = 'anchor'; a.href = '#' + h.id; a.textContent = '#';
      a.setAttribute('aria-label', '本节链接');
      h.appendChild(a);
    });
  }

  /* ---------------- Tabs ---------------- */
  function mountTabs() {
    document.querySelectorAll('.tabs').forEach(function (box) {
      var bar = box.querySelector('.tabbar');
      if (!bar) return;
      var btns = Array.prototype.slice.call(bar.querySelectorAll('button'));
      var panels = Array.prototype.slice.call(box.querySelectorAll(':scope > .tabpanel'));
      function sel(i) {
        btns.forEach(function (b, j) { b.setAttribute('aria-selected', j === i ? 'true' : 'false'); });
        panels.forEach(function (p, j) { p.hidden = j !== i; mountCharts(p); });
      }
      btns.forEach(function (b, i) {
        b.setAttribute('role', 'tab');
        b.addEventListener('click', function () { sel(i); });
        b.addEventListener('keydown', function (e) {
          if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
            var d = e.key === 'ArrowRight' ? 1 : -1;
            var t = (i + d + btns.length) % btns.length;
            btns[t].focus(); sel(t); e.preventDefault();
          }
        });
      });
      bar.setAttribute('role', 'tablist');
      sel(0);
    });
  }

  /* ---------------- 表格排序 ---------------- */
  function mountSortable() {
    document.querySelectorAll('table.sortable').forEach(function (tb) {
      var ths = tb.querySelectorAll('thead th');
      ths.forEach(function (th, ci) {
        if (th.dataset.nosort !== undefined) return;
        th.style.cursor = 'pointer';
        th.title = '点击排序';
        var dir = 1;
        th.addEventListener('click', function () {
          var body = tb.tBodies[0];
          var rows = Array.prototype.slice.call(body.rows);
          rows.sort(function (a, b) {
            var x = (a.cells[ci] || {}).textContent || '', y = (b.cells[ci] || {}).textContent || '';
            var nx = parseFloat(x.replace(/[^\d.\-]/g, '')), ny = parseFloat(y.replace(/[^\d.\-]/g, ''));
            if (!isNaN(nx) && !isNaN(ny)) return (nx - ny) * dir;
            return x.localeCompare(y, 'zh') * dir;
          });
          rows.forEach(function (r) { body.appendChild(r); });
          dir = -dir;
        });
      });
    });
  }

  /* ---------------- 原型：日志流打字机 ---------------- */
  function mountLogStream() {
    document.querySelectorAll('.logbox[data-stream]').forEach(function (box) {
      var lines = Array.prototype.slice.call(box.children);
      if (!lines.length) return;
      if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
      lines.forEach(function (l) { l.style.display = 'none'; });
      var i = 0;
      var timer = setInterval(function () {
        if (i >= lines.length) { clearInterval(timer); return; }
        lines[i].style.display = ''; box.scrollTop = box.scrollHeight; i++;
      }, 260);
    });
  }

  /* ---------------- 原型：媒体网格点选 ---------------- */
  function mountMediaSelect() {
    document.querySelectorAll('.mediagrid[data-select]').forEach(function (g) {
      g.addEventListener('click', function (e) {
        var m = e.target.closest('.media');
        if (!m || !g.contains(m)) return;
        g.querySelectorAll('.media').forEach(function (x) { x.classList.remove('sel'); });
        m.classList.add('sel');
      });
    });
  }

  function init() {
    mountChrome();
    mountToc();
    mountTabs();
    mountCharts(document);
    mountSortable();
    mountLogStream();
    mountMediaSelect();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
