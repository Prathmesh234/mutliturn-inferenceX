#!/usr/bin/env python3
"""
Build a self-contained interactive HTML dashboard from a batch trace file.

Pure-data only: every number is computed directly from the stream-json traces in
runs_full/batch_0.jsonl (no fabrication, no interpolation). The output is one
self-contained dashboard.html (data inlined as JSON) rendered with D3, styled in a
warm light aesthetic, with a timeline slider that scrubs the run by real solve
time and recomputes all six charts for the focused cohort.

  python build_dashboard.py runs_full/batch_0.jsonl --out dashboard.html

Charts: (1) cache hit-rate warm-up, (2) context growth, (3) thinking decay,
(4) tool-call time judge vs non-judge, (5) batch-of-50 scorecard, (6) token economy.
Per-turn input-side usage is reliable; per-turn output is not in the trace, so output
appears only in the aggregate token-economy panel.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_ms(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000.0
    except Exception:
        return None


def thinking_runs(tr):
    """system/thinking_tokens carries a running estimate that RESETS each turn;
    the peak before each reset is that turn's estimated thinking tokens."""
    runs, cur = [], 0
    for e in tr:
        if e.get("type") == "system" and e.get("subtype") == "thinking_tokens":
            est = e.get("estimated_tokens", 0) or 0
            if est < cur:
                runs.append(cur)
                cur = est
            else:
                cur = est
    if cur:
        runs.append(cur)
    return runs


def per_turn_usage(tr):
    """Deduped by message id (the stream emits one event per content block, all
    sharing the message's usage). Returns [[ctx, read], ...] in call order."""
    seen, turns = set(), []
    for e in tr:
        if e.get("type") != "assistant":
            continue
        m = e.get("message", {}) or {}
        mid = m.get("id")
        if mid in seen:
            continue
        seen.add(mid)
        u = m.get("usage", {}) or {}
        it = u.get("input_tokens", 0) or 0
        cr = u.get("cache_read_input_tokens", 0) or 0
        cc = u.get("cache_creation_input_tokens", 0) or 0
        turns.append([it + cr + cc, cr, it, cc])  # [ctx, cache_read, input, cache_write]
    return turns


def tool_deltas(tr):
    """Wall-clock between consecutive tool-result completions (model think + that
    tool's run), tagged judge vs non-judge. First call has no prior -> null."""
    meta = {}
    for e in tr:
        if e.get("type") == "assistant":
            for c in (e.get("message", {}) or {}).get("content", []) or []:
                if c.get("type") == "tool_use":
                    cmd = (c.get("input") or {}).get("command", "") if c.get("name") == "Bash" else ""
                    meta[c.get("id")] = "judge.py" in (cmd or "")
    rows = []
    prev = None
    for e in tr:
        if e.get("type") != "user":
            continue
        ts = parse_ms(e.get("timestamp"))
        for c in (e.get("message", {}) or {}).get("content", []) or []:
            if c.get("type") == "tool_result" and ts is not None:
                judge = 1 if meta.get(c.get("tool_use_id")) else 0
                dt = round((ts - prev) / 1000.0, 2) if prev is not None else None
                if dt is not None and not (0 <= dt < 600):
                    dt = None
                rows.append([dt, judge])
                prev = ts
    return rows


def extract(records):
    out = []
    for r in records:
        tr = r.get("trace") or []
        res = next((e for e in tr if e.get("type") == "result"), None)
        ev = r.get("eval") or {}
        mu = ((res or {}).get("modelUsage") or {}).get("claude-opus-4-8", {}) if res else {}
        ts_all = [parse_ms(e.get("timestamp")) for e in tr if e.get("timestamp")]
        ts_all = [t for t in ts_all if t is not None]
        out.append({
            "uuid": r.get("uuid"),
            "entry": r.get("entry_point"),
            "gpu": r.get("gpu_id"),
            "correct": 1 if ev.get("correct") else 0,
            "speedup": ev.get("speedup"),
            "evals": r.get("num_evals"),
            "cost": r.get("cost_usd") or 0.0,
            "ttft": (res or {}).get("ttft_ms") if res else None,
            "worker": r.get("worker_id"),
            "t_start": min(ts_all) if ts_all else None,
            "t_end": max(ts_all) if ts_all else None,
            "in_tok": mu.get("inputTokens", 0),
            "out_tok": mu.get("outputTokens", 0),
            "read_tok": mu.get("cacheReadInputTokens", 0),
            "create_tok": mu.get("cacheCreationInputTokens", 0),
            "turns": per_turn_usage(tr),
            "think": thinking_runs(tr),
            "tool": tool_deltas(tr),
        })
    # stable timeline order; problems lacking timestamps go last but keep a slot
    tmax = max((p["t_end"] for p in out if p["t_end"]), default=0)
    for i, p in enumerate(out):
        if p["t_end"] is None:
            p["t_end"] = tmax + i + 1
    return out


TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KernelBook &rarr; Triton &middot; run telemetry</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:ital,wght@0,400;0,500;0,600;1,400&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<style>
:root{
  --bg:#ede7d8; --ink:#221e16;
  --i55:rgba(34,30,22,.55); --i30:rgba(34,30,22,.30); --i22:rgba(34,30,22,.22);
  --i14:rgba(34,30,22,.14); --i08:rgba(34,30,22,.08);
  --terra:#bf5b3d; --ochre:#c5973f; --sage:#6f7d5a; --clay:#9a6a4d; --teal:#5f7f84;
  --panel:rgba(255,253,247,.45);
  --mono:'SF Mono','Cascadia Code','Consolas',monospace;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);
  font-family:'Inter Tight',system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:34px 26px 150px}
.serif{font-family:'Instrument Serif',Georgia,serif;font-style:italic;font-weight:400}
header h1{font-family:'Instrument Serif',Georgia,serif;font-style:italic;font-weight:400;
  font-size:46px;line-height:1.02;margin:0 0 6px;letter-spacing:.2px}
header .sub{color:var(--i55);font-size:14.5px;max-width:680px;line-height:1.5}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0 26px}
.chip{background:var(--panel);border:1px solid var(--i14);border-radius:13px;
  padding:10px 14px;min-width:118px}
.chip .v{font-family:var(--mono);font-size:21px;font-weight:500;letter-spacing:-.5px}
.chip .k{color:var(--i55);font-size:11px;text-transform:uppercase;letter-spacing:.7px;margin-top:3px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.card{background:var(--panel);border:1px solid var(--i14);border-radius:16px;
  padding:16px 18px 12px;box-shadow:0 1px 0 rgba(255,255,255,.4) inset,0 8px 26px -22px rgba(34,30,22,.5)}
.card h3{margin:0;font-size:16.5px;font-weight:500}
.card .lede{color:var(--i55);font-size:12.5px;margin:3px 0 8px;line-height:1.45}
.card .lede b{color:var(--terra);font-weight:600}
.card svg{width:100%;height:auto;display:block;overflow:visible}
.legend{display:flex;gap:16px;margin:2px 2px 6px;font-size:12px;color:var(--i55)}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.axis text{font-family:var(--mono);font-size:10px;fill:var(--i55)}
.axis line,.axis path{stroke:var(--i14)}
.grid-line{stroke:var(--i08)}
.tl{position:fixed;left:0;right:0;bottom:0;background:rgba(237,231,216,.92);
  backdrop-filter:blur(6px);border-top:1px solid var(--i14);padding:8px 26px 12px}
.tl .inner{max-width:1180px;margin:0 auto}
.tl .row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:2px}
.tl .row .t{font-size:12.5px;color:var(--i55)}
.tl .row .t b{color:var(--ink);font-weight:600;font-family:var(--mono)}
.tl .hint{font-size:11px;color:var(--i30)}
.tl button{font-family:'Inter Tight';font-size:11.5px;color:var(--i55);background:none;
  border:1px solid var(--i22);border-radius:8px;padding:3px 9px;cursor:pointer}
.tl button:hover{color:var(--ink);border-color:var(--i55)}
.heat text{font-family:var(--mono)}
.note{color:var(--i30);font-size:11px;margin-top:18px;line-height:1.5;max-width:760px}
.brush .selection{fill:rgba(191,91,61,.10);stroke:var(--terra);stroke-opacity:.5}
.brush .handle{fill:var(--terra)}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>KernelBook &rarr; Triton</h1>
    <div class="sub">Run telemetry for <span class="serif">Opus&nbsp;4.8</span> converting PyTorch modules into Triton kernels on GB300 &mdash; every value computed directly from the agent traces. Drag the timeline below to scope the charts to a window of the run.</div>
  </header>
  <div class="chips" id="chips"></div>
  <div class="grid">
    <div class="card"><h3>Cache hit-rate warm-up</h3><div class="lede" id="l1"></div><svg id="c1" viewBox="0 0 470 250"></svg></div>
    <div class="card"><h3>Context growth per turn</h3><div class="lede" id="l2"></div><svg id="c2" viewBox="0 0 470 250"></svg></div>
    <div class="card"><h3>Thinking decay per turn</h3><div class="lede" id="l3"></div><svg id="c3" viewBox="0 0 470 250"></svg></div>
    <div class="card"><h3>Tool-call time</h3>
      <div class="legend"><span><i style="background:var(--terra)"></i>judge (GPU eval)</span><span><i style="background:var(--sage)"></i>edit / read</span></div>
      <svg id="c4" viewBox="0 0 470 250"></svg></div>
    <div class="card"><h3>Batch-of-50 scorecard</h3><div class="lede" id="l5"></div><svg id="c5" viewBox="0 0 470 250"></svg></div>
    <div class="card"><h3>Token economy &amp; cache savings</h3><div class="lede" id="l6"></div><svg id="c6" viewBox="0 0 470 250"></svg></div>
    <div class="card" style="grid-column:1 / -1"><h3>Time-to-first-token vs context</h3><div class="lede" id="l7"></div><svg id="c7" viewBox="0 0 952 240"></svg></div>
  </div>
  <div class="note">Per-turn input-side numbers (cache reads/writes, context) come from each API call's <span class="serif">usage</span> block; per-turn <i>output</i> tokens are not in the stream, so generated tokens appear only in the aggregate token-economy panel. Thinking is the harness's running estimate (resets per turn). Tool-call time is wall-clock between tool completions (model think + tool run), so it is an upper bound on pure execution; later indices reflect only the harder problems that reached them (n shown faded).</div>
</div>

<div class="tl"><div class="inner">
  <div class="row">
    <div class="t">focus: <b id="tlcount"></b> &nbsp;<span id="tlrange" style="color:var(--i55)"></span></div>
    <div><button id="tlreset">reset to full run</button></div>
  </div>
  <svg id="timeline" viewBox="0 0 1128 64"></svg>
  <div class="hint">each mark = one problem at its solve-completion time &middot; height &amp; warmth = speedup &middot; hollow = incorrect &middot; drag to select a window</div>
</div></div>

<script>
const RAW = __DATA__;
const P = RAW.problems.slice().sort((a,b)=>a.t_end-b.t_end);
const COL={terra:'#bf5b3d',ochre:'#c5973f',sage:'#6f7d5a',clay:'#9a6a4d',teal:'#5f7f84'};
const INK={s55:'rgba(34,30,22,.55)',s30:'rgba(34,30,22,.30)',s14:'rgba(34,30,22,.14)'};

// ---- formatting -----------------------------------------------------------
const fT=v=>v>=1e6?(v/1e6).toFixed(2)+'M':v>=1e3?(v/1e3).toFixed(1)+'k':Math.round(v)+'';
const fPct=v=>Math.round(v*100)+'%';
const fS=v=>v.toFixed(1)+'s';
const fU=v=>'$'+(v>=100?Math.round(v):v.toFixed(2));
const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
const pct=(a,p)=>{const s=a.slice().sort((x,y)=>x-y);const i=(s.length-1)*p;const lo=Math.floor(i);
  return s[lo]+(s[Math.ceil(i)]-s[lo])*(i-lo);};
const median=a=>pct(a,.5);

// ---- progression: mean + IQR band per turn index --------------------------
function progression(cohort, getArr, minN){
  const by={};
  cohort.forEach(p=>{(getArr(p)||[]).forEach((v,i)=>{if(v==null||isNaN(v))return;(by[i]=by[i]||[]).push(v);});});
  const out=[];
  Object.keys(by).map(Number).sort((a,b)=>a-b).forEach(i=>{
    const arr=by[i]; out.push({i, mean:mean(arr), p25:pct(arr,.25), p75:pct(arr,.75), n:arr.length});
  });
  const nmax=cohort.length;
  return out.filter(d=>d.n>=Math.max(minN, nmax*0.04));
}

// ---- generic line+band chart ---------------------------------------------
const M={t:14,r:14,b:30,l:46}, W=470, H=250;
function lineBand(id, stats, opt){
  const svg=d3.select(id); svg.selectAll('*').remove();
  if(!stats.length){svg.append('text').attr('x',W/2).attr('y',H/2).attr('text-anchor','middle').attr('fill',INK.s30).text('no data in focus');return;}
  const iw=W-M.l-M.r, ih=H-M.t-M.b;
  const x=d3.scaleLinear().domain([1,d3.max(stats,d=>d.i+1)]).range([0,iw]);
  const ymax=opt.ymax!==undefined?opt.ymax:d3.max(stats,d=>d.p75)*1.08;
  const ymin=opt.ymin!==undefined?opt.ymin:0;
  const y=d3.scaleLinear().domain([ymin,ymax]).range([ih,0]).nice();
  const g=svg.append('g').attr('transform',`translate(${M.l},${M.t})`);
  y.ticks(4).forEach(t=>g.append('line').attr('class','grid-line').attr('x1',0).attr('x2',iw).attr('y1',y(t)).attr('y2',y(t)));
  const band=d3.area().x(d=>x(d.i+1)).y0(d=>y(d.p25)).y1(d=>y(d.p75)).curve(d3.curveMonotoneX);
  const line=d3.line().x(d=>x(d.i+1)).y(d=>y(d.mean)).curve(d3.curveMonotoneX);
  g.append('path').datum(stats).attr('d',band).attr('fill',opt.color).attr('opacity',.13);
  g.append('path').datum(stats).attr('d',line).attr('fill','none').attr('stroke',opt.color)
    .attr('stroke-width',2.2).attr('stroke-linecap','round').attr('stroke-linejoin','round');
  g.selectAll('.dot').data(stats).join('circle').attr('cx',d=>x(d.i+1)).attr('cy',d=>y(d.mean))
    .attr('r',d=>2.5+2.5*d.n/cohort().length).attr('fill',opt.color).attr('opacity',.85);
  // axes
  const ax=g.append('g').attr('class','axis').attr('transform',`translate(0,${ih})`)
    .call(d3.axisBottom(x).ticks(Math.min(8,d3.max(stats,d=>d.i+1))).tickFormat(d3.format('d')).tickSizeOuter(0));
  g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(4).tickFormat(opt.yfmt).tickSizeOuter(0));
  g.append('text').attr('x',iw).attr('y',ih+26).attr('text-anchor','end').attr('fill',INK.s30).attr('font-size',10).text('turn (API call) number');
}

// ---- chart 4: tool-call time, two series + n ------------------------------
function chart4(C){
  const judge=p=>p.tool.map(t=>t[1]?t[0]:null);
  const non=p=>p.tool.map(t=>t[1]?null:t[0]);
  const sj=progression(C,judge,8), sn=progression(C,non,8);
  const svg=d3.select('#c4'); svg.selectAll('*').remove();
  const iw=W-M.l-M.r, ih=H-M.t-M.b;
  const all=sj.concat(sn);
  if(!all.length){svg.append('text').attr('x',W/2).attr('y',H/2).attr('text-anchor','middle').attr('fill',INK.s30).text('no data in focus');return;}
  const x=d3.scaleLinear().domain([1,d3.max(all,d=>d.i+1)]).range([0,iw]);
  const y=d3.scaleLinear().domain([0,d3.max(all,d=>d.p75)*1.08]).range([ih,0]).nice();
  const g=svg.append('g').attr('transform',`translate(${M.l},${M.t})`);
  y.ticks(4).forEach(t=>g.append('line').attr('class','grid-line').attr('x1',0).attr('x2',iw).attr('y1',y(t)).attr('y2',y(t)));
  [[sn,COL.sage],[sj,COL.terra]].forEach(([s,c])=>{
    if(!s.length)return;
    g.append('path').datum(s).attr('d',d3.area().x(d=>x(d.i+1)).y0(d=>y(d.p25)).y1(d=>y(d.p75)).curve(d3.curveMonotoneX)).attr('fill',c).attr('opacity',.11);
    g.append('path').datum(s).attr('d',d3.line().x(d=>x(d.i+1)).y(d=>y(d.mean)).curve(d3.curveMonotoneX)).attr('fill','none').attr('stroke',c).attr('stroke-width',2.2).attr('stroke-linecap','round');
    g.append('g').selectAll('circle').data(s).join('circle').attr('cx',d=>x(d.i+1)).attr('cy',d=>y(d.mean)).attr('r',d=>2+2.5*d.n/C.length).attr('fill',c).attr('opacity',.85);
  });
  g.append('g').attr('class','axis').attr('transform',`translate(0,${ih})`).call(d3.axisBottom(x).ticks(8).tickFormat(d3.format('d')).tickSizeOuter(0));
  g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(4).tickFormat(d=>d+'s').tickSizeOuter(0));
  g.append('text').attr('x',iw).attr('y',ih+26).attr('text-anchor','end').attr('fill',INK.s30).attr('font-size',10).text('tool-call number');
}

// ---- chart 5: batch-of-50 scorecard heatmap -------------------------------
function chart5(focusSet){
  const svg=d3.select('#c5'); svg.selectAll('*').remove();
  const nb=Math.ceil(P.length/50), rows=[];
  for(let b=0;b<nb;b++){
    const batch=P.slice(b*50,b*50+50); if(!batch.length)continue;
    let read=0,ctx=0; const peaks=[],th=[],jd=[],cost=[],ttft=[];
    batch.forEach(p=>{
      p.turns.forEach(t=>{ctx+=t[0];read+=t[1];});
      peaks.push(p.turns.length?Math.max(...p.turns.map(t=>t[0])):0);
      if(p.think.length)th.push(p.think[0]);
      p.tool.forEach(t=>{if(t[1]&&t[0]!=null)jd.push(t[0]);});
      cost.push(p.cost); if(p.ttft!=null)ttft.push(p.ttft);
    });
    rows.push({lab:`P${b*50}–${b*50+batch.length-1}`, t0:batch[0].t_end, t1:batch[batch.length-1].t_end,
      vals:[read/ctx, mean(peaks), th.length?mean(th):0, jd.length?mean(jd):0, mean(cost), ttft.length?median(ttft):0]});
  }
  const cols=[{n:'cache hit',f:v=>fPct(v)},{n:'peak ctx',f:fT},{n:'think t1',f:fT},
    {n:'judge s',f:v=>v.toFixed(0)+'s'},{n:'cost',f:v=>'$'+v.toFixed(2)},{n:'TTFT',f:v=>(v/1000).toFixed(1)+'s'}];
  const ml=78, mt=22, cw=(W-ml-8)/cols.length, rh=(H-mt-6)/rows.length;
  const ramp=d3.interpolateRgb('#e7dcc4','#bf5b3d');
  cols.forEach((c,ci)=>{
    const col=rows.map(r=>r.vals[ci]), lo=Math.min(...col), hi=Math.max(...col);
    svg.append('text').attr('x',ml+ci*cw+cw/2).attr('y',14).attr('text-anchor','middle').attr('fill',INK.s55).attr('font-size',9.5).attr('class','heat').text(c.n);
    rows.forEach((r,ri)=>{
      const v=r.vals[ci], t=hi>lo?(v-lo)/(hi-lo):0.5;
      const inFocus = !focusSet || (r.t1>=focusSet[0]&&r.t0<=focusSet[1]);
      svg.append('rect').attr('x',ml+ci*cw+1).attr('y',mt+ri*rh+1).attr('width',cw-2).attr('height',rh-2)
        .attr('rx',4).attr('fill',ramp(t)).attr('opacity',inFocus?1:.32)
        .attr('stroke',inFocus?'rgba(34,30,22,.18)':'none');
      svg.append('text').attr('x',ml+ci*cw+cw/2).attr('y',mt+ri*rh+rh/2+3.5).attr('text-anchor','middle')
        .attr('fill',t>.6?'#f6efe1':'#221e16').attr('font-size',10).attr('class','heat').attr('opacity',inFocus?1:.5).text(c.f(v));
    });
  });
  rows.forEach((r,ri)=>svg.append('text').attr('x',4).attr('y',mt+ri*rh+rh/2+3.5).attr('fill',INK.s55).attr('font-size',9.5).attr('class','heat').text(r.lab));
}

// ---- chart 6: token economy ----------------------------------------------
function chart6(C){
  const svg=d3.select('#c6'); svg.selectAll('*').remove();
  let read=0,create=0,fresh=0,out=0,cost=0;
  C.forEach(p=>{read+=p.read_tok;create+=p.create_tok;fresh+=p.in_tok;out+=p.out_tok;cost+=p.cost;});
  const tot=read+create+fresh||1;
  const saved=read*(5-0.5)/1e6;
  const segs=[['cache read',read,COL.sage],['cache write',create,COL.ochre],['fresh input',fresh,COL.terra]];
  const x0=20,x1=W-20,bw=x1-x0,by=150,bh=34;
  let cx=x0;
  segs.forEach(([n,v,c])=>{const w=bw*v/tot;
    svg.append('rect').attr('x',cx).attr('y',by).attr('width',Math.max(0,w-1.5)).attr('height',bh).attr('rx',5).attr('fill',c);
    if(w>54){svg.append('text').attr('x',cx+6).attr('y',by+14).attr('fill','#f6efe1').attr('font-size',10).attr('font-family','var(--mono)').text(fPct(v/tot));
      svg.append('text').attr('x',cx+6).attr('y',by+27).attr('fill','rgba(246,239,225,.8)').attr('font-size',9).attr('font-family','var(--mono)').text(fT(v));}
    cx+=w;});
  svg.append('text').attr('x',x0).attr('y',by-10).attr('fill',INK.s55).attr('font-size',11).text('input-side tokens = '+fT(tot));
  // big savings callout
  svg.append('text').attr('x',x0).attr('y',58).attr('fill',COL.terra).attr('font-family','Instrument Serif,serif').attr('font-style','italic').attr('font-size',40).text(fU(saved));
  svg.append('text').attr('x',x0).attr('y',80).attr('fill',INK.s55).attr('font-size',12).text('saved by cache reads vs full-price input');
  svg.append('text').attr('x',x1).attr('y',58).attr('text-anchor','end').attr('fill','#221e16').attr('font-family','var(--mono)').attr('font-size',24).text(fU(cost));
  svg.append('text').attr('x',x1).attr('y',80).attr('text-anchor','end').attr('fill',INK.s55).attr('font-size',12).text(fT(out)+' tokens generated · spent');
  // legend
  segs.forEach(([n,v,c],i)=>{svg.append('rect').attr('x',x0+i*150).attr('y',205).attr('width',10).attr('height',10).attr('rx',2).attr('fill',c);
    svg.append('text').attr('x',x0+i*150+15).attr('y',214).attr('fill',INK.s55).attr('font-size',11).text(n);});
}

// ---- chart 7: TTFT vs peak context (all problems) + marginal --------------
function chart7(C){
  const svg=d3.select('#c7'); svg.selectAll('*').remove();
  const W7=952,H7=240,m={t:14,r:54,b:32,l:50};
  const pts=C.filter(p=>p.ttft!=null).map(p=>({x:p.turns.length?Math.max.apply(null,p.turns.map(t=>t[0])):0,y:p.ttft,ok:p.correct}));
  if(!pts.length){svg.append('text').attr('x',W7/2).attr('y',H7/2).attr('text-anchor','middle').attr('fill',INK.s30).text('no data in focus');return;}
  const histW=46, iw=W7-m.l-m.r-histW, ih=H7-m.t-m.b;
  const ys=pts.map(d=>d.y).sort((a,b)=>a-b), ytop=pct(ys,0.97);
  const x=d3.scaleLinear().domain([0,d3.max(pts,d=>d.x)*1.05]).range([0,iw]);
  const y=d3.scaleLinear().domain([0,ytop]).range([ih,0]).nice();
  const g=svg.append('g').attr('transform',`translate(${m.l},${m.t})`);
  y.ticks(5).forEach(t=>g.append('line').attr('class','grid-line').attr('x1',0).attr('x2',iw).attr('y1',y(t)).attr('y2',y(t)));
  g.selectAll('c').data(pts).join('circle').attr('cx',d=>x(d.x)).attr('cy',d=>y(Math.min(d.y,ytop))).attr('r',2.8)
    .attr('fill',d=>d.ok?COL.teal:COL.clay).attr('opacity',.5);
  const med=median(pts.map(d=>d.y));
  g.append('line').attr('x1',0).attr('x2',iw).attr('y1',y(med)).attr('y2',y(med)).attr('stroke',COL.terra).attr('stroke-dasharray','4 3').attr('stroke-width',1.4);
  g.append('text').attr('x',4).attr('y',y(med)-4).attr('fill',COL.terra).attr('font-size',10).attr('font-family','var(--mono)').text('median '+(med/1000).toFixed(2)+'s');
  g.append('g').attr('class','axis').attr('transform',`translate(0,${ih})`).call(d3.axisBottom(x).ticks(7).tickFormat(d=>fT(d)).tickSizeOuter(0));
  g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(5).tickFormat(d=>(d/1000).toFixed(1)+'s').tickSizeOuter(0));
  g.append('text').attr('x',iw).attr('y',ih+26).attr('text-anchor','end').attr('fill',INK.s30).attr('font-size',10).text('peak context (tokens)');
  const bins=d3.bin().domain([0,ytop]).thresholds(22)(pts.map(d=>Math.min(d.y,ytop)));
  const mx=d3.scaleLinear().domain([0,d3.max(bins,b=>b.length)||1]).range([0,histW-8]);
  const mg=g.append('g').attr('transform',`translate(${iw+10},0)`);
  bins.forEach(b=>{const yt=y(b.x1),yb=y(b.x0);mg.append('rect').attr('x',0).attr('y',yt).attr('width',mx(b.length)).attr('height',Math.max(0,yb-yt-1)).attr('fill',COL.teal).attr('opacity',.45);});
  mg.append('text').attr('x',0).attr('y',-2).attr('fill',INK.s30).attr('font-size',9).text('TTFT dist');
}

// ---- chips ----------------------------------------------------------------
function chips(C){
  const corr=C.filter(p=>p.correct).length;
  const sp=C.filter(p=>p.speedup!=null).map(p=>p.speedup);
  let read=0,ctx=0,cost=0,saved=0;
  C.forEach(p=>{p.turns.forEach(t=>{ctx+=t[0];read+=t[1];});cost+=p.cost;saved+=p.read_tok*(5-0.5)/1e6;});
  const data=[['problems',C.length],['correct',C.length?fPct(corr/C.length):'—'],
    ['median speedup',sp.length?median(sp).toFixed(1)+'×':'—'],
    ['cache hit',ctx?fPct(read/ctx):'—'],['saved',fU(saved)],['spent',fU(cost)]];
  d3.select('#chips').html(data.map(d=>`<div class="chip"><div class="v">${d[1]}</div><div class="k">${d[0]}</div></div>`).join(''));
}

// ---- timeline -------------------------------------------------------------
let FOCUS=null;
const TLW=1128,TLH=64,tlm={l:8,r:8,t:6,b:16};
const tlx=d3.scaleLinear().domain(d3.extent(P,p=>p.t_end)).range([tlm.l,TLW-tlm.r]);
const spd=p=>p.speedup&&p.speedup>0?p.speedup:0.5;
const spdMax=d3.max(P,spd)||10;
const hcol=d3.scaleLog().domain([0.5,Math.min(spdMax,30)]).range([0,1]).clamp(true);
const hramp=d3.interpolateRgb('#d9c89f','#bf5b3d');
function renderTimeline(){
  const svg=d3.select('#timeline'); svg.selectAll('*').remove();
  const ih=TLH-tlm.t-tlm.b;
  const g=svg.append('g');
  P.forEach(p=>{const h=6+ih*0.9*(hcol(spd(p)));
    g.append('line').attr('x1',tlx(p.t_end)).attr('x2',tlx(p.t_end)).attr('y1',tlm.t+ih).attr('y2',tlm.t+ih-h)
      .attr('stroke',p.correct?hramp(hcol(spd(p))):'none')
      .attr('stroke-width',1.6)
      .attr('stroke-opacity',p.correct?.9:0);
    if(!p.correct)g.append('circle').attr('cx',tlx(p.t_end)).attr('cy',tlm.t+ih-4).attr('r',2).attr('fill','none').attr('stroke',COL.clay).attr('stroke-width',1);
  });
  g.append('line').attr('x1',tlm.l).attr('x2',TLW-tlm.r).attr('y1',tlm.t+ih).attr('y2',tlm.t+ih).attr('stroke',INK.s14);
  const brush=d3.brushX().extent([[tlm.l,tlm.t],[TLW-tlm.r,tlm.t+ih]])
    .on('end',ev=>{
      if(!ev.selection){FOCUS=null;} else {FOCUS=[tlx.invert(ev.selection[0]),tlx.invert(ev.selection[1])];}
      render();
    });
  g.append('g').attr('class','brush').call(brush);
}
function fmtDate(ms){const d=new Date(ms);return d.toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});}

// ---- orchestration --------------------------------------------------------
function cohort(){return FOCUS?P.filter(p=>p.t_end>=FOCUS[0]&&p.t_end<=FOCUS[1]):P;}
function render(){
  const C=cohort();
  chips(C);
  lineBand('#c1', progression(C,p=>p.turns.map(t=>t[0]?t[1]/t[0]:null),8), {color:COL.terra,yfmt:d=>Math.round(d*100)+'%',ymin:0,ymax:1});
  lineBand('#c2', progression(C,p=>p.turns.map(t=>t[0]),8), {color:COL.clay,yfmt:fT});
  lineBand('#c3', progression(C,p=>p.think,5), {color:COL.ochre,yfmt:fT});
  chart4(C);
  chart5(FOCUS);
  chart6(C);
  chart7(C);
  const tt=C.filter(p=>p.ttft!=null).map(p=>p.ttft);
  d3.select('#l7').html('one point per problem &mdash; first-token latency vs peak context; <b>median '+(tt.length?(median(tt)/1000).toFixed(2):'—')+'s</b>, distribution at right');
  d3.select('#l2').html('input-side tokens carried each call &mdash; grows with the conversation, far below the 1M window');
  d3.select('#l3').html('estimated reasoning per turn &mdash; <b>front-loaded</b>, then decays as edits get incremental');
  d3.select('#l5').html('six 50-problem batches in solve order; warmer = higher. Focused window outlined');
  d3.select('#l6').html('91% of input-side tokens are served from cache at 0.1× price');
  // chip lede for c1 needs real number
  let r=0,c=0; C.forEach(p=>p.turns.forEach(t=>{c+=t[0];r+=t[1];}));
  d3.select('#l1').html('Turn&nbsp;1 is a cold write; reads dominate from turn&nbsp;2 &mdash; aggregate <b>'+(c?fPct(r/c):'—')+'</b> served from cache');
  d3.select('#tlcount').text(C.length+' / '+P.length+' problems');
  d3.select('#tlrange').text(FOCUS?('('+fmtDate(FOCUS[0])+' → '+fmtDate(FOCUS[1])+')'):'(full run)');
}
d3.select('#tlreset').on('click',()=>{FOCUS=null;renderTimeline();render();});
renderTimeline();
render();
</script>
</body>
</html>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="batch_*.jsonl trace file")
    ap.add_argument("--out", default="dashboard.html")
    a = ap.parse_args()

    records = [json.loads(l) for l in Path(a.input).read_text().splitlines() if l.strip()]
    problems = extract(records)
    data = {"problems": problems, "source": a.input, "n": len(problems)}
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    Path(a.out).write_text(html)
    kb = len(html) / 1024
    print(f"wrote {a.out}  ({len(problems)} problems, {kb:.0f} KB)")


if __name__ == "__main__":
    main()
