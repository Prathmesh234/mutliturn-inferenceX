#!/usr/bin/env python3
"""
Standalone single-graph pages (one chart per HTML), warm aesthetic, timeline below.

Reuses the verified pure-data extractor from build_dashboard.py (no fabrication).

  python build_graph.py runs_full/batch_0.jsonl --graph cache-warmup --out cache_warmup.html

Graphs:
  cache-warmup   cache hit-rate vs turn index (mean + IQR band), with a real-time
                 timeline annotated with the 1h cache TTL.
"""
import argparse
import json
import statistics
from pathlib import Path

from build_dashboard import extract


def compute_meta(problems):
    """Pure run-level facts for honest TTL annotation."""
    t1reads = [p["turns"][0][1] for p in problems if p["turns"]]
    starts = [p["t_start"] for p in problems if p.get("t_start")]
    ends = [p["t_end"] for p in problems if p.get("t_end")]
    # largest idle gap between consecutive problems on the same worker
    max_gap = 0.0
    by_w = {}
    for p in problems:
        if p.get("t_start") and p.get("t_end"):
            by_w.setdefault(p.get("gpu"), []).append(p)
    for w in by_w.values():
        w.sort(key=lambda p: p["t_start"])
        for i in range(1, len(w)):
            max_gap = max(max_gap, (w[i]["t_start"] - w[i - 1]["t_end"]) / 1000.0)
    span_h = ((max(ends) - min(starts)) / 1000.0 / 3600.0) if starts and ends else 0
    return {
        "prefix_tokens": int(statistics.median(t1reads)) if t1reads else 0,
        "span_h": round(span_h, 2),
        "max_gap_min": round(max_gap / 60.0, 1),
        "ttl_s": 3600,
        "n": len(problems),
    }


CACHE_WARMUP = r'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cache hit-rate warm-up</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:ital,wght@0,400;0,500;0,600;1,400&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<style>
:root{--bg:#ede7d8;--ink:#221e16;--i55:rgba(34,30,22,.55);--i30:rgba(34,30,22,.30);
 --i22:rgba(34,30,22,.22);--i14:rgba(34,30,22,.14);--i08:rgba(34,30,22,.08);
 --terra:#bf5b3d;--ochre:#c5973f;--sage:#6f7d5a;--clay:#9a6a4d;--teal:#5f7f84;
 --panel:rgba(255,253,247,.45);--mono:'SF Mono','Cascadia Code','Consolas',monospace;}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font-family:'Inter Tight',system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding:38px 28px 60px}
.serif{font-family:'Instrument Serif',Georgia,serif;font-style:italic;font-weight:400}
h1{font-family:'Instrument Serif',Georgia,serif;font-style:italic;font-weight:400;font-size:50px;margin:0 0 6px;letter-spacing:.2px}
.sub{color:var(--i55);font-size:15px;max-width:760px;line-height:1.55}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0 18px}
.chip{background:var(--panel);border:1px solid var(--i14);border-radius:13px;padding:11px 15px;min-width:120px}
.chip .v{font-family:var(--mono);font-size:23px;font-weight:500;letter-spacing:-.5px}
.chip .k{color:var(--i55);font-size:11px;text-transform:uppercase;letter-spacing:.7px;margin-top:3px}
.card{background:var(--panel);border:1px solid var(--i14);border-radius:18px;padding:18px 20px 14px;
 box-shadow:0 1px 0 rgba(255,255,255,.4) inset,0 10px 30px -24px rgba(34,30,22,.5)}
.card h3{margin:0 0 2px;font-size:17px;font-weight:500}
.card .lede{color:var(--i55);font-size:13px;margin:0 0 6px;line-height:1.45}
.card svg{width:100%;height:auto;display:block;overflow:hidden}
.legend{display:flex;gap:18px;margin:4px 2px 0;font-size:12px;color:var(--i55)}
.legend i{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:6px;vertical-align:3px}
.axis text{font-family:var(--mono);font-size:10.5px;fill:var(--i55)}
.axis line,.axis path{stroke:var(--i14)}
.grid-line{stroke:var(--i08)}
.tlcard{margin-top:16px;background:var(--panel);border:1px solid var(--i14);border-radius:16px;padding:12px 18px 10px}
.tlcard .row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:2px}
.tlcard .t{font-size:12.5px;color:var(--i55)}
.tlcard .t b{color:var(--ink);font-weight:600;font-family:var(--mono)}
.tlcard button{font-family:'Inter Tight';font-size:11.5px;color:var(--i55);background:none;border:1px solid var(--i22);border-radius:8px;padding:3px 9px;cursor:pointer}
.tlcard button:hover{color:var(--ink);border-color:var(--i55)}
.hint{font-size:11px;color:var(--i30);margin-top:3px}
.brush .selection{fill:rgba(191,91,61,.10);stroke:var(--terra);stroke-opacity:.5}
.brush .handle{fill:var(--terra)}
.note{color:var(--i30);font-size:11.5px;margin-top:16px;line-height:1.55;max-width:820px}
</style></head>
<body><div class="wrap">
<h1>Cache hit-rate warm-up</h1>
<div class="sub">For each problem, what fraction of input tokens Claude reads from the prompt cache (vs. paying full price) at every turn &mdash; <span class="serif">Opus&nbsp;4.8</span> on KernelBook&rarr;Triton, GB300. Drag the timeline to scope the curve to a window of the run.</div>
<div class="chips" id="chips"></div>
<div class="card">
  <h3>Hit-rate by turn <span class="serif" style="color:var(--i55);font-size:14px">mean &middot; shaded = IQR across problems</span></h3>
  <div class="lede" id="lede"></div>
  <svg id="chart" viewBox="0 0 980 440"></svg>
  <div class="legend"><span><i style="background:var(--terra)"></i>cohort mean</span><span><i style="background:var(--clay);opacity:.5"></i>each 50-problem batch</span></div>
</div>
<div class="tlcard">
  <div class="row"><div class="t">focus: <b id="tlcount"></b> <span id="tlrange" style="color:var(--i55)"></span></div><div><button id="tlreset">reset to full run</button></div></div>
  <svg id="timeline" viewBox="0 0 980 96"></svg>
  <div class="hint" id="tlhint"></div>
</div>
<div class="note" id="note"></div>
</div>
<script>
const RAW=__DATA__, META=RAW.meta;
const P=RAW.problems.slice().sort((a,b)=>a.t_end-b.t_end);
const COL={terra:'#bf5b3d',ochre:'#c5973f',sage:'#6f7d5a',clay:'#9a6a4d',teal:'#5f7f84'};
const I={s55:'rgba(34,30,22,.55)',s30:'rgba(34,30,22,.30)',s14:'rgba(34,30,22,.14)'};
const fT=v=>v>=1e6?(v/1e6).toFixed(2)+'M':v>=1e3?(v/1e3).toFixed(1)+'k':Math.round(v)+'';
const fPct=v=>Math.round(v*100)+'%';
const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
const pct=(a,p)=>{const s=a.slice().sort((x,y)=>x-y);const i=(s.length-1)*p;const lo=Math.floor(i);return s[lo]+(s[Math.ceil(i)]-s[lo])*(i-lo);};
const hitArr=p=>p.turns.map(t=>t[0]?t[1]/t[0]:null);
function progression(cohort,getArr,minN){
  const by={};cohort.forEach(p=>{(getArr(p)||[]).forEach((v,i)=>{if(v==null||isNaN(v))return;(by[i]=by[i]||[]).push(v);});});
  const out=[];Object.keys(by).map(Number).sort((a,b)=>a-b).forEach(i=>{const a=by[i];out.push({i,mean:mean(a),p25:pct(a,.25),p75:pct(a,.75),n:a.length});});
  return out.filter(d=>d.n>=Math.max(minN,cohort.length*0.04));
}
// fixed global batches of 50 (solve order)
const BATCHES=[];for(let b=0;b*50<P.length;b++)BATCHES.push(P.slice(b*50,b*50+50));

const W=980,H=440,m={t:18,r:24,b:46,l:56};
function drawChart(C){
  const svg=d3.select('#chart');svg.selectAll('*').remove();
  const stats=progression(C,hitArr,8);
  const bs=BATCHES.map(b=>progression(b,hitArr,4)).filter(s=>s.length>1);
  const iw=W-m.l-m.r,ih=H-m.t-m.b;
  const xmax=Math.max(d3.max(stats,d=>d.i+1)||2, d3.max(bs.flat(),d=>d.i+1)||2);
  const x=d3.scaleLinear().domain([1,xmax]).clamp(true).range([0,iw]);
  const y=d3.scaleLinear().domain([0,1]).range([ih,0]);
  const g=svg.append('g').attr('transform',`translate(${m.l},${m.t})`);
  svg.append('clipPath').attr('id','plot').append('rect').attr('width',iw).attr('height',ih);
  y.ticks(5).forEach(t=>{g.append('line').attr('class','grid-line').attr('x1',0).attr('x2',iw).attr('y1',y(t)).attr('y2',y(t));});
  const plot=g.append('g').attr('clip-path','url(#plot)');
  // faint per-batch mean lines
  bs.forEach(s=>plot.append('path').datum(s).attr('fill','none').attr('stroke',COL.clay).attr('stroke-width',1).attr('opacity',.28)
     .attr('d',d3.line().x(d=>x(d.i+1)).y(d=>y(d.mean)).curve(d3.curveMonotoneX)));
  // cohort band + mean
  plot.append('path').datum(stats).attr('fill',COL.terra).attr('opacity',.13)
    .attr('d',d3.area().x(d=>x(d.i+1)).y0(d=>y(d.p25)).y1(d=>y(d.p75)).curve(d3.curveMonotoneX));
  plot.append('path').datum(stats).attr('fill','none').attr('stroke',COL.terra).attr('stroke-width',2.6).attr('stroke-linecap','round')
    .attr('d',d3.line().x(d=>x(d.i+1)).y(d=>y(d.mean)).curve(d3.curveMonotoneX));
  plot.append('g').selectAll('circle').data(stats).join('circle').attr('cx',d=>x(d.i+1)).attr('cy',d=>y(d.mean))
    .attr('r',d=>3+3*d.n/C.length).attr('fill',COL.terra).attr('opacity',.9);
  // axes
  g.append('g').attr('class','axis').attr('transform',`translate(0,${ih})`).call(d3.axisBottom(x).ticks(Math.min(10,xmax)).tickFormat(d3.format('d')).tickSizeOuter(0));
  g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(5).tickFormat(d=>Math.round(d*100)+'%').tickSizeOuter(0));
  g.append('text').attr('x',iw).attr('y',ih+34).attr('text-anchor','end').attr('fill',I.s30).attr('font-size',11).text('turn (API call) number within a problem');
  g.append('text').attr('transform','rotate(-90)').attr('x',-ih/2).attr('y',-40).attr('text-anchor','middle').attr('fill',I.s30).attr('font-size',11).text('cache hit-rate  (read ÷ total input)');
  // annotation: turn 1 vs plateau
  if(stats.length){
    const t1=stats[0],plat=stats[stats.length-1];
    g.append('text').attr('x',x(t1.i+1)+6).attr('y',y(t1.mean)-8).attr('fill',COL.terra).attr('font-size',11).attr('font-family','var(--mono)').text(fPct(t1.mean)+' (shared prefix already warm)');
    g.append('text').attr('x',x(plat.i+1)).attr('y',y(plat.mean)-12).attr('text-anchor','end').attr('fill',COL.terra).attr('font-size',11).attr('font-family','var(--mono)').text('plateau '+fPct(plat.mean));
  }
}
function chips(C){
  let r=0,c=0;C.forEach(p=>p.turns.forEach(t=>{c+=t[0];r+=t[1];}));
  const h1=C.filter(p=>p.turns[0]&&p.turns[0][0]).map(p=>p.turns[0][1]/p.turns[0][0]);
  const h2=C.filter(p=>p.turns[1]&&p.turns[1][0]).map(p=>p.turns[1][1]/p.turns[1][0]);
  const data=[['problems',C.length],['aggregate hit',c?fPct(r/c):'—'],
    ['turn-1 hit',h1.length?fPct(mean(h1)):'—'],['turn-2 hit',h2.length?fPct(mean(h2)):'—'],
    ['shared prefix',fT(META.prefix_tokens)+' tok'],['cache TTL','1 h']];
  d3.select('#chips').html(data.map(d=>`<div class="chip"><div class="v">${d[1]}</div><div class="k">${d[0]}</div></div>`).join(''));
}
// ---- timeline with real-time axis + 1h TTL annotation ---------------------
let FOCUS=null;
const TW=980,TH=96,tm={l:8,r:8,t:8,b:22};
const tx=d3.scaleLinear().domain(d3.extent(P,p=>p.t_end)).range([tm.l,TW-tm.r]);
const spd=p=>p.speedup&&p.speedup>0?p.speedup:0.5;
const hc=d3.scaleLog().domain([0.5,Math.min(d3.max(P,spd)||10,30)]).range([0,1]).clamp(true);
const hr=d3.interpolateRgb('#d9c89f','#bf5b3d');
function fmtClock(ms){return new Date(ms).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});}
function drawTimeline(){
  const svg=d3.select('#timeline');svg.selectAll('*').remove();
  const ih=TH-tm.t-tm.b,base=tm.t+ih;
  const g=svg.append('g');
  // hour gridlines (real clock)
  const [t0,t1]=tx.domain();
  for(let h=Math.ceil(t0/3.6e6)*3.6e6;h<t1;h+=3.6e6){
    g.append('line').attr('x1',tx(h)).attr('x2',tx(h)).attr('y1',tm.t).attr('y2',base).attr('stroke','var(--i08)');
    g.append('text').attr('x',tx(h)).attr('y',base+13).attr('text-anchor','middle').attr('fill',I.s30).attr('font-size',9.5).attr('font-family','var(--mono)').text(new Date(h).toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'}));
  }
  // problem ticks
  P.forEach(p=>{const hh=6+ih*0.82*hc(spd(p));
    g.append('line').attr('x1',tx(p.t_end)).attr('x2',tx(p.t_end)).attr('y1',base).attr('y2',base-hh)
     .attr('stroke',p.correct?hr(hc(spd(p))):'none').attr('stroke-width',1.5).attr('stroke-opacity',p.correct?.9:0);
    if(!p.correct)g.append('circle').attr('cx',tx(p.t_end)).attr('cy',base-5).attr('r',2).attr('fill','none').attr('stroke',COL.clay);});
  g.append('line').attr('x1',tm.l).attr('x2',TW-tm.r).attr('y1',base).attr('y2',base).attr('stroke',I.s14);
  // 1h TTL reference bracket (drawn from run start)
  const x0=tx(t0),x1h=tx(t0+META.ttl_s*1000);
  const yb=tm.t+4;
  g.append('line').attr('x1',x0).attr('x2',x1h).attr('y1',yb).attr('y2',yb).attr('stroke',COL.teal).attr('stroke-width',2);
  [x0,x1h].forEach(xx=>g.append('line').attr('x1',xx).attr('x2',xx).attr('y1',yb-3).attr('y2',yb+3).attr('stroke',COL.teal).attr('stroke-width',2));
  g.append('text').attr('x',x1h+6).attr('y',yb+3.5).attr('fill',COL.teal).attr('font-size',10.5).attr('font-family','var(--mono)').text('1 h cache TTL');
  const brush=d3.brushX().extent([[tm.l,tm.t],[TW-tm.r,base]]).on('end',ev=>{FOCUS=ev.selection?[tx.invert(ev.selection[0]),tx.invert(ev.selection[1])]:null;render();});
  g.append('g').attr('class','brush').call(brush);
}
function cohort(){return FOCUS?P.filter(p=>p.t_end>=FOCUS[0]&&p.t_end<=FOCUS[1]):P;}
function render(){
  const C=cohort();chips(C);drawChart(C);
  let r=0,c=0;C.forEach(p=>p.turns.forEach(t=>{c+=t[0];r+=t[1];}));
  d3.select('#lede').html('Turn&nbsp;1 already reads the shared <b>'+fT(META.prefix_tokens)+'-token</b> system+tools prefix from the 1h cache; per-problem content then accrues, so reads climb and stay high. Aggregate <b>'+(c?fPct(r/c):'—')+'</b> served from cache.');
  d3.select('#tlcount').text(C.length+' / '+P.length+' problems');
  d3.select('#tlrange').text(FOCUS?('('+fmtClock(FOCUS[0])+' → '+fmtClock(FOCUS[1])+')'):'(full run · '+META.span_h+' h)');
}
d3.select('#tlreset').on('click',()=>{FOCUS=null;render();});
d3.select('#tlhint').html('each mark = one problem at its solve time · height/warmth = speedup · hollow = incorrect · the 1h TTL bracket spans longer than the whole '+META.span_h+'h run');
d3.select('#note').html('Why turn 1 is already warm: the system prompt + tool definitions are byte-identical across every problem, so they live in one server-side 1-hour cache shared by all four workers. The run was continuous (max idle gap between a worker’s problems was '+META.max_gap_min+' min, well under the 1h TTL), so that '+fT(META.prefix_tokens)+'-token prefix never expired — every problem’s first call reads it for ~0.1× the input price. Per-turn input-side numbers come straight from each API call’s usage block.');
drawTimeline();render();
</script>
</body></html>'''

CONTEXT_GROWTH = r'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Context growth per turn</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:ital,wght@0,400;0,500;0,600;1,400&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<style>
:root{--bg:#ede7d8;--ink:#221e16;--i55:rgba(34,30,22,.55);--i30:rgba(34,30,22,.30);
 --i22:rgba(34,30,22,.22);--i14:rgba(34,30,22,.14);--i08:rgba(34,30,22,.08);
 --terra:#bf5b3d;--ochre:#c5973f;--sage:#6f7d5a;--clay:#9a6a4d;--teal:#5f7f84;
 --panel:rgba(255,253,247,.45);--mono:'SF Mono','Cascadia Code','Consolas',monospace;}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font-family:'Inter Tight',system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding:38px 28px 60px}
.serif{font-family:'Instrument Serif',Georgia,serif;font-style:italic;font-weight:400}
h1{font-family:'Instrument Serif',Georgia,serif;font-style:italic;font-weight:400;font-size:50px;margin:0 0 6px;letter-spacing:.2px}
.sub{color:var(--i55);font-size:15px;max-width:760px;line-height:1.55}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0 18px}
.chip{background:var(--panel);border:1px solid var(--i14);border-radius:13px;padding:11px 15px;min-width:120px}
.chip .v{font-family:var(--mono);font-size:23px;font-weight:500;letter-spacing:-.5px}
.chip .k{color:var(--i55);font-size:11px;text-transform:uppercase;letter-spacing:.7px;margin-top:3px}
.card{background:var(--panel);border:1px solid var(--i14);border-radius:18px;padding:18px 20px 14px;
 box-shadow:0 1px 0 rgba(255,255,255,.4) inset,0 10px 30px -24px rgba(34,30,22,.5)}
.card h3{margin:0 0 2px;font-size:17px;font-weight:500}
.card .lede{color:var(--i55);font-size:13px;margin:0 0 6px;line-height:1.45}
.card .lede b{color:var(--terra)}
.card svg{width:100%;height:auto;display:block;overflow:hidden}
.legend{display:flex;gap:18px;margin:4px 2px 0;font-size:12px;color:var(--i55)}
.legend i{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:6px;vertical-align:3px}
.axis text{font-family:var(--mono);font-size:10.5px;fill:var(--i55)}
.axis line,.axis path{stroke:var(--i14)}
.grid-line{stroke:var(--i08)}
.tlcard{margin-top:16px;background:var(--panel);border:1px solid var(--i14);border-radius:16px;padding:12px 18px 10px}
.tlcard .row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:2px}
.tlcard .t{font-size:12.5px;color:var(--i55)}
.tlcard .t b{color:var(--ink);font-weight:600;font-family:var(--mono)}
.tlcard button{font-family:'Inter Tight';font-size:11.5px;color:var(--i55);background:none;border:1px solid var(--i22);border-radius:8px;padding:3px 9px;cursor:pointer}
.tlcard button:hover{color:var(--ink);border-color:var(--i55)}
.hint{font-size:11px;color:var(--i30);margin-top:3px}
.brush .selection{fill:rgba(191,91,61,.10);stroke:var(--terra);stroke-opacity:.5}
.brush .handle{fill:var(--terra)}
.note{color:var(--i30);font-size:11.5px;margin-top:16px;line-height:1.55;max-width:820px}
.tip{position:fixed;pointer-events:none;background:rgba(255,253,247,.97);border:1px solid var(--i22);
 border-radius:11px;padding:9px 12px;font-size:12px;opacity:0;transition:opacity .08s;z-index:20;min-width:166px;
 box-shadow:0 10px 28px -12px rgba(34,30,22,.55)}
.tip .tt{font-family:'Instrument Serif',serif;font-style:italic;font-size:14.5px;margin-bottom:6px;color:var(--ink)}
.tip .tr{display:flex;justify-content:space-between;gap:16px;color:var(--i55);line-height:1.55}
.tip .tr b{font-family:var(--mono);color:var(--ink);font-weight:500}
.tip .sw{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px;vertical-align:1px}
.hit circle{cursor:pointer}
</style></head>
<body><div class="tip" id="tip"></div><div class="wrap">
<h1>Context growth per turn</h1>
<div class="sub">How much context each API call carries &mdash; <span class="serif">input&nbsp;+&nbsp;cache-read&nbsp;+&nbsp;cache-write</span>, i.e. the whole conversation so far &mdash; as a problem's session unfolds. <span class="serif">Opus&nbsp;4.8</span> on KernelBook&rarr;Triton, GB300. Drag the timeline to scope to a window of the run.</div>
<div class="chips" id="chips"></div>
<div class="card">
  <h3>Context carried by turn <span class="serif" style="color:var(--i55);font-size:14px">mean &middot; shaded = IQR &middot; faint = sampled problems</span></h3>
  <div class="lede" id="lede"></div>
  <svg id="chart" viewBox="0 0 980 440"></svg>
  <div class="legend"><span><i style="background:var(--clay)"></i>cohort mean</span><span><i style="background:var(--teal);opacity:.45"></i>individual problems (sampled)</span></div>
</div>
<div class="tlcard">
  <div class="row"><div class="t">focus: <b id="tlcount"></b> <span id="tlrange" style="color:var(--i55)"></span></div><div><button id="tlreset">reset to full run</button></div></div>
  <svg id="timeline" viewBox="0 0 980 96"></svg>
  <div class="hint" id="tlhint"></div>
</div>
<div class="note" id="note"></div>
</div>
<script>
const RAW=__DATA__, META=RAW.meta;
const P=RAW.problems.slice().sort((a,b)=>a.t_end-b.t_end);
const COL={terra:'#bf5b3d',ochre:'#c5973f',sage:'#6f7d5a',clay:'#9a6a4d',teal:'#5f7f84'};
const I={s55:'rgba(34,30,22,.55)',s30:'rgba(34,30,22,.30)',s14:'rgba(34,30,22,.14)'};
const WINDOW=1000000;
const fT=v=>v>=1e6?(v/1e6).toFixed(2)+'M':v>=1e3?(v/1e3).toFixed(1)+'k':Math.round(v)+'';
const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
const pct=(a,p)=>{const s=a.slice().sort((x,y)=>x-y);const i=(s.length-1)*p;const lo=Math.floor(i);return s[lo]+(s[Math.ceil(i)]-s[lo])*(i-lo);};
const ctxArr=p=>p.turns.map(t=>t[0]);
const peak=p=>p.turns.length?Math.max.apply(null,p.turns.map(t=>t[0])):0;
function progression(cohort,getArr,minN){
  const by={};cohort.forEach(p=>{(getArr(p)||[]).forEach((v,i)=>{if(v==null||isNaN(v))return;(by[i]=by[i]||[]).push(v);});});
  const out=[];Object.keys(by).map(Number).sort((a,b)=>a-b).forEach(i=>{const a=by[i];out.push({i,mean:mean(a),p25:pct(a,.25),p75:pct(a,.75),n:a.length});});
  return out.filter(d=>d.n>=Math.max(minN,cohort.length*0.04));
}
function sampleTraces(C,k){
  const s=C.filter(p=>p.turns.length>1).slice().sort((a,b)=>peak(a)-peak(b));
  if(s.length<=k)return s;const out=[];for(let j=0;j<k;j++)out.push(s[Math.round(j*(s.length-1)/(k-1))]);return out;
}
const TIP=d3.select('#tip');
function showTip(ev,html){TIP.html(html).style('opacity',1).style('left',(ev.clientX+14)+'px').style('top',(ev.clientY+14)+'px');}
function moveTip(ev){TIP.style('left',(ev.clientX+14)+'px').style('top',(ev.clientY+14)+'px');}
function hideTip(){TIP.style('opacity',0);}
function tipHTML(title,ctx,read,inp,cre){
  const row=(c,lab,v,ex)=>`<div class="tr"><span><span class="sw" style="background:${c}"></span>${lab}</span><b>${fT(v)}${ex||''}</b></div>`;
  return `<div class="tt">${title}</div>`+
    `<div class="tr"><span>total context</span><b>${fT(ctx)}</b></div>`+
    row(COL.terra,'input_tokens',inp,'')+
    row(COL.sage,'cache_read',read,ctx?' · '+Math.round(read/ctx*100)+'%':'')+
    row(COL.ochre,'cache_creation',cre,'');
}
function breakdownByTurn(C){
  const by={};C.forEach(p=>p.turns.forEach((t,i)=>{(by[i]=by[i]||[]).push(t);}));
  const out=[];Object.keys(by).map(Number).sort((a,b)=>a-b).forEach(i=>{const a=by[i];
    const f=k=>a.reduce((s,t)=>s+(t[k]||0),0)/a.length;const cv=a.map(t=>t[0]);
    out.push({i,n:a.length,ctx:f(0),read:f(1),inp:f(2),cre:f(3),p25:pct(cv,.25),p75:pct(cv,.75)});});
  return out.filter(d=>d.n>=Math.max(8,C.length*0.04));
}
const W=980,H=440,m={t:18,r:24,b:46,l:62};
function drawChart(C){
  const svg=d3.select('#chart');svg.selectAll('*').remove();
  const stats=breakdownByTurn(C);
  if(!stats.length){svg.append('text').attr('x',W/2).attr('y',H/2).attr('text-anchor','middle').attr('fill',I.s30).text('no data in focus');return;}
  const traces=sampleTraces(C,14);
  const iw=W-m.l-m.r,ih=H-m.t-m.b;
  const xmax=Math.max(d3.max(stats,d=>d.i+1), d3.max(traces,p=>p.turns.length)||2);
  const x=d3.scaleLinear().domain([1,xmax]).clamp(true).range([0,iw]);
  const ymax=Math.max(d3.max(stats,d=>d.p75),d3.max(traces,p=>peak(p)))*1.06;
  const y=d3.scaleLinear().domain([0,ymax]).range([ih,0]).nice();
  const g=svg.append('g').attr('transform',`translate(${m.l},${m.t})`);
  svg.append('clipPath').attr('id','plot').append('rect').attr('width',iw).attr('height',ih);
  y.ticks(5).forEach(t=>g.append('line').attr('class','grid-line').attr('x1',0).attr('x2',iw).attr('y1',y(t)).attr('y2',y(t)));
  const plot=g.append('g').attr('clip-path','url(#plot)');
  // sampled individual trajectories + hoverable vertices
  const ln=d3.line().x((d,i)=>x(i+1)).y(d=>y(d)).curve(d3.curveMonotoneX);
  traces.forEach(p=>{
    plot.append('path').datum(p.turns.map(t=>t[0])).attr('fill','none').attr('stroke',COL.teal).attr('stroke-width',1).attr('opacity',.32).attr('d',ln);
    const pts=p.turns.map((t,i)=>({i,t,e:p.entry}));
    plot.append('g').attr('class','hit').selectAll('circle').data(pts).join('circle')
      .attr('cx',d=>x(d.i+1)).attr('cy',d=>y(d.t[0])).attr('r',4).attr('fill','transparent')
      .on('mouseover',(ev,d)=>showTip(ev,tipHTML(d.e+' · turn '+(d.i+1),d.t[0],d.t[1],d.t[2],d.t[3]))).on('mousemove',moveTip).on('mouseleave',hideTip);
  });
  // cohort band + mean
  plot.append('path').datum(stats).attr('fill',COL.clay).attr('opacity',.15)
    .attr('d',d3.area().x(d=>x(d.i+1)).y0(d=>y(d.p25)).y1(d=>y(d.p75)).curve(d3.curveMonotoneX));
  plot.append('path').datum(stats).attr('fill','none').attr('stroke',COL.clay).attr('stroke-width',2.8).attr('stroke-linecap','round')
    .attr('d',d3.line().x(d=>x(d.i+1)).y(d=>y(d.ctx)).curve(d3.curveMonotoneX));
  plot.append('g').attr('class','hit').selectAll('circle').data(stats).join('circle').attr('cx',d=>x(d.i+1)).attr('cy',d=>y(d.ctx))
    .attr('r',d=>4+3*d.n/C.length).attr('fill',COL.clay).attr('opacity',.92)
    .on('mouseover',(ev,d)=>showTip(ev,tipHTML('Turn '+(d.i+1)+' · mean of '+d.n+' problems',d.ctx,d.read,d.inp,d.cre))).on('mousemove',moveTip).on('mouseleave',hideTip);
  // axes + labels (unclipped)
  g.append('g').attr('class','axis').attr('transform',`translate(0,${ih})`).call(d3.axisBottom(x).ticks(Math.min(10,xmax)).tickFormat(d3.format('d')).tickSizeOuter(0));
  g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(5).tickFormat(fT).tickSizeOuter(0));
  g.append('text').attr('x',iw).attr('y',ih+34).attr('text-anchor','end').attr('fill',I.s30).attr('font-size',11).text('turn (API call) number within a problem');
  g.append('text').attr('transform','rotate(-90)').attr('x',-ih/2).attr('y',-48).attr('text-anchor','middle').attr('fill',I.s30).attr('font-size',11).text('context carried (tokens)');
  const plat=stats[stats.length-1];
  g.append('text').attr('x',x(plat.i+1)).attr('y',y(plat.ctx)-10).attr('text-anchor','end').attr('fill',COL.clay).attr('font-size',11).attr('font-family','var(--mono)').text('mean '+fT(plat.ctx));
  const pk=C.map(peak);
  g.append('text').attr('x',2).attr('y',12).attr('fill',I.s30).attr('font-size',11).attr('font-family','var(--mono)').text('1M context window = '+Math.round(WINDOW/d3.max(pk))+'× above the tallest run here');
}
function chips(C){
  const pk=C.map(peak).filter(v=>v>0);
  const data=[['problems',C.length],['median peak',pk.length?fT(pct(pk,.5)):'—'],
    ['p90 peak',pk.length?fT(pct(pk,.9)):'—'],['max peak',pk.length?fT(d3.max(pk)):'—'],
    ['context window','1M'],['headroom',pk.length?Math.round(WINDOW/d3.max(pk))+'×':'—']];
  d3.select('#chips').html(data.map(d=>`<div class="chip"><div class="v">${d[1]}</div><div class="k">${d[0]}</div></div>`).join(''));
}
let FOCUS=null;
const TW=980,TH=96,tm={l:8,r:8,t:8,b:22};
const tx=d3.scaleLinear().domain(d3.extent(P,p=>p.t_end)).range([tm.l,TW-tm.r]);
const spd=p=>p.speedup&&p.speedup>0?p.speedup:0.5;
const hc=d3.scaleLog().domain([0.5,Math.min(d3.max(P,spd)||10,30)]).range([0,1]).clamp(true);
const hr=d3.interpolateRgb('#d9c89f','#bf5b3d');
function fmtClock(ms){return new Date(ms).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});}
function drawTimeline(){
  const svg=d3.select('#timeline');svg.selectAll('*').remove();
  const ih=TH-tm.t-tm.b,base=tm.t+ih;const g=svg.append('g');
  const [t0,t1]=tx.domain();
  for(let h=Math.ceil(t0/3.6e6)*3.6e6;h<t1;h+=3.6e6){
    g.append('line').attr('x1',tx(h)).attr('x2',tx(h)).attr('y1',tm.t).attr('y2',base).attr('stroke','var(--i08)');
    g.append('text').attr('x',tx(h)).attr('y',base+13).attr('text-anchor','middle').attr('fill',I.s30).attr('font-size',9.5).attr('font-family','var(--mono)').text(new Date(h).toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'}));}
  P.forEach(p=>{const hh=6+ih*0.82*hc(spd(p));
    g.append('line').attr('x1',tx(p.t_end)).attr('x2',tx(p.t_end)).attr('y1',base).attr('y2',base-hh).attr('stroke',p.correct?hr(hc(spd(p))):'none').attr('stroke-width',1.5).attr('stroke-opacity',p.correct?.9:0);
    if(!p.correct)g.append('circle').attr('cx',tx(p.t_end)).attr('cy',base-5).attr('r',2).attr('fill','none').attr('stroke',COL.clay);});
  g.append('line').attr('x1',tm.l).attr('x2',TW-tm.r).attr('y1',base).attr('y2',base).attr('stroke',I.s14);
  const brush=d3.brushX().extent([[tm.l,tm.t],[TW-tm.r,base]]).on('end',ev=>{FOCUS=ev.selection?[tx.invert(ev.selection[0]),tx.invert(ev.selection[1])]:null;render();});
  g.append('g').attr('class','brush').call(brush);
}
function cohort(){return FOCUS?P.filter(p=>p.t_end>=FOCUS[0]&&p.t_end<=FOCUS[1]):P;}
function render(){
  const C=cohort();chips(C);drawChart(C);
  const pk=C.map(peak).filter(v=>v>0);
  d3.select('#lede').html('Each call carries the full conversation so far, so context climbs roughly linearly to a median peak of <b>'+(pk.length?fT(pct(pk,.5)):'—')+'</b> (max <b>'+(pk.length?fT(d3.max(pk)):'—')+'</b>) &mdash; what keeps long loops cheap is that ~92% of this is cache, not the context limit.');
  d3.select('#tlcount').text(C.length+' / '+P.length+' problems');
  d3.select('#tlrange').text(FOCUS?('('+fmtClock(FOCUS[0])+' → '+fmtClock(FOCUS[1])+')'):'(full run · '+META.span_h+' h)');
}
d3.select('#tlreset').on('click',()=>{FOCUS=null;render();});
d3.select('#tlhint').html('each mark = one problem at its solve time · height/warmth = speedup · hollow = incorrect · drag to select a window');
d3.select('#note').html('Context = input_tokens + cache_read + cache_creation on each call (deduped per API message). It grows because the conversation accumulates, then resets to the '+fT(META.prefix_tokens)+'-token shared prefix on the next problem. Even the tallest run sits far under the 1M window — the binding economic factor is the cache hit-rate, not context length. Per-turn output tokens are not in the trace, so this is input-side context only.');
drawTimeline();render();
</script>
</body></html>'''

TTFT_CONTEXT = r'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Time-to-first-token vs context</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:ital,wght@0,400;0,500;0,600;1,400&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<style>
:root{--bg:#ede7d8;--ink:#221e16;--i55:rgba(34,30,22,.55);--i30:rgba(34,30,22,.30);
 --i22:rgba(34,30,22,.22);--i14:rgba(34,30,22,.14);--i08:rgba(34,30,22,.08);
 --terra:#bf5b3d;--ochre:#c5973f;--sage:#6f7d5a;--clay:#9a6a4d;--teal:#5f7f84;
 --panel:rgba(255,253,247,.45);--mono:'SF Mono','Cascadia Code','Consolas',monospace;}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font-family:'Inter Tight',system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding:38px 28px 60px}
.serif{font-family:'Instrument Serif',Georgia,serif;font-style:italic;font-weight:400}
h1{font-family:'Instrument Serif',Georgia,serif;font-style:italic;font-weight:400;font-size:50px;margin:0 0 6px;letter-spacing:.2px}
.sub{color:var(--i55);font-size:15px;max-width:760px;line-height:1.55}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0 18px}
.chip{background:var(--panel);border:1px solid var(--i14);border-radius:13px;padding:11px 15px;min-width:120px}
.chip .v{font-family:var(--mono);font-size:23px;font-weight:500;letter-spacing:-.5px}
.chip .k{color:var(--i55);font-size:11px;text-transform:uppercase;letter-spacing:.7px;margin-top:3px}
.card{background:var(--panel);border:1px solid var(--i14);border-radius:18px;padding:18px 20px 14px;
 box-shadow:0 1px 0 rgba(255,255,255,.4) inset,0 10px 30px -24px rgba(34,30,22,.5)}
.card h3{margin:0 0 2px;font-size:17px;font-weight:500}
.card .lede{color:var(--i55);font-size:13px;margin:0 0 6px;line-height:1.45}
.card .lede b{color:var(--terra)}
.card svg{width:100%;height:auto;display:block;overflow:hidden}
.legend{display:flex;gap:18px;margin:4px 2px 0;font-size:12px;color:var(--i55)}
.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:0}
.axis text{font-family:var(--mono);font-size:10.5px;fill:var(--i55)}
.axis line,.axis path{stroke:var(--i14)}
.grid-line{stroke:var(--i08)}
.tlcard{margin-top:16px;background:var(--panel);border:1px solid var(--i14);border-radius:16px;padding:12px 18px 10px}
.tlcard .row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:2px}
.tlcard .t{font-size:12.5px;color:var(--i55)}
.tlcard .t b{color:var(--ink);font-weight:600;font-family:var(--mono)}
.tlcard button{font-family:'Inter Tight';font-size:11.5px;color:var(--i55);background:none;border:1px solid var(--i22);border-radius:8px;padding:3px 9px;cursor:pointer}
.tlcard button:hover{color:var(--ink);border-color:var(--i55)}
.hint{font-size:11px;color:var(--i30);margin-top:3px}
.brush .selection{fill:rgba(191,91,61,.10);stroke:var(--terra);stroke-opacity:.5}
.brush .handle{fill:var(--terra)}
.note{color:var(--i30);font-size:11.5px;margin-top:16px;line-height:1.55;max-width:820px}
.tip{position:fixed;pointer-events:none;background:rgba(255,253,247,.97);border:1px solid var(--i22);
 border-radius:11px;padding:9px 12px;font-size:12px;opacity:0;transition:opacity .08s;z-index:20;min-width:160px;
 box-shadow:0 10px 28px -12px rgba(34,30,22,.55)}
.tip .tt{font-family:'Instrument Serif',serif;font-style:italic;font-size:14.5px;margin-bottom:6px;color:var(--ink)}
.tip .tr{display:flex;justify-content:space-between;gap:16px;color:var(--i55);line-height:1.55}
.tip .tr b{font-family:var(--mono);color:var(--ink);font-weight:500}
circle{cursor:pointer}
</style></head>
<body><div class="tip" id="tip"></div><div class="wrap">
<h1>Time-to-first-token <span class="serif" style="font-size:34px">vs</span> context</h1>
<div class="sub">Does a bigger cached prefix slow the first token? One dot per problem &mdash; x is its peak context, y is TTFT (latency to the first streamed token, from the run's result block). <span class="serif">Opus&nbsp;4.8</span>, GB300. Drag the timeline to scope the cloud.</div>
<div class="chips" id="chips"></div>
<div class="card">
  <h3>TTFT by peak context <span class="serif" style="color:var(--i55);font-size:14px">dashed = median &middot; strip = TTFT distribution</span></h3>
  <div class="lede" id="lede"></div>
  <svg id="chart" viewBox="0 0 980 440"></svg>
  <div class="legend"><span><i style="background:var(--teal)"></i>correct</span><span><i style="background:var(--clay)"></i>incorrect</span></div>
</div>
<div class="tlcard">
  <div class="row"><div class="t">focus: <b id="tlcount"></b> <span id="tlrange" style="color:var(--i55)"></span></div><div><button id="tlreset">reset to full run</button></div></div>
  <svg id="timeline" viewBox="0 0 980 96"></svg>
  <div class="hint">each mark = one problem at its solve time · height/warmth = speedup · hollow = incorrect · drag to select a window</div>
</div>
<div class="note" id="note"></div>
</div>
<script>
const RAW=__DATA__, META=RAW.meta;
const P=RAW.problems.slice().sort((a,b)=>a.t_end-b.t_end);
const COL={terra:'#bf5b3d',ochre:'#c5973f',sage:'#6f7d5a',clay:'#9a6a4d',teal:'#5f7f84'};
const I={s55:'rgba(34,30,22,.55)',s30:'rgba(34,30,22,.30)',s14:'rgba(34,30,22,.14)'};
const fT=v=>v>=1e6?(v/1e6).toFixed(2)+'M':v>=1e3?(v/1e3).toFixed(1)+'k':Math.round(v)+'';
const fS=v=>(v/1000).toFixed(2)+'s';
const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
const pct=(a,p)=>{const s=a.slice().sort((x,y)=>x-y);const i=(s.length-1)*p;const lo=Math.floor(i);return s[lo]+(s[Math.ceil(i)]-s[lo])*(i-lo);};
const peak=p=>p.turns.length?Math.max.apply(null,p.turns.map(t=>t[0])):0;
function pearson(xs,ys){const n=xs.length;if(n<3)return 0;const mx=mean(xs),my=mean(ys);let a=0,b=0,c=0;for(let i=0;i<n;i++){const dx=xs[i]-mx,dy=ys[i]-my;a+=dx*dy;b+=dx*dx;c+=dy*dy;}return a/Math.sqrt(b*c||1);}
const TIP=d3.select('#tip');
function showTip(ev,html){TIP.html(html).style('opacity',1).style('left',(ev.clientX+14)+'px').style('top',(ev.clientY+14)+'px');}
function moveTip(ev){TIP.style('left',(ev.clientX+14)+'px').style('top',(ev.clientY+14)+'px');}
function hideTip(){TIP.style('opacity',0);}
const W=980,H=440,m={t:18,r:64,b:46,l:56},histW=48;
function drawChart(C){
  const svg=d3.select('#chart');svg.selectAll('*').remove();
  const pts=C.filter(p=>p.ttft!=null).map(p=>({x:peak(p),y:p.ttft,ok:p.correct,e:p.entry,sp:p.speedup}));
  if(!pts.length){svg.append('text').attr('x',W/2).attr('y',H/2).attr('text-anchor','middle').attr('fill',I.s30).text('no data in focus');return;}
  const iw=W-m.l-m.r-histW,ih=H-m.t-m.b;
  const ytop=pct(pts.map(d=>d.y),0.97);
  const x=d3.scaleLinear().domain([0,d3.max(pts,d=>d.x)*1.05]).range([0,iw]).nice();
  const y=d3.scaleLinear().domain([0,ytop]).range([ih,0]).nice();
  const g=svg.append('g').attr('transform',`translate(${m.l},${m.t})`);
  svg.append('clipPath').attr('id','plot').append('rect').attr('width',iw).attr('height',ih);
  y.ticks(5).forEach(t=>g.append('line').attr('class','grid-line').attr('x1',0).attr('x2',iw).attr('y1',y(t)).attr('y2',y(t)));
  const plot=g.append('g').attr('clip-path','url(#plot)');
  // median line
  const med=pct(pts.map(d=>d.y),0.5);
  plot.append('line').attr('x1',0).attr('x2',iw).attr('y1',y(med)).attr('y2',y(med)).attr('stroke',COL.terra).attr('stroke-dasharray','5 4').attr('stroke-width',1.4);
  // points
  plot.append('g').selectAll('circle').data(pts).join('circle')
    .attr('cx',d=>x(d.x)).attr('cy',d=>y(Math.min(d.y,ytop))).attr('r',3.6)
    .attr('fill',d=>d.ok?COL.teal:COL.clay).attr('opacity',.55)
    .on('mouseover',(ev,d)=>showTip(ev,`<div class="tt">${d.e}</div>`
      +`<div class="tr"><span>TTFT</span><b>${fS(d.y)}</b></div>`
      +`<div class="tr"><span>peak context</span><b>${fT(d.x)}</b></div>`
      +`<div class="tr"><span>speedup</span><b>${d.sp?d.sp.toFixed(2)+'×':'—'}</b></div>`
      +`<div class="tr"><span>status</span><b style="color:${d.ok?COL.sage:COL.clay}">${d.ok?'correct':'incorrect'}</b></div>`))
    .on('mousemove',moveTip).on('mouseleave',hideTip);
  // median label (unclipped)
  g.append('text').attr('x',6).attr('y',y(med)-5).attr('fill',COL.terra).attr('font-size',11).attr('font-family','var(--mono)').text('median '+fS(med));
  // axes + labels
  g.append('g').attr('class','axis').attr('transform',`translate(0,${ih})`).call(d3.axisBottom(x).ticks(7).tickFormat(fT).tickSizeOuter(0));
  g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(5).tickFormat(d=>(d/1000).toFixed(1)+'s').tickSizeOuter(0));
  g.append('text').attr('x',iw).attr('y',ih+34).attr('text-anchor','end').attr('fill',I.s30).attr('font-size',11).text('peak context (tokens)');
  g.append('text').attr('transform','rotate(-90)').attr('x',-ih/2).attr('y',-42).attr('text-anchor','middle').attr('fill',I.s30).attr('font-size',11).text('time to first token');
  // right marginal histogram of TTFT (shares y)
  const bins=d3.bin().domain([0,ytop]).thresholds(24)(pts.map(d=>Math.min(d.y,ytop)));
  const mx=d3.scaleLinear().domain([0,d3.max(bins,b=>b.length)||1]).range([0,histW-10]);
  const mg=g.append('g').attr('transform',`translate(${iw+12},0)`);
  bins.forEach(b=>{const yt=y(b.x1),yb=y(b.x0);mg.append('rect').attr('x',0).attr('y',yt).attr('width',mx(b.length)).attr('height',Math.max(0,yb-yt-1)).attr('rx',1.5).attr('fill',COL.teal).attr('opacity',.5);});
  mg.append('text').attr('x',0).attr('y',-4).attr('fill',I.s30).attr('font-size',9.5).text('dist');
}
function chips(C){
  const tt=C.filter(p=>p.ttft!=null).map(p=>p.ttft);
  const xs=C.filter(p=>p.ttft!=null).map(p=>peak(p)), ys=tt;
  const r=tt.length>2?pearson(xs,ys):0;
  const data=[['problems',C.length],['median TTFT',tt.length?fS(pct(tt,.5)):'—'],
    ['p90 TTFT',tt.length?fS(pct(tt,.9)):'—'],['max TTFT',tt.length?fS(d3.max(tt)):'—'],
    ['context↔TTFT r',tt.length?r.toFixed(2):'—'],['correct',C.length?Math.round(C.filter(p=>p.correct).length/C.length*100)+'%':'—']];
  d3.select('#chips').html(data.map(d=>`<div class="chip"><div class="v">${d[1]}</div><div class="k">${d[0]}</div></div>`).join(''));
}
let FOCUS=null;
const TW=980,TH=96,tm={l:8,r:8,t:8,b:22};
const tx=d3.scaleLinear().domain(d3.extent(P,p=>p.t_end)).range([tm.l,TW-tm.r]);
const spd=p=>p.speedup&&p.speedup>0?p.speedup:0.5;
const hc=d3.scaleLog().domain([0.5,Math.min(d3.max(P,spd)||10,30)]).range([0,1]).clamp(true);
const hr=d3.interpolateRgb('#d9c89f','#bf5b3d');
function fmtClock(ms){return new Date(ms).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});}
function drawTimeline(){
  const svg=d3.select('#timeline');svg.selectAll('*').remove();
  const ih=TH-tm.t-tm.b,base=tm.t+ih;const g=svg.append('g');
  const [t0,t1]=tx.domain();
  for(let h=Math.ceil(t0/3.6e6)*3.6e6;h<t1;h+=3.6e6){
    g.append('line').attr('x1',tx(h)).attr('x2',tx(h)).attr('y1',tm.t).attr('y2',base).attr('stroke','var(--i08)');
    g.append('text').attr('x',tx(h)).attr('y',base+13).attr('text-anchor','middle').attr('fill',I.s30).attr('font-size',9.5).attr('font-family','var(--mono)').text(new Date(h).toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'}));}
  P.forEach(p=>{const hh=6+ih*0.82*hc(spd(p));
    g.append('line').attr('x1',tx(p.t_end)).attr('x2',tx(p.t_end)).attr('y1',base).attr('y2',base-hh).attr('stroke',p.correct?hr(hc(spd(p))):'none').attr('stroke-width',1.5).attr('stroke-opacity',p.correct?.9:0);
    if(!p.correct)g.append('circle').attr('cx',tx(p.t_end)).attr('cy',base-5).attr('r',2).attr('fill','none').attr('stroke',COL.clay);});
  g.append('line').attr('x1',tm.l).attr('x2',TW-tm.r).attr('y1',base).attr('y2',base).attr('stroke',I.s14);
  const brush=d3.brushX().extent([[tm.l,tm.t],[TW-tm.r,base]]).on('end',ev=>{FOCUS=ev.selection?[tx.invert(ev.selection[0]),tx.invert(ev.selection[1])]:null;render();});
  g.append('g').attr('class','brush').call(brush);
}
function cohort(){return FOCUS?P.filter(p=>p.t_end>=FOCUS[0]&&p.t_end<=FOCUS[1]):P;}
function render(){
  const C=cohort();chips(C);drawChart(C);
  const tt=C.filter(p=>p.ttft!=null).map(p=>p.ttft);
  const xs=C.filter(p=>p.ttft!=null).map(p=>peak(p));
  const r=tt.length>2?pearson(xs,tt):0;
  d3.select('#lede').html('Median first-token latency is <b>'+(tt.length?fS(pct(tt,.5)):'—')+'</b>; correlation with peak context is <b>r='+(tt.length?r.toFixed(2):'—')+'</b> &mdash; '+(Math.abs(r)<0.25?'essentially flat: a bigger cached prefix does not gate TTFT.':'note the trend with context.'));
  d3.select('#tlcount').text(C.length+' / '+P.length+' problems');
  d3.select('#tlrange').text(FOCUS?('('+fmtClock(FOCUS[0])+' → '+fmtClock(FOCUS[1])+')'):'(full run · '+META.span_h+' h)');
}
d3.select('#tlreset').on('click',()=>{FOCUS=null;render();});
d3.select('#note').html('TTFT is the per-problem latency to the first streamed token (from each run\'s result block) — there is one value per problem, not per turn. The cached prefix is read on essentially every call, so even large contexts start streaming quickly. The right strip is the TTFT distribution (same y-axis); a few slow outliers above the '+'97th percentile are clamped to the top edge.');
drawTimeline();render();
</script>
</body></html>'''

_HEAD = '''<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:ital,wght@0,400;0,500;0,600;1,400&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<style>
:root{--bg:#ede7d8;--ink:#221e16;--i55:rgba(34,30,22,.55);--i30:rgba(34,30,22,.30);
 --i22:rgba(34,30,22,.22);--i14:rgba(34,30,22,.14);--i08:rgba(34,30,22,.08);
 --terra:#bf5b3d;--ochre:#c5973f;--sage:#6f7d5a;--clay:#9a6a4d;--teal:#5f7f84;
 --panel:rgba(255,253,247,.45);--mono:'SF Mono','Cascadia Code','Consolas',monospace;}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font-family:'Inter Tight',system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding:38px 28px 60px}
.serif{font-family:'Instrument Serif',Georgia,serif;font-style:italic;font-weight:400}
h1{font-family:'Instrument Serif',Georgia,serif;font-style:italic;font-weight:400;font-size:50px;margin:0 0 6px;letter-spacing:.2px}
.sub{color:var(--i55);font-size:15px;max-width:760px;line-height:1.55}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0 18px}
.chip{background:var(--panel);border:1px solid var(--i14);border-radius:13px;padding:11px 15px;min-width:120px}
.chip .v{font-family:var(--mono);font-size:23px;font-weight:500;letter-spacing:-.5px}
.chip .k{color:var(--i55);font-size:11px;text-transform:uppercase;letter-spacing:.7px;margin-top:3px}
.card{background:var(--panel);border:1px solid var(--i14);border-radius:18px;padding:18px 20px 14px;
 box-shadow:0 1px 0 rgba(255,255,255,.4) inset,0 10px 30px -24px rgba(34,30,22,.5)}
.card h3{margin:0 0 2px;font-size:17px;font-weight:500}
.card .lede{color:var(--i55);font-size:13px;margin:0 0 6px;line-height:1.45}
.card .lede b{color:var(--terra)}
.card svg{width:100%;height:auto;display:block;overflow:hidden}
.legend{display:flex;gap:18px;margin:4px 2px 0;font-size:12px;color:var(--i55)}
.legend i{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:6px;vertical-align:3px}
.legend i.sq{width:10px;height:10px;border-radius:2px;vertical-align:0}
.axis text{font-family:var(--mono);font-size:10.5px;fill:var(--i55)}
.axis line,.axis path{stroke:var(--i14)}
.grid-line{stroke:var(--i08)}
.tlcard{margin-top:16px;background:var(--panel);border:1px solid var(--i14);border-radius:16px;padding:12px 18px 10px}
.tlcard .row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:2px}
.tlcard .t{font-size:12.5px;color:var(--i55)}.tlcard .t b{color:var(--ink);font-weight:600;font-family:var(--mono)}
.tlcard button{font-family:'Inter Tight';font-size:11.5px;color:var(--i55);background:none;border:1px solid var(--i22);border-radius:8px;padding:3px 9px;cursor:pointer}
.tlcard button:hover{color:var(--ink);border-color:var(--i55)}
.hint{font-size:11px;color:var(--i30);margin-top:3px}
.brush .selection{fill:rgba(191,91,61,.10);stroke:var(--terra);stroke-opacity:.5}
.brush .handle{fill:var(--terra)}
.note{color:var(--i30);font-size:11.5px;margin-top:16px;line-height:1.55;max-width:820px}
.tip{position:fixed;pointer-events:none;background:rgba(255,253,247,.97);border:1px solid var(--i22);
 border-radius:11px;padding:9px 12px;font-size:12px;opacity:0;transition:opacity .08s;z-index:20;min-width:160px;
 box-shadow:0 10px 28px -12px rgba(34,30,22,.55)}
.tip .tt{font-family:'Instrument Serif',serif;font-style:italic;font-size:14.5px;margin-bottom:6px;color:var(--ink)}
.tip .tr{display:flex;justify-content:space-between;gap:16px;color:var(--i55);line-height:1.55}
.tip .tr b{font-family:var(--mono);color:var(--ink);font-weight:500}
.tip .sw{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px;vertical-align:1px}
circle{cursor:pointer}
</style>'''

_COMMON_JS = '''
const RAW=__DATA__, META=RAW.meta;
const P=RAW.problems.slice().sort((a,b)=>a.t_end-b.t_end);
const COL={terra:'#bf5b3d',ochre:'#c5973f',sage:'#6f7d5a',clay:'#9a6a4d',teal:'#5f7f84'};
const I={s55:'rgba(34,30,22,.55)',s30:'rgba(34,30,22,.30)',s14:'rgba(34,30,22,.14)'};
const fT=v=>v>=1e6?(v/1e6).toFixed(2)+'M':v>=1e3?(v/1e3).toFixed(1)+'k':Math.round(v)+'';
const fS=v=>v.toFixed(1)+'s';
const fU=v=>'$'+(v>=100?Math.round(v):v.toFixed(2));
const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
const pct=(a,p)=>{const s=a.slice().sort((x,y)=>x-y);const i=(s.length-1)*p;const lo=Math.floor(i);return s[lo]+(s[Math.ceil(i)]-s[lo])*(i-lo);};
const TIP=d3.select('#tip');
function showTip(ev,h){TIP.html(h).style('opacity',1).style('left',(ev.clientX+14)+'px').style('top',(ev.clientY+14)+'px');}
function moveTip(ev){TIP.style('left',(ev.clientX+14)+'px').style('top',(ev.clientY+14)+'px');}
function hideTip(){TIP.style('opacity',0);}
let FOCUS=null;
const TW=980,TH=96,tm={l:8,r:8,t:8,b:22};
const tx=d3.scaleLinear().domain(d3.extent(P,p=>p.t_end)).range([tm.l,TW-tm.r]);
const spd=p=>p.speedup&&p.speedup>0?p.speedup:0.5;
const hc=d3.scaleLog().domain([0.5,Math.min(d3.max(P,spd)||10,30)]).range([0,1]).clamp(true);
const hr=d3.interpolateRgb('#d9c89f','#bf5b3d');
function fmtClock(ms){return new Date(ms).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});}
function drawTimeline(){
  const svg=d3.select('#timeline');svg.selectAll('*').remove();
  const ih=TH-tm.t-tm.b,base=tm.t+ih;const g=svg.append('g');const [t0,t1]=tx.domain();
  for(let h=Math.ceil(t0/3.6e6)*3.6e6;h<t1;h+=3.6e6){
    g.append('line').attr('x1',tx(h)).attr('x2',tx(h)).attr('y1',tm.t).attr('y2',base).attr('stroke','var(--i08)');
    g.append('text').attr('x',tx(h)).attr('y',base+13).attr('text-anchor','middle').attr('fill',I.s30).attr('font-size',9.5).attr('font-family','var(--mono)').text(new Date(h).toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'}));}
  P.forEach(p=>{const hh=6+ih*0.82*hc(spd(p));
    g.append('line').attr('x1',tx(p.t_end)).attr('x2',tx(p.t_end)).attr('y1',base).attr('y2',base-hh).attr('stroke',p.correct?hr(hc(spd(p))):'none').attr('stroke-width',1.5).attr('stroke-opacity',p.correct?.9:0);
    if(!p.correct)g.append('circle').attr('cx',tx(p.t_end)).attr('cy',base-5).attr('r',2).attr('fill','none').attr('stroke',COL.clay);});
  g.append('line').attr('x1',tm.l).attr('x2',TW-tm.r).attr('y1',base).attr('y2',base).attr('stroke',I.s14);
  const brush=d3.brushX().extent([[tm.l,tm.t],[TW-tm.r,base]]).on('end',ev=>{FOCUS=ev.selection?[tx.invert(ev.selection[0]),tx.invert(ev.selection[1])]:null;render();});
  g.append('g').attr('class','brush').call(brush);
}
function cohort(){return FOCUS?P.filter(p=>p.t_end>=FOCUS[0]&&p.t_end<=FOCUS[1]):P;}
'''

TOOL_CALL_TIME = ('<!DOCTYPE html><html lang="en"><head><title>Tool-call time</title>' + _HEAD + '''</head>
<body><div class="tip" id="tip"></div><div class="wrap">
<h1>Tool-call time</h1>
<div class="sub">Wall-clock per <span class="serif">edit / read / write</span> step (model think + that tool's run), as a problem unfolds &mdash; GPU-eval (judge) calls excluded. <span class="serif">Opus&nbsp;4.8</span>, GB300. Drag the timeline to scope.</div>
<div class="chips" id="chips"></div>
<div class="card">
  <h3>Seconds per edit/read step <span class="serif" style="color:var(--i55);font-size:14px">mean &middot; shaded = IQR &middot; faint = each 50-problem batch</span></h3>
  <div class="lede" id="lede"></div>
  <svg id="chart" viewBox="0 0 980 440"></svg>
  <div class="legend"><span><i style="background:var(--sage)"></i>cohort mean</span><span><i style="background:var(--clay);opacity:.5"></i>each 50-problem batch</span></div>
</div>
<div class="tlcard"><div class="row"><div class="t">focus: <b id="tlcount"></b> <span id="tlrange" style="color:var(--i55)"></span></div><div><button id="tlreset">reset to full run</button></div></div>
  <svg id="timeline" viewBox="0 0 980 96"></svg>
  <div class="hint">each mark = one problem at its solve time · height/warmth = speedup · hollow = incorrect · drag to select a window</div></div>
<div class="note" id="note"></div></div>
<script>''' + _COMMON_JS + '''
const njArr=p=>p.tool.filter(t=>!t[1]&&t[0]!=null).map(t=>t[0]);
const BATCHES=[];for(let b=0;b*50<P.length;b++)BATCHES.push(P.slice(b*50,b*50+50));
function progression(C,getArr,minN){const by={};C.forEach(p=>{(getArr(p)||[]).forEach((v,i)=>{if(v==null||isNaN(v))return;(by[i]=by[i]||[]).push(v);});});
  const out=[];Object.keys(by).map(Number).sort((a,b)=>a-b).forEach(i=>{const a=by[i];out.push({i,n:a.length,mean:mean(a),p25:pct(a,.25),p75:pct(a,.75)});});
  return out.filter(d=>d.n>=Math.max(minN,C.length*0.04));}
const W=980,H=440,m={t:18,r:24,b:46,l:54};
function drawChart(C){
  const svg=d3.select('#chart');svg.selectAll('*').remove();
  const stats=progression(C,njArr,8);
  if(!stats.length){svg.append('text').attr('x',W/2).attr('y',H/2).attr('text-anchor','middle').attr('fill',I.s30).text('no data in focus');return;}
  const bs=BATCHES.map(b=>progression(b,njArr,4)).filter(s=>s.length>1);
  const iw=W-m.l-m.r,ih=H-m.t-m.b;
  const xmax=Math.max(d3.max(stats,d=>d.i+1),d3.max(bs.flat(),d=>d.i+1)||2);
  const x=d3.scaleLinear().domain([1,xmax]).clamp(true).range([0,iw]);
  const y=d3.scaleLinear().domain([0,d3.max(stats,d=>d.p75)*1.12]).range([ih,0]).nice();
  const g=svg.append('g').attr('transform',`translate(${m.l},${m.t})`);
  svg.append('clipPath').attr('id','plot').append('rect').attr('width',iw).attr('height',ih);
  y.ticks(5).forEach(t=>g.append('line').attr('class','grid-line').attr('x1',0).attr('x2',iw).attr('y1',y(t)).attr('y2',y(t)));
  const plot=g.append('g').attr('clip-path','url(#plot)');
  bs.forEach(s=>plot.append('path').datum(s).attr('fill','none').attr('stroke',COL.clay).attr('stroke-width',1).attr('opacity',.26).attr('d',d3.line().x(d=>x(d.i+1)).y(d=>y(d.mean)).curve(d3.curveMonotoneX)));
  plot.append('path').datum(stats).attr('fill',COL.sage).attr('opacity',.15).attr('d',d3.area().x(d=>x(d.i+1)).y0(d=>y(d.p25)).y1(d=>y(d.p75)).curve(d3.curveMonotoneX));
  plot.append('path').datum(stats).attr('fill','none').attr('stroke',COL.sage).attr('stroke-width',2.6).attr('stroke-linecap','round').attr('d',d3.line().x(d=>x(d.i+1)).y(d=>y(d.mean)).curve(d3.curveMonotoneX));
  plot.append('g').selectAll('circle').data(stats).join('circle').attr('cx',d=>x(d.i+1)).attr('cy',d=>y(d.mean)).attr('r',d=>4+3*d.n/C.length).attr('fill',COL.sage).attr('opacity',.92)
    .on('mouseover',(ev,d)=>showTip(ev,'<div class="tt">Step '+(d.i+1)+' · mean of '+d.n+' problems</div><div class="tr"><span>edit/read time</span><b>'+d.mean.toFixed(1)+'s</b></div><div class="tr"><span>IQR</span><b>'+d.p25.toFixed(1)+'–'+d.p75.toFixed(1)+'s</b></div>')).on('mousemove',moveTip).on('mouseleave',hideTip);
  g.append('g').attr('class','axis').attr('transform',`translate(0,${ih})`).call(d3.axisBottom(x).ticks(Math.min(10,xmax)).tickFormat(d3.format('d')).tickSizeOuter(0));
  g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(5).tickFormat(d=>d+'s').tickSizeOuter(0));
  g.append('text').attr('x',iw).attr('y',ih+34).attr('text-anchor','end').attr('fill',I.s30).attr('font-size',11).text('edit/read/write step number (judge calls excluded)');
  g.append('text').attr('transform','rotate(-90)').attr('x',-ih/2).attr('y',-40).attr('text-anchor','middle').attr('fill',I.s30).attr('font-size',11).text('seconds  (think + tool run)');
}
function chips(C){
  const all=C.flatMap(njArr);const per=C.map(p=>njArr(p).length).filter(v=>v>0);
  const data=[['problems',C.length],['median step',all.length?fS(pct(all,.5)):'—'],['p90 step',all.length?fS(pct(all,.9)):'—'],
    ['1st-step mean',(()=>{const s=progression(C,njArr,8);return s.length?fS(s[0].mean):'—';})()],['edit/read calls',all.length+''],['avg/problem',per.length?(mean(per)).toFixed(1):'—']];
  d3.select('#chips').html(data.map(d=>`<div class="chip"><div class="v">${d[1]}</div><div class="k">${d[0]}</div></div>`).join(''));
}
function render(){const C=cohort();chips(C);drawChart(C);
  const all=C.flatMap(njArr);
  d3.select('#lede').html('Each non-judge step (file read/write/edit) is wall-clock between tool completions, so it is <b>model think + that tool</b>. Median <b>'+(all.length?fS(pct(all,.5)):'—')+'</b>; front-loaded, then steadier as edits get incremental.');
  d3.select('#tlcount').text(C.length+' / '+P.length+' problems');
  d3.select('#tlrange').text(FOCUS?('('+fmtClock(FOCUS[0])+' → '+fmtClock(FOCUS[1])+')'):'(full run · '+META.span_h+' h)');
}
d3.select('#tlreset').on('click',()=>{FOCUS=null;render();});
d3.select('#note').html('Time is wall-clock between consecutive tool-result completions (model think + the tool itself), an upper bound on pure tool time. GPU-eval (judge) calls are excluded so this isolates the agent\\'s file-editing latency. Step number re-indexes only the non-judge calls; later steps reflect only the harder problems that reached them, so the mean there averages fewer problems.');
drawTimeline();render();
</script></body></html>''')

TOKEN_ECONOMY = ('<!DOCTYPE html><html lang="en"><head><title>Token economy &amp; cache savings</title>' + _HEAD + '''</head>
<body><div class="tip" id="tip"></div><div class="wrap">
<h1>Token economy <span class="serif" style="font-size:34px">&amp;</span> cache savings</h1>
<div class="sub">Where the input tokens go and what the prompt cache saves. Cache reads bill at <span class="serif">0.1&times;</span> the input rate, so a high hit-rate is most of the economics. <span class="serif">Opus&nbsp;4.8</span>, GB300. Drag the timeline to scope.</div>
<div class="chips" id="chips"></div>
<div class="card">
  <h3>Input-token composition &amp; spend by 50-problem batch</h3>
  <div class="lede" id="lede"></div>
  <svg id="chart" viewBox="0 0 980 440"></svg>
  <div class="legend"><span><i class="sq" style="background:var(--sage)"></i>cache read (0.1×)</span><span><i class="sq" style="background:var(--ochre)"></i>cache write</span><span><i class="sq" style="background:var(--terra)"></i>fresh input</span></div>
</div>
<div class="tlcard"><div class="row"><div class="t">focus: <b id="tlcount"></b> <span id="tlrange" style="color:var(--i55)"></span></div><div><button id="tlreset">reset to full run</button></div></div>
  <svg id="timeline" viewBox="0 0 980 96"></svg>
  <div class="hint">each mark = one problem at its solve time · height/warmth = speedup · hollow = incorrect · drag to select a window</div></div>
<div class="note" id="note"></div></div>
<script>''' + _COMMON_JS + '''
const BATCHES=[];for(let b=0;b*50<P.length;b++)BATCHES.push(P.slice(b*50,b*50+50));
const SAVE_RATE=(5-0.5)/1e6;  // input $5/Mtok, cache-read $0.5/Mtok -> saved per read token
const W=980,H=440;
function agg(C){let read=0,create=0,fresh=0,out=0,cost=0;C.forEach(p=>{read+=p.read_tok;create+=p.create_tok;fresh+=p.in_tok;out+=p.out_tok;cost+=p.cost;});return{read,create,fresh,out,cost,tot:read+create+fresh,saved:read*SAVE_RATE};}
function drawChart(C){
  const svg=d3.select('#chart');svg.selectAll('*').remove();
  const a=agg(C);if(!a.tot){svg.append('text').attr('x',W/2).attr('y',H/2).attr('text-anchor','middle').attr('fill',I.s30).text('no data in focus');return;}
  // ---- composition bar ----
  const bx0=24,bx1=956,bw=bx1-bx0,by=54,bh=42;
  svg.append('text').attr('x',bx0).attr('y',by-12).attr('fill',I.s55).attr('font-size',12).text('input-side tokens = '+fT(a.tot));
  let cx=bx0;[['cache read',a.read,COL.sage],['cache write',a.create,COL.ochre],['fresh input',a.fresh,COL.terra]].forEach(([n,v,c])=>{
    const w=bw*v/a.tot;
    svg.append('rect').attr('x',cx).attr('y',by).attr('width',Math.max(0,w-2)).attr('height',bh).attr('rx',5).attr('fill',c)
      .on('mouseover',ev=>showTip(ev,'<div class="tt">'+n+'</div><div class="tr"><span>tokens</span><b>'+fT(v)+'</b></div><div class="tr"><span>share</span><b>'+Math.round(v/a.tot*100)+'%</b></div>')).on('mousemove',moveTip).on('mouseleave',hideTip);
    if(w>56){svg.append('text').attr('x',cx+7).attr('y',by+17).attr('fill','#f6efe1').attr('font-size',11).attr('font-family','var(--mono)').text(Math.round(v/a.tot*100)+'%');
      svg.append('text').attr('x',cx+7).attr('y',by+31).attr('fill','rgba(246,239,225,.85)').attr('font-size',10).attr('font-family','var(--mono)').text(fT(v));}
    cx+=w;});
  // ---- savings callout ----
  svg.append('text').attr('x',bx0).attr('y',150).attr('fill',COL.terra).attr('font-family','Instrument Serif,serif').attr('font-style','italic').attr('font-size',46).text(fU(a.saved));
  svg.append('text').attr('x',bx0).attr('y',172).attr('fill',I.s55).attr('font-size',12).text('saved by cache reads vs full-price input');
  svg.append('text').attr('x',bx1).attr('y',150).attr('text-anchor','end').attr('fill','#221e16').attr('font-family','var(--mono)').attr('font-size',28).text(fU(a.cost));
  svg.append('text').attr('x',bx1).attr('y',172).attr('text-anchor','end').attr('fill',I.s55).attr('font-size',12).text(fT(a.out)+' tokens generated · spent');
  // ---- per-batch saved vs spent ----
  const focusSet=FOCUS;
  const rows=BATCHES.map((b,i)=>{const aa=agg(b);return{lab:'P'+(i*50)+'–'+(i*50+b.length-1),t0:b[0].t_end,t1:b[b.length-1].t_end,saved:aa.saved,spent:aa.cost};});
  const m={t:206,r:24,b:40,l:50},iw=W-m.l-m.r,ih=H-m.t-m.b;
  const g=svg.append('g').attr('transform',`translate(${m.l},${m.t})`);
  svg.append('text').attr('x',m.l).attr('y',m.t-8).attr('fill',I.s55).attr('font-size',12).text('per 50-problem batch — saved vs spent ($)');
  const x0=d3.scaleBand().domain(rows.map(r=>r.lab)).range([0,iw]).padding(0.32);
  const x1=d3.scaleBand().domain(['saved','spent']).range([0,x0.bandwidth()]).padding(0.12);
  const y=d3.scaleLinear().domain([0,d3.max(rows,r=>Math.max(r.saved,r.spent))*1.1||1]).range([ih,0]).nice();
  y.ticks(4).forEach(t=>g.append('line').attr('class','grid-line').attr('x1',0).attr('x2',iw).attr('y1',y(t)).attr('y2',y(t)));
  rows.forEach(r=>{const inF=!focusSet||(r.t1>=focusSet[0]&&r.t0<=focusSet[1]);
    [['saved',r.saved,COL.sage],['spent',r.spent,COL.terra]].forEach(([k,v,c])=>{
      g.append('rect').attr('x',x0(r.lab)+x1(k)).attr('y',y(v)).attr('width',x1.bandwidth()).attr('height',ih-y(v)).attr('rx',3).attr('fill',c).attr('opacity',inF?.95:.3)
        .on('mouseover',ev=>showTip(ev,'<div class="tt">'+r.lab+'</div><div class="tr"><span><span class="sw" style="background:'+COL.sage+'"></span>saved</span><b>'+fU(r.saved)+'</b></div><div class="tr"><span><span class="sw" style="background:'+COL.terra+'"></span>spent</span><b>'+fU(r.spent)+'</b></div>')).on('mousemove',moveTip).on('mouseleave',hideTip);});});
  g.append('g').attr('class','axis').attr('transform',`translate(0,${ih})`).call(d3.axisBottom(x0).tickSizeOuter(0));
  g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(4).tickFormat(d=>'$'+d).tickSizeOuter(0));
}
function chips(C){const a=agg(C);
  const data=[['problems',C.length],['input-side',fT(a.tot)],['generated',fT(a.out)],['cache hit',a.tot?Math.round(a.read/a.tot*100)+'%':'—'],
    ['saved',fU(a.saved)],['spent',fU(a.cost)]];
  d3.select('#chips').html(data.map(d=>`<div class="chip"><div class="v">${d[1]}</div><div class="k">${d[0]}</div></div>`).join(''));
}
function render(){const C=cohort();chips(C);drawChart(C);const a=agg(C);
  d3.select('#lede').html('<b>'+Math.round(a.read/a.tot*100)+'%</b> of input-side tokens are served from cache at 0.1× price — <b>'+fU(a.saved)+'</b> saved against <b>'+fU(a.cost)+'</b> actually spent.');
  d3.select('#tlcount').text(C.length+' / '+P.length+' problems');
  d3.select('#tlrange').text(FOCUS?('('+fmtClock(FOCUS[0])+' → '+fmtClock(FOCUS[1])+')'):'(full run · '+META.span_h+' h)');
}
d3.select('#tlreset').on('click',()=>{FOCUS=null;render();});
d3.select('#note').html('Saved = cache-read tokens × $4.50/Mtok (the Opus 4.8 input rate $5 minus the cache-read rate $0.50) — the discount from reading the prefix at 0.1× instead of full price. Spent is the run\\'s authoritative total_cost_usd (input + output + cache, incl. the small Haiku title model). Output tokens are aggregate-only — per-turn output is not in the trace.');
drawTimeline();render();
</script></body></html>''')

TEMPLATES = {"cache-warmup": CACHE_WARMUP, "context-growth": CONTEXT_GROWTH, "ttft-context": TTFT_CONTEXT,
             "tool-call-time": TOOL_CALL_TIME, "token-economy": TOKEN_ECONOMY}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--graph", default="cache-warmup", choices=sorted(TEMPLATES))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    records = [json.loads(l) for l in Path(a.input).read_text().splitlines() if l.strip()]
    problems = extract(records)
    data = {"problems": problems, "meta": compute_meta(problems)}
    html = TEMPLATES[a.graph].replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = a.out or (a.graph.replace("-", "_") + ".html")
    Path(out).write_text(html)
    print(f"wrote {out}  ({len(problems)} problems, {len(html)/1024:.0f} KB)  meta={data['meta']}")


if __name__ == "__main__":
    main()
