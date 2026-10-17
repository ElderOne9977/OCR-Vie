#!/usr/bin/env python3
"""Minimal local annotation tool — stdlib only (no frontend framework).

Opens a browser page where annotator can:
  - navigate prev/next
  - draw/edit rectangles (stored as 4-point polygons; quadrilateral drag supported)
  - enter exact GT text (verbatim, no normalization)
  - set annotation_level (image/region), review_status (unreviewed/reviewed/rejected)
  - mark regions as ignore/illegible
  - save

Run:  python benchmarks/annotate.py --data benchmark_data --port 8765
Then open http://localhost:8765/

All writes go to annotations.jsonl (atomic replace). Images never overwritten.
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse, unquote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.dataset import load_annotations, sha256_file, VALID_SPLITS, validate_ground_truth
DEFAULT_DATA = ROOT / "benchmark_data"

HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>OCR-Vie Annotator — minimal</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;display:flex;flex-direction:column;height:100vh}
header{padding:8px 12px;background:#111;color:#fff;display:flex;gap:12px;align-items:center}
header button{padding:6px 10px}
main{display:flex;flex:1;overflow:hidden}
#left{flex:1;display:flex;flex-direction:column;overflow:auto;padding:8px}
#right{width:420px;border-left:1px solid #ddd;padding:8px;overflow:auto}
canvas{border:1px solid #999;cursor:crosshair;max-width:100%}
label{display:block;margin:6px 0 2px;font-weight:600}
input,select,textarea{width:100%;box-sizing:border-box;padding:6px}
textarea{height:72px}
.row{display:flex;gap:6px}
.region{border:1px solid #ddd;padding:6px;margin:6px 0}
.hint{color:#666;font-size:12px}
code{font-size:12px}
</style>
<header>
 <b id="title">OCR-Vie Annotator</b>
 <button id="prev">◀ Prev</button>
 <button id="next">Next ▶</button>
 <span id="counter"></span>
 <select id="filterStatus"><option value="">all</option><option value="unreviewed">unreviewed</option><option value="reviewed">reviewed</option><option value="rejected">rejected</option></select>
 <button id="save">Save</button>
 <span id="msg"></span>
</header>
<main>
 <div id="left">
  <div class="hint">Drag on image to add rectangle (saved as 4-point polygon). Click a region in the list to select. Drag to move selected rectangle. Exact GT required — whitespace/punctuation preserved as visible.</div>
  <canvas id="cv"></canvas>
  <div class="hint">Keys: [ / ] prev/next, s save</div>
 </div>
 <div id="right">
  <label>id</label><input id="fid" disabled>
  <label>image</label><input id="fimage" disabled>
  <label>split</label><select id="fsplit"><option>train</option><option>val</option><option>test</option></select>
  <label>category</label><input id="fcategory" placeholder="scene">
  <label>language</label><input id="flang" placeholder="vie">
  <label>tags (comma-separated)</label><input id="ftags">
  <label>Image text (visible text, verbatim — do not normalize)</label><textarea id="ftext" placeholder="Exact visible text"></textarea>
  <label>annotation_level</label><select id="flevel"><option value="image">image</option><option value="region">region</option></select>
  <label>review_status</label><select id="fstatus"><option value="unreviewed">unreviewed</option><option value="reviewed">reviewed</option><option value="rejected">rejected</option></select>
  <label>privacy</label><select id="fprivacy"><option value="safe">safe</option><option value="restricted">restricted</option><option value="unknown">unknown</option></select>
  <label>license</label><input id="flicense">
  <label>notes</label><input id="fnotes">
  <label>annotated_by</label><input id="fannotated" placeholder="annotator_1">
  <label>reviewed_by</label><input id="freviewed" placeholder="reviewer_1">
  <hr>
  <div class="row"><b>Regions</b> <button id="addRegion">+ Add selected rect as region</button> <button id="clearRegions">Clear all</button></div>
  <div id="regions"></div>
  <div class="hint">Each region: polygon [x,y]×4, text verbatim, tags (e.g. occluded,cropped,partial,rotated), mark <code>ignore</code> for illegible.</div>
 </div>
</main>
<script>
let records=[], idx=0, img=null, scale=1, drag=null, selRegion=null, dirty=false, revision=null, original=null;
let saving=false, loading=false;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const g=id=>document.getElementById(id);
async function loadList(){
  if(saving || loading) return;
  if(dirty && !confirm('Discard unsaved changes?')) return;
  loading=true;
  drag=null; start=null;
  const controls=[...(document.querySelectorAll?.('input,select,textarea,button') || [])].map(el=>[el,el.disabled]);
  controls.forEach(([el])=>el.disabled=true);
  try {
  const q=g('filterStatus').value? '?status='+g('filterStatus').value:'';
  const r=await fetch('/api/list'+q); const j=await r.json();
  if(!r.ok){g('msg').textContent=j.error||'Cannot load annotations'; return;}
  records=j; revision=r.headers.get('ETag'); dirty=false;
  if(!records.length){ original=null; g('right').hidden=true; cv.hidden=true; g('save').disabled=true; g('msg').textContent='no records'; return; }
  g('right').hidden=false; cv.hidden=false; g('save').disabled=false;
  idx=Math.min(idx, records.length-1); loading=false; show(idx);
  } catch { g('msg').textContent='Cannot load annotations'; }
  finally { loading=false; controls.forEach(([el,disabled])=>el.disabled=disabled); g('save').disabled=!records.length; }
}
function show(i){
  if(saving || loading || !records[i]) return;
  if(dirty && !confirm('Discard unsaved changes?')) return;
  if(dirty && original) records[idx]=JSON.parse(original);
  dirty=false; selRegion=null;
  const rec=records[i]; idx=i;
  original=JSON.stringify(rec);
  rec.regions ||= [];
  g('counter').textContent=(i+1)+'/'+records.length;
  g('fid').value=rec.id||''; g('fimage').value=rec.image||'';
  g('fsplit').value=rec.split||'test'; g('fcategory').value=rec.category||'';
  g('flang').value=rec.language||'vie'; g('ftags').value=(rec.tags||[]).join(', ');
  g('ftext').value=rec.text||''; g('flevel').value=rec.annotation_level|| (rec.regions&&rec.regions.length?'region':'image');
  g('fstatus').value=rec.review_status||'unreviewed'; g('fprivacy').value=rec.privacy||'safe';
  g('flicense').value=rec.license||''; g('fnotes').value=rec.notes||'';
  g('fannotated').value=rec.annotated_by||''; g('freviewed').value=rec.reviewed_by||'';
  renderRegions(rec.regions||[]);
  loadImage(rec.image);
}
function renderRegions(regions){
  const c=g('regions'); c.innerHTML='';
  regions.forEach((r,i)=>{
    const d=document.createElement('div'); d.className='region';
    if(selRegion===i) d.style.background='#eef';
    d.innerHTML=`<div><b>#${i}</b> <button data-del="${i}">Delete</button> <button data-sel="${i}">Select</button> <label><input type="checkbox" data-ig="${i}"> ignore (illegible)</label> reason <input data-reason="${i}" placeholder="illegible"> tags <input data-rtags="${i}"></div>
      <div>polygon: <code></code></div>
      <input data-field="${i}" placeholder="field_key (optional, must exist in fields)">
      <textarea data-text="${i}" placeholder="GT text verbatim"></textarea>`;
    d.querySelector('[data-ig]').checked=!!r.ignore;
    d.querySelector('[data-reason]').value=r.reason||'';
    d.querySelector('[data-rtags]').value=(r.tags||[]).join(', ');
    d.querySelector('code').textContent=JSON.stringify(r.polygon||[]);
    d.querySelector('textarea').value=r.text||'';
    d.querySelector('[data-field]').value=r.field_key||'';
    c.appendChild(d);
  });
  c.querySelectorAll('[data-del]').forEach(b=>b.onclick=()=>{dirty=true; records[idx].regions.splice(parseInt(b.dataset.del),1); renderRegions(records[idx].regions); draw();});
  c.querySelectorAll('[data-sel]').forEach(b=>b.onclick=()=>{selRegion=parseInt(b.dataset.sel); renderRegions(records[idx].regions); draw();});
  c.querySelectorAll('[data-ig]').forEach(b=>b.onchange=()=>{const i=parseInt(b.dataset.ig); records[idx].regions[i].ignore=b.checked;});
  c.querySelectorAll('[data-reason]').forEach(b=>b.oninput=()=>{records[idx].regions[parseInt(b.dataset.reason)].reason=b.value;});
  c.querySelectorAll('[data-rtags]').forEach(b=>b.oninput=()=>{records[idx].regions[parseInt(b.dataset.rtags)].tags=b.value.split(',').map(s=>s.trim()).filter(Boolean);});
  c.querySelectorAll('[data-field]').forEach(b=>b.oninput=()=>{records[idx].regions[parseInt(b.dataset.field)].field_key=b.value.trim();});
  c.querySelectorAll('[data-text]').forEach(b=>b.oninput=()=>{records[idx].regions[parseInt(b.dataset.text)].text=b.value; const rec=records[idx]; rec.text=(rec.regions||[]).filter(r=>!r.ignore).map(r=>r.text||'').join(' '); g('ftext').value=rec.text;});
}
function loadImage(rel){
  const next=new Image(); img=next;
  next.onload=()=>{if(img!==next)return; cv.width=next.width; cv.height=next.height; draw();};
  next.onerror=()=>{if(img!==next)return; img=null; ctx.clearRect(0,0,cv.width,cv.height);};
  next.src='/image/'+rel;
}
function draw(){
  if(!img) return;
  ctx.clearRect(0,0,cv.width,cv.height); ctx.drawImage(img,0,0);
  const regions=records[idx]?.regions||[];
  regions.forEach((r,i)=>{
    const poly=r.polygon; if(!poly||poly.length<3) return;
    ctx.strokeStyle= (i===selRegion? '#f00':'#0a0'); ctx.lineWidth=2; ctx.beginPath();
    ctx.moveTo(poly[0][0], poly[0][1]);
    for(let k=1;k<poly.length;k++) ctx.lineTo(poly[k][0], poly[k][1]);
    ctx.closePath(); ctx.stroke();
    if(r.ignore){ ctx.fillStyle='rgba(255,0,0,0.15)'; ctx.fill(); }
  });
  if(drag&&drag.r){ const r=drag.r; ctx.strokeStyle='#00f'; ctx.strokeRect(r[0],r[1],r[2]-r[0],r[3]-r[1]); }
}
let start=null;
cv.addEventListener('mousedown', e=>{
  if(saving || loading || !records[idx]) return;
  const rect=cv.getBoundingClientRect(), sx=cv.width/rect.width, sy=cv.height/rect.height;
  const x=(e.clientX-rect.left)*sx, y=(e.clientY-rect.top)*sy;
  // if clicking near selected region, start move
  if(selRegion!=null){
    const poly=records[idx].regions[selRegion]?.polygon;
    if(poly){ const xs=poly.map(p=>p[0]), ys=poly.map(p=>p[1]); const x0=Math.min(...xs), y0=Math.min(...ys), x1=Math.max(...xs), y1=Math.max(...ys);
      if(x>=x0&&x<=x1&&y>=y0&&y<=y1){ drag={mode:'move', ox:x, oy:y, orig:poly.map(p=>[...p])}; return; }
    }
  }
  start=[x,y]; drag={r:[x,y,x,y]};
});
cv.addEventListener('mousemove', e=>{
  if(saving || loading) return;
  if(!drag) return;
  const rect=cv.getBoundingClientRect(), sx=cv.width/rect.width, sy=cv.height/rect.height;
  const x=(e.clientX-rect.left)*sx, y=(e.clientY-rect.top)*sy;
  if(drag.mode==='move'){ const dx=x-drag.ox, dy=y-drag.oy; const poly=records[idx].regions[selRegion].polygon; poly.forEach((p,i)=>{p[0]=drag.orig[i][0]+dx; p[1]=drag.orig[i][1]+dy;}); draw(); return; }
  if(start){ drag.r=[Math.min(start[0],x), Math.min(start[1],y), Math.max(start[0],x), Math.max(start[1],y)]; draw(); }
});
cv.addEventListener('mouseup', e=>{
  if(saving || loading) return;
  if(drag&&drag.mode==='move'){ dirty=true; drag=null; start=null; renderRegions(records[idx].regions); return; }
  if(!start) return;
  const r=drag.r; const w=r[2]-r[0], h=r[3]-r[1];
  if(w>8&&h>8){ dirty=true; selRegion=records[idx].regions.length; records[idx].regions.push({polygon:[[r[0],r[1]],[r[2],r[1]],[r[2],r[3]],[r[0],r[3]]], text:"", tags:[]}); renderRegions(records[idx].regions); }
  drag=null; start=null; draw();
});
g('prev').onclick=()=>{ if(idx>0) show(idx-1);};
g('next').onclick=()=>{ if(idx<records.length-1) show(idx+1);};
g('filterStatus').onchange=loadList;
g('addRegion').onclick=()=>{
  if(!drag||!drag.r) return alert('Draw a rectangle on the image first');
  const r=drag.r; records[idx].regions.push({polygon:[[r[0],r[1]],[r[2],r[1]],[r[2],r[3]],[r[0],r[3]]], text:"", tags:[]}); drag=null; renderRegions(records[idx].regions); draw();
};
g('clearRegions').onclick=()=>{ if(confirm('Clear all regions?')){ dirty=true; records[idx].regions=[]; selRegion=null; renderRegions([]); draw(); }};
async function save(){
  if(saving || loading || !records[idx]) return;
  const rec=records[idx];
  rec.split=g('fsplit').value; rec.category=g('fcategory').value||rec.category;
  rec.language=g('flang').value||'vie'; rec.tags=g('ftags').value.split(',').map(s=>s.trim()).filter(Boolean);
  rec.text=g('ftext').value; rec.annotation_level=g('flevel').value; rec.review_status=g('fstatus').value;
  rec.privacy=g('fprivacy').value; rec.license=g('flicense').value; rec.notes=g('fnotes').value;
  rec.annotated_by=g('fannotated').value||undefined; rec.reviewed_by=g('freviewed').value||undefined;
  if(!rec.annotated_by) delete rec.annotated_by; if(!rec.reviewed_by) delete rec.reviewed_by;
  rec.annotation_complete = !!(rec.text && rec.text.trim());
  const payload=JSON.stringify(rec);
  saving=true; dirty=true; drag=null; start=null;
  const controls=[...document.querySelectorAll('input,select,textarea,button')].map(el=>[el,el.disabled]);
  controls.forEach(([el])=>el.disabled=true);
  try {
  const r=await fetch('/api/save',{method:'POST', headers:{'Content-Type':'application/json','If-Match':revision||''}, body:payload});
  const j=await r.json();
  if(r.ok && j.ok){revision=r.headers.get('ETag'); original=payload; dirty=false;}
  g('msg').textContent=r.ok && j.ok?'saved':'error: '+(j.error||'Cannot save');
  } catch { g('msg').textContent='Cannot save; changes kept. Retry after checking connection.'; }
  finally { saving=false; controls.forEach(([el,disabled])=>el.disabled=disabled); }
}
g('save').onclick=save;
document.addEventListener('input', e=>{if(e.target.id!=='filterStatus') dirty=true;});
document.addEventListener('change', e=>{if(e.target.id!=='filterStatus') dirty=true;});
window.addEventListener('beforeunload', e=>{if(dirty){e.preventDefault(); e.returnValue='';}});
document.addEventListener('keydown', e=>{ if(e.key==='s'&&(e.ctrlKey||e.metaKey)){ e.preventDefault(); save(); } if(e.target.matches('input,textarea,select')) return; if(e.key==='[') g('prev').click(); if(e.key===']') g('next').click(); });
loadList();
</script>
"""


def _load_records(ann_path: Path) -> list[dict]:
    return load_annotations(ann_path)


def _save_records(ann_path: Path, records: list[dict], *, expected_revision: str | None = None) -> None:
    if expected_revision is not None and sha256_file(ann_path) != expected_revision:
        raise ValueError('Dataset changed; reload before saving')
    ids = set()
    for record in records:
        if (not isinstance(record, dict) or not isinstance(record.get('id'), str)
                or not record['id'] or record['id'] in ids
                or not isinstance(record.get('split'), str)
                or record.get('split') not in VALID_SPLITS
                or not isinstance(record.get('image'), str)
                or not isinstance(record.get('text'), str)
                or not isinstance(record.get('regions', []), list)):
            raise ValueError('Invalid annotation schema or duplicate ID')
        ids.add(record['id'])
        validate_ground_truth(record)
    tmp = ann_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(ann_path)
    # rewrite splits
    splits_dir = ann_path.parent / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val", "test"):
        ids = [str(r["id"]) for r in records if r.get("split") == sp]
        (splits_dir / f"{sp}.txt").write_text("".join(i + "\n" for i in ids), encoding="utf-8")


def make_handler(data_dir: Path):
    ann_path = data_dir / "annotations.jsonl"

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, payload, revision=None):
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            if revision:
                self.send_header('ETag', revision)
            self.end_headers()
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode('utf-8'))

        def log_message(self, fmt, *args):
            sys.stderr.write(fmt % args + "\n")

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML.encode("utf-8"))
                return
            if parsed.path.startswith("/api/list"):
                qs = parse_qs(parsed.query)
                status = (qs.get("status") or [None])[0]
                try:
                    revision = sha256_file(ann_path)
                    recs = _load_records(ann_path)
                    if revision != sha256_file(ann_path):
                        raise ValueError('Dataset changed; reload')
                except ValueError as exc:
                    self.reply(400, {'ok': False, 'error': str(exc)})
                    return
                if status:
                    recs = [r for r in recs if r.get("review_status") == status]
                self.reply(200, recs, revision)
                return
            if parsed.path.startswith("/image/"):
                rel = unquote(parsed.path[len("/image/") :])
                # prevent directory traversal
                p = (data_dir / rel).resolve()
                try:
                    p.relative_to(data_dir.resolve())
                except Exception:
                    self.send_error(403)
                    return
                if not p.is_file() or p.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'):
                    self.send_error(404)
                    return
                self.send_response(200)
                # guess content type
                ctype = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
                self.send_header("Content-Type", ctype)
                self.end_headers()
                self.wfile.write(p.read_bytes())
                return
            self.send_error(404)

        def do_POST(self):
            if self.path != "/api/save":
                self.send_error(404)
                return
            revision = self.headers.get('If-Match')
            if not revision:
                self.reply(428, {'ok': False, 'error': 'Reload dataset before saving (missing revision)'})
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
            except ValueError:
                length = 0
            if not 0 < length <= 1024 * 1024:
                self.reply(413, {'ok': False, 'error': 'Invalid or oversized save body'})
                return
            body = self.rfile.read(length)
            try:
                rec = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode())
                return
            rid = str(rec.get("id", "")) if isinstance(rec, dict) else ''
            if not rid:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"ok":false,"error":"missing id"}')
                return
            try:
                records = _load_records(ann_path)
            except ValueError as exc:
                self.reply(400, {'ok': False, 'error': str(exc)})
                return
            found = False
            for i, r in enumerate(records):
                if str(r.get("id")) == rid:
                    for key in ('image', 'sha256', 'source_type'):
                        if key in r:
                            rec[key] = r[key]
                    records[i] = rec
                    found = True
                    break
            if not found:
                self.reply(404, {'ok': False, 'error': 'Unknown annotation ID'})
                return
            try:
                _save_records(ann_path, records, expected_revision=revision)
            except ValueError as exc:
                self.reply(409 if 'changed' in str(exc) else 400, {'ok': False, 'error': str(exc)})
                return
            except OSError:
                self.reply(500, {'ok': False, 'error': 'Storage write failed; primary annotations may be saved. Reload before retrying.'})
                return
            self.reply(200, {'ok': True}, sha256_file(ann_path))

    return Handler


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve() if not data_dir.exists() else data_dir.resolve()
        alt = ROOT / args.data
        if alt.exists():
            data_dir = alt.resolve()
    if data_dir.is_file():
        data_dir = data_dir.parent
    if not (data_dir / "annotations.jsonl").exists():
        print(f"ERROR: {data_dir / 'annotations.jsonl'} not found", file=sys.stderr)
        return 1
    handler = make_handler(data_dir)
    httpd = HTTPServer((args.host, args.port), handler)
    print(f"Annotator running at http://{args.host}:{args.port}/  data={data_dir}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
