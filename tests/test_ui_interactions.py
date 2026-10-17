"""UI regressions; execute the actual shipped script without loading OCR models."""
import re
import json
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient
from app.main import app, INDEX_HTML


def test_workspace_accessibility_and_responsive_states():
    html = TestClient(app).get("/").text
    assert '<main id="workspace"' in html
    assert 'id="status" role="status"' in html
    assert 'id="copy-btn" type="button" disabled' in html
    assert 'id="download-btn" type="button" disabled' in html
    assert 'prefers-reduced-motion' in html
    assert '@media' in html


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js needed for inline JS regression check")
def test_real_script_request_lock_results_and_preview_lifecycle():
    script = re.search(r"<script>(.*?)</script>", INDEX_HTML, re.S).group(1)
    harness = r'''
const vm = require('node:vm'), assert = require('node:assert/strict');
const nodes = {};
function element() {
  return {textContent:'', _value:'auto', get value(){return this._value},
    set value(v){this._value=v;if(v==='')this.files=[]}, files:[], style:{}, dataset:{}, hidden:false,
    disabled:false, handlers:{}, classList:{add(){},remove(){},toggle(){}},
    addEventListener(k, fn){this.handlers[k]=fn}, setAttribute(k,v){this[k]=v},
    removeAttribute(k){delete this[k]}, focus(){}, click(){}, select(){}, remove(){}};
}
for (const id of [...html.matchAll(/id="([^"]+)"/g)].map(x=>x[1])) nodes[id]=element();
nodes.lang.value='vie+eng';
let calls=0, resolveOCR, revoked=[], copied='', completed=false, downloadBlob, clipboardFails=false, timeoutOCR;
process.on('beforeExit',()=>{if(!completed){console.error('UI scenario did not finish');process.exitCode=1}});
const ctx = vm.createContext({console, AbortController, setTimeout(fn,ms){if(ms>60000){timeoutOCR=fn;return 0}return setTimeout(fn,ms)}, clearTimeout, Blob, URLSearchParams,
  navigator:{clipboard:{async writeText(t){if(clipboardFails)throw Error('denied');copied=t}}},
  URL:{createObjectURL(obj){if(obj instanceof Blob)downloadBlob=obj;return 'blob:'+Math.random()},revokeObjectURL(u){revoked.push(u)}},
  FormData:class{append(){}}, window:{addEventListener(){}},
  document:{getElementById(id){return nodes[id]},createElement:element,
    querySelectorAll(){return Object.values(nodes)},body:{appendChild(){}},execCommand(){return false}},
  fetch(url,options){if(url==='/health/ready') return Promise.resolve({ok:false,status:503,json:async()=>({
    primary_scene_pipeline:{status:'not_loaded'}, primary_document_pipeline:{status:'ready'}})});
    calls++; return new Promise((r,reject)=>{resolveOCR=r;options?.signal?.addEventListener('abort',()=>reject(Object.assign(new Error('aborted'),{name:'AbortError'})));});}
});
vm.runInContext(script,ctx);
const event={preventDefault(){}};
const response=(text)=>({ok:true,json:async()=>({success:true,text,regions:[],warnings:[],
  route:{route:'document'},engines:{recognizer:'tesseract'},provenance:{},document:null,
    fallback_used:false,timing_ms:{classification:1,detection:0,recognition:12,total:13}})});
(async()=>{
  const f={name:'test.png',size:120,type:'image/png'};
  nodes.file.files=[f]; nodes.file.handlers.change();
  assert.equal(nodes['upload-zone']?.dataset.selected,'true','selected image uses compact upload controls');
  const first=nodes['ocr-form'].handlers.submit(event);
  const duplicate=nodes['ocr-form'].handlers.submit(event);
  assert.equal(calls,1,'busy submissions must not start another OCR request');
  resolveOCR(response('  Tiếng Việt có dấu   \r\nDòng 2  ')); await first; await duplicate;
  assert.equal(nodes['ocr-text'].textContent,'Tiếng Việt có dấu\nDòng 2');
  assert.equal(nodes['copy-btn'].disabled,false);
  await nodes['copy-btn'].handlers.click(); assert.equal(copied,'Tiếng Việt có dấu\nDòng 2');
  clipboardFails=true; await nodes['copy-btn'].handlers.click();
  assert.match(nodes['copy-feedback'].textContent,/Không sao chép được/);
  nodes['download-btn'].handlers.click();
  assert.equal(await downloadBlob.text(),'Tiếng Việt có dấu\nDòng 2','download must contain only clean text');
  nodes.file.handlers.change(); assert.equal(revoked.length,1,'old image URL must be revoked');
  assert.equal(nodes['copy-btn'].disabled,true,'changing input invalidates old result');
  const empty=nodes['ocr-form'].handlers.submit(event); resolveOCR(response('')); await empty;
  assert.equal(nodes['download-btn'].disabled,true,'empty text cannot be exported');
  const failed=nodes['ocr-form'].handlers.submit(event);
  resolveOCR({ok:false,status:413,json:async()=>({detail:'File too large'})}); await failed;
  assert.equal(nodes['copy-btn'].disabled,true);
  assert.equal(nodes.file.disabled,false,'controls must recover after an error');
  assert.match(nodes.status.textContent,/lớn/i,'error should provide a useful Vietnamese explanation');
  const warned=nodes['ocr-form'].handlers.submit(event);
  const warnResponse=response('Xin chào'); const warnData=await warnResponse.json();
  warnData.warnings=["classifier_low_confidence: fallback to 'scene' (reason=ambiguous_layout)"];
  resolveOCR({ok:true,json:async()=>warnData}); await warned;
  assert.match(nodes.warning.textContent,/Chưa xác định chắc loại ảnh/);
  const unsupported=nodes['ocr-form'].handlers.submit(event);
  resolveOCR({ok:false,status:503,json:async()=>({detail:{code:'MODEL_LANGUAGE_UNSUPPORTED',message:'Unsupported'}})}); await unsupported;
  assert.match(nodes.status.textContent,/chưa đủ ký tự tiếng Việt/i);
  const overloaded=nodes['ocr-form'].handlers.submit(event);
  resolveOCR({ok:false,status:503,headers:{get(){return '2'}},json:async()=>({detail:{code:'OCR_BUSY'}})}); await overloaded;
  assert.match(nodes.status.textContent,/đang bận/i);
  assert.match(nodes.status.textContent,/2 giây/i);
  assert.equal(nodes.file.files.length,1,'busy error keeps image for retry');
  const timedOut=nodes['ocr-form'].handlers.submit(event);
  resolveOCR({ok:false,status:504,json:async()=>({detail:{code:'OCR_TIMEOUT'}})}); await timedOut;
  assert.match(nodes.status.textContent,/thời gian|quá lâu/i);
  assert.equal(nodes.file.files.length,1,'timeout keeps image');
  const hanging=nodes['ocr-form'].handlers.submit(event);
  assert.equal(typeof timeoutOCR,'function','a hanging request needs a browser deadline');
  timeoutOCR(); await hanging;
  assert.equal(nodes.file.disabled,false,'browser timeout unlocks controls');
  assert.equal(nodes.file.files.length,1);
  assert.match(nodes.status.textContent,/quá lâu|thời gian/i);
  nodes.file.dataset.maxBytes='100'; nodes.file.handlers.change();
  assert.equal(nodes.file.files.length,0,'oversized file must be cleared');
  assert.equal(nodes['submit-btn'].disabled,true);
  assert.equal(nodes['upload-zone'].dataset.selected,'false');
  assert.match(nodes.status.textContent,/lớn/i);
  assert.match(nodes.health.textContent,/khởi tạo/i,'cold model is not unavailable');
  completed=true;
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    result = subprocess.run(
        [shutil.which("node")],
        input="const html=" + json.dumps(INDEX_HTML) + "; const script=" + json.dumps(script) + ";" + harness,
        capture_output=True, text=True, encoding="utf-8", timeout=15,
    )
    assert result.returncode == 0, result.stderr
