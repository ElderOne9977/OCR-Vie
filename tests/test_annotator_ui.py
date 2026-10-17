import json
import re
import shutil
import subprocess

import pytest
from benchmarks.annotate import HTML


@pytest.mark.skipif(not shutil.which('node'), reason='Node required for shipped JS check')
def test_discard_restores_region_edits_and_cancel_keeps_current_image():
    script = re.search(r'<script>(.*?)</script>', HTML, re.S).group(1)
    harness = r'''
const vm=require('node:vm'), assert=require('node:assert/strict');
const nodes={}, ctx=vm.createContext({console, confirm:()=>true, Image:class{},
 fetch:()=>new Promise(()=>{}), window:{addEventListener(){}},
 document:{addEventListener(){},getElementById(id){return nodes[id] ||= {
 value:'',textContent:'',getContext(){return {}},addEventListener(){}}}}});
vm.runInContext(script,ctx);
vm.runInContext(`loading=false;renderRegions=()=>{}; loadImage=()=>{};
 records=[{id:'a',regions:[{text:'original'}]}, {id:'b',regions:[]}]; show(0);
 records[0].regions[0].text='unsaved'; dirty=true; show(1); show(0);`,ctx);
assert.equal(vm.runInContext('records[0].regions[0].text',ctx),'original');
ctx.confirm=()=>false;
vm.runInContext('dirty=true; show(1)',ctx);
assert.equal(vm.runInContext('idx',ctx),0);
assert.equal(vm.runInContext('dirty',ctx),true);
'''
    result = subprocess.run([shutil.which('node')], input='const script='+json.dumps(script)+';'+harness,
                            capture_output=True, text=True, encoding='utf-8', timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not shutil.which('node'), reason='Node required')
def test_save_locks_navigation_and_recovers_after_network_error():
    script = re.search(r'<script>(.*?)</script>', HTML, re.S).group(1)
    harness = r'''
const vm=require('node:vm'), assert=require('node:assert/strict'), nodes={};
const ctx=vm.createContext({console,confirm:()=>true,Image:class{},
 fetch:()=>new Promise(()=>{}),window:{addEventListener(){}},
 document:{querySelectorAll:()=>Object.values(nodes),addEventListener(){},getElementById(id){return nodes[id] ||= {
 value:'',textContent:'',disabled:false,getContext(){return {}},addEventListener(){}}}}});
vm.runInContext(script,ctx);
let completed=false;process.on('beforeExit',()=>{if(!completed)process.exitCode=1});
    vm.runInContext(`loading=false;renderRegions=()=>{};loadImage=()=>{};
records=[{id:'a',text:'a',regions:[]},{id:'b',text:'b',regions:[]}];show(0);
let resolveSave,calls=0;fetch=()=>{calls++;return new Promise(r=>resolveSave=r)};
(async()=>{
dirty=true;let pending=save();show(1);await save();
if(idx!==0||calls!==1||!g('ftext').disabled)throw Error('save must lock edits/navigation/duplicates');
resolveSave({ok:true,json:async()=>({ok:true}),headers:{get:()=> 'new'}});await pending;
if(dirty||g('ftext').disabled)throw Error('success must unlock');
dirty=true;fetch=async()=>{throw Error('offline')};await save();
if(!dirty||g('ftext').disabled)throw Error('failure must preserve edits and unlock');
show(1);if(idx!==1)throw Error('navigation must recover');
fetch=()=>new Promise(r=>resolveSave=r);const listing=loadList();show(0);
if(idx!==1||!g('ftext').disabled)throw Error('loading must lock navigation and edits');
resolveSave({ok:true,json:async()=>[],headers:{get:()=> 'empty'}});await listing;
if(!g('save').disabled||!g('right').hidden)throw Error('empty filter must disable editor');
})()`,ctx).then(()=>{completed=true}).catch(e=>{console.error(e);process.exitCode=1});
'''
    result=subprocess.run([shutil.which('node')],input='const script='+json.dumps(script)+';'+harness,
                          capture_output=True,text=True,encoding='utf-8',timeout=10)
    assert result.returncode == 0, result.stderr
