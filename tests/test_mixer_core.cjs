// Numeric tests for the actual shared Mixer primitives, without a browser.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const calls=[];
const context={Uint8ClampedArray,document:{createElement(){return {width:0,height:0,getContext(){return {drawImage(...args){calls.push(args)},getImageData(x,y,w,h){return {x,y,w,h}}}}}}}};
vm.createContext(context);
vm.runInContext(fs.readFileSync('scripts/mixer_core.js','utf8'),context);
const {blend,aligned}=context.MixerCore;
assert.deepEqual([...blend([255,0,0,255],[0,0,255,255],.5)],[128,0,128,255]);
assert.deepEqual([...blend([255,0,0,0],[0,0,255,255],.5)],[0,0,255,128]);
assert.deepEqual([...blend([255,0,0,0],[0,0,255,0],.5)],[0,0,0,0]);
assert.deepEqual([...blend([12,34,56,255],[98,76,54,255],1)],[12,34,56,255]);
const image={naturalWidth:100,naturalHeight:200};
aligned(image,200,200,'cover');assert.deepEqual(calls.pop().slice(1),[0,-100,200,400]);
aligned(image,200,200,'contain');assert.deepEqual(calls.pop().slice(1),[50,0,100,200]);
aligned(image,200,200,'stretch');assert.deepEqual(calls.pop().slice(1),[0,0,200,200]);
assert.match(fs.readFileSync('scripts/image_mixer.html','utf8'),/const \{aligned,blend\}=MixerCore/);
console.log('Mixer: 8 alpha/alignment/integration checks passed');
