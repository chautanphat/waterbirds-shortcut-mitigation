// Serialize the actual Lab controls; verify physical units and separate operators.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('src/counterfactual/web/lab.js','utf8');
const code=source.slice(source.indexOf('function config(){'),source.indexOf('\nasync function run('));
const fields={severity:{value:'all'},operator:{value:'gaussian'},land:{selectedOptions:[]},water:{selectedOptions:[]},
 birds:{selectedOptions:[{value:'hf-fg-1'}]},quota:{value:'1'},seed:{value:'2026'},sigma:{value:'4'},
 'operator-strength':{value:'.4'},opacity:{value:'1'},fit:{value:'cover'},'bg-control':{checked:true},
 'neutral-control':{checked:true},approximate:{checked:false}};
const context={$:id=>fields[id],selected:id=>fields[id].selectedOptions.map(x=>x.value)};
vm.createContext(context);vm.runInContext(code,context);
const config=()=>JSON.parse(JSON.stringify(context.config()));
let c=config();assert.deepEqual(c.severities,{clean:0,mild:1.5,moderate:4,strong:9});assert.equal(c.operator,'gaussian');
fields.operator.value='contrast';c=config();assert.deepEqual(Object.values(c.severities),[0,0,0,0]);
assert.deepEqual(c.operator_strengths,{clean:1,mild:.75,moderate:.4,strong:.1});
fields.operator.value='occlusion';c=config();assert.deepEqual(c.operator_strengths,{clean:0,mild:.1,moderate:.3,strong:.5});
fields.severity.value='custom';fields.opacity.value='.5';c=config();assert.deepEqual(c.severities,{custom:0});assert.equal(c.operator_strengths.custom,.4);assert.equal(c.opacity,.5);
fields.operator.value='gaussian';fields.sigma.value='9.5';assert.deepEqual(config().severities,{custom:9.5});
fields.land.selectedOptions=[{value:'hf-bg-1'}];assert.throws(config,/same number/);
console.log('Lab controls: primary presets, separate operators, custom units and balanced quotas passed');
