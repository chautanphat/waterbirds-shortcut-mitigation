// Read-only browser acceptance of the real saved mild-blur failure gallery.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
const root=path.resolve('.');
const require=createRequire(path.join(root,'outputs/01a1ee6e-d18d-7d83-a20d-409d909f4897/browser-check.cjs'));
const {chromium}=require('playwright');
const browser=await chromium.launch({headless:true,
  executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
const page=await browser.newPage({viewport:{width:1440,height:1050}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
page.on('response',r=>{if(r.url().includes('/api/')&&r.status()!==200)errors.push(r.status()+' '+r.url())});
try{
  await page.goto('http://127.0.0.1:8789/',{waitUntil:'networkidle'});
  await page.locator('#gallery tr').first().waitFor();
  await page.selectOption('#mode','all');await page.selectOption('#severity','mild');
  await page.waitForFunction(()=>document.getElementById('status').textContent.startsWith('400 ảnh'));
  await page.locator('#gallery tr').first().click();
  await page.locator('#detail').scrollIntoViewIfNeeded();
  await page.waitForFunction(()=>[...document.querySelectorAll('#panes img')].every(im=>im.complete&&im.naturalWidth>0));
  const title=await page.locator('#case-title').textContent();
  if(!title.includes('σ=0.5'))throw Error('Selected case did not retain actual mild sigma');
  const exactModels=await page.locator('#comparison tbody tr').count();
  if(exactModels!==3)throw Error('Missing real model comparison');
  await page.screenshot({path:path.join(root,'experiments/counterfactual_verification/comparison_mild_v2_browser.png')});
  await page.selectOption('#cohort','original');
  await page.waitForFunction(()=>document.getElementById('status').textContent.startsWith('100 ảnh'));
  await page.selectOption('#cohort','replication');
  await page.waitForFunction(()=>document.getElementById('status').textContent.includes('/ 3000'));
  await page.selectOption('#control','background_only');
  await page.waitForFunction(()=>document.getElementById('status').textContent.startsWith('400 ảnh'));
  if(await page.locator('#error').textContent())throw Error(await page.locator('#error').textContent());
  if(errors.length)throw Error(errors.join('\n'));
  await fs.writeFile(path.join(root,'experiments/counterfactual_verification/comparison_mild_v2_browser.json'),
    JSON.stringify({status:'passed',selected_title:title,models:exactModels,original_sources:100,
      mild_natural_images:400,background_only_assignments:400,page_errors:errors},null,2));
  console.log('Browser passed: saved three-model predictions, exact images, mild filter, original subset and OOD controls.');
}finally{await browser.close()}
