import { chromium } from '/root/autodl-tmp/Quest/web/frontend/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';

const output=new URL('.',import.meta.url);
const browser=await chromium.launch({headless:true,executablePath:'/root/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome',args:['--no-sandbox']});
const errors=[];
const page=await browser.newPage({viewport:{width:1500,height:1100}});
page.on('pageerror',error=>errors.push(error.message));
const result={};
try {
  await page.goto('http://127.0.0.1:6080/tool-dashboard.html?dataset=clm_hermes_34_c12_20261108_123828&trace=row0093');
  await page.waitForFunction(()=>document.querySelectorAll('#tool-table tr').length>=4);
  result.grouping=await page.locator('#tool-table').innerText();
  assert(result.grouping.includes('web_search · 搜索'));
  assert(result.grouping.includes('web_extract · 网页提取'));
  assert(result.grouping.includes('上下文编辑'));
  await page.selectOption('#tool','context_edit');
  await page.selectOption('#status-filter','all');
  await page.waitForFunction(()=>document.querySelector('#details').textContent.includes('执行入口：bash'));
  result.contextFilter=await page.locator('#details').innerText();
  assert(result.contextFilter.includes('查看 bash 命令'));
  const download=page.waitForEvent('download');
  await page.click('#download');
  const file=await download;
  await file.saveAs(new URL('tool-health.csv',output).pathname);
  result.csvHeader=(await fs.readFile(new URL('tool-health.csv',output),'utf8')).split('\r\n')[0];
  assert(result.csvHeader.includes('执行入口')&&result.csvHeader.includes('原始命令'));
  await page.selectOption('#tool','web_search');
  await page.waitForFunction(()=>document.querySelector('#tool-table').textContent.includes('web_search'));
  result.searchFilter=await page.locator('#tool-table').innerText();
  assert(!result.searchFilter.includes('上下文编辑'));
  await page.selectOption('#tool','');
  await page.screenshot({path:new URL('dashboard.png',output).pathname,fullPage:true});
  assert.deepEqual(errors,[]);
  result.errors=errors;
  result.status='passed';
} catch(error) {
  result.status='failed';result.error=String(error);throw error;
} finally {
  await fs.writeFile(new URL('browser-check.json',output),JSON.stringify(result,null,2));
  await browser.close();
}
console.log(JSON.stringify(result));
