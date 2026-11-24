// Run with Node's built-in WebSocket client and the existing Chromium test image.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';

const base = process.argv[2] || 'http://127.0.0.1:31800';
const dataset = process.argv[3];
assert.match(dataset || '', /^ds_[A-Za-z0-9]+$/);
const search = process.argv[4] || 'Standing_High_Jump_00001';
const profile = mkdtempSync('/tmp/md-chromium-');
const browser = spawn('/usr/bin/chromium', ['--headless=new', '--no-sandbox', '--disable-gpu',
  '--disable-dev-shm-usage', '--autoplay-policy=no-user-gesture-required',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--window-size=1440,1000', 'about:blank'],
  { stdio: 'ignore' });
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let socket;
try {
  let port;
  for (let i = 0; i < 100; i++) {
    try { port = readFileSync(`${profile}/DevToolsActivePort`, 'utf8').split('\n')[0]; break; }
    catch { await sleep(100); }
  }
  assert.ok(port, 'Chromium did not start');
  const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  socket = new WebSocket(targets.find((target) => target.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  let nextId = 0;
  const pending = new Map();
  socket.addEventListener('message', ({ data }) => {
    const message = JSON.parse(data);
    if (message.id && pending.has(message.id)) {
      const { resolve, reject } = pending.get(message.id); pending.delete(message.id);
      message.error ? reject(new Error(JSON.stringify(message.error))) : resolve(message.result);
    }
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++nextId; pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async (expression) => {
    const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const until = async (expression, seconds = 60) => {
    for (let i = 0; i < seconds; i++) {
      if (await evaluate(expression)) return;
      if (i && i % 30 === 0) console.log('WAIT', i, await evaluate('document.querySelector(".motiondecode")?.innerText.slice(-450)'));
      await sleep(1000);
    }
    throw new Error(`Browser condition timed out: ${expression}`);
  };
  await send('Network.enable');
  await send('Network.setExtraHTTPHeaders', { headers: { Authorization: `Bearer ${process.env.EB_TEST_API_TOKEN || 'eb-dev-key'}` } });
  await send('Page.enable');
  await send('Page.navigate', { url: `${base}/data/${dataset}` });
  await until('!!document.querySelector(".motiondecode input[type=search]")');
  await evaluate(`(() => { const input=document.querySelector('.motiondecode input[type=search]'); input.value=${JSON.stringify(search)}; input.dispatchEvent(new Event('input',{bubbles:true})); input.closest('form').requestSubmit(); })()`);
  await until(`document.querySelector('.motiondecode tbody')?.innerText.includes(${JSON.stringify(search)}) && !document.querySelector('.motiondecode [role=status]')`);
  await evaluate("document.querySelector('.motiondecode tbody button').click()");
  await until('!!document.querySelector(".motiondecode video") && !!document.querySelector(".motiondecode table tbody tr th")', 600);
  await until('document.querySelector(".motiondecode video").readyState >= 2');
  const first = await evaluate("(() => { const v=document.querySelector('.motiondecode video'); return {width:v.videoWidth,height:v.videoHeight,duration:v.duration,firstFrame:document.querySelector('.motiondecode [role=region] table tbody tr th')?.textContent,columns:document.querySelectorAll('.motiondecode [role=region] table thead th').length} })()");
  assert.equal(first.width, 640); assert.equal(first.height, 480); assert.ok(first.duration > 0);
  assert.equal(first.firstFrame, '0'); assert.equal(first.columns, 37);
  await evaluate("document.querySelector('.motiondecode video').play()");
  await sleep(1500);
  assert.ok(await evaluate("document.querySelector('.motiondecode video').currentTime > .2"));
  mkdirSync('/artifacts', { recursive: true });
  await evaluate("document.querySelector('.motiondecode video').scrollIntoView({block:'center'})");
  writeFileSync('/artifacts/motiondecode-video.png', Buffer.from((await send('Page.captureScreenshot')).data, 'base64'));
  await evaluate("document.querySelector('.motiondecode [aria-label=源数据表分页] button:last-child').click()");
  await until("document.querySelector('.motiondecode table tbody tr th')?.textContent==='100'");
  await evaluate("document.querySelector('.motiondecode [aria-label=\"CSV 原始数值，横向滚动查看全部列\"]').scrollIntoView({block:'start'})");
  writeFileSync('/artifacts/motiondecode-table.png', Buffer.from((await send('Page.captureScreenshot')).data, 'base64'));
  writeFileSync('/artifacts/motiondecode-browser.json', JSON.stringify({ dataset, search, ...first, tableSecondPageFirstFrame:100, videoPlayback:true }, null, 2));
  console.log('BROWSER_TABLE_PAGING_AND_VIDEO_PLAYBACK_OK', JSON.stringify(first));
} finally {
  socket?.close(); browser.kill('SIGTERM');
}
