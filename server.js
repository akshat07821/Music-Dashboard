'use strict';
// Watches the LabVIEW data.csv, runs the trained mood models, and serves the dashboard and JSON API.
// No npm install needed (built-in modules only).
const fs = require('fs');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');

const ROOT = __dirname;
const cfg = Object.assign(
  { python_cmd: 'python', data_file: 'data.csv', output_dir: 'output', port: 3000, poll_ms: 1000, min_interval_sec: 5 },
  JSON.parse(fs.readFileSync(path.join(ROOT, 'config.json'), 'utf8'))
);
if (process.env.PORT) cfg.port = Number(process.env.PORT);
const rp = (p) => (path.isAbsolute(p) ? p : path.join(ROOT, p));
const DATA = rp(cfg.data_file);
const OUT = rp(cfg.output_dir);

const state = { watching: DATA, running: false, runs: 0, last_run: null, last_ok: null, last_error: null, last_ms: null };
let pending = false;
let timer = null;
let lastStart = 0;

function runAnalyzer(reason) {
  if (state.running) { pending = true; return; }
  state.running = true;
  lastStart = Date.now();
  let err = '';
  let done = false;
  const finish = (code, msg) => {
    if (done) return;
    done = true;
    state.running = false;
    state.runs++;
    state.last_run = new Date().toISOString();
    state.last_ms = Date.now() - lastStart;
    if (code === 0) { state.last_ok = state.last_run; state.last_error = null; }
    else { state.last_error = msg || `analyzer exited with code ${code}`; console.error('[node] ' + state.last_error); }
    if (pending) { pending = false; schedule('queued'); }
  };
  console.log(`[node] analyzing (${reason})`);
  const p = spawn(cfg.python_cmd, [path.join(ROOT, 'analyzer.py')], { cwd: ROOT });
  p.stdout.on('data', (d) => process.stdout.write('[py] ' + d));
  p.stderr.on('data', (d) => { err += d; process.stderr.write('[py] ' + d); });
  p.on('error', (e) => finish(1, `Cannot start "${cfg.python_cmd}": ${e.message}. Try "py" or "python3" in config.json.`));
  p.on('close', (code) => finish(code, code ? err.trim().split('\n').slice(-2).join(' | ') : null));
}

// leading-edge throttle: never more than one run per min_interval_sec, never starved by a busy writer
function schedule(reason) {
  if (timer) return;
  const wait = Math.max(0, cfg.min_interval_sec * 1000 - (Date.now() - lastStart));
  timer = setTimeout(() => { timer = null; runAnalyzer(reason); }, wait);
}

fs.watchFile(DATA, { interval: cfg.poll_ms }, (cur, prev) => {
  if (cur.mtimeMs !== prev.mtimeMs || cur.size !== prev.size) schedule('data.csv changed');
});

// ---- tiny API ----
function readJson(name) {
  const file = path.join(OUT, name);
  if (!fs.existsSync(file)) return null;
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}
function parseCsv(text) {
  const rows = []; let row = [], cur = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { cur += '"'; i++; } else q = false; } else cur += c; }
    else if (c === '"') q = true;
    else if (c === ',') { row.push(cur); cur = ''; }
    else if (c === '\n') { row.push(cur); rows.push(row); row = []; cur = ''; }
    else if (c !== '\r') cur += c;
  }
  if (cur !== '' || row.length) { row.push(cur); rows.push(row); }
  return rows;
}
function readCsv(name) {
  try {
    const rows = parseCsv(fs.readFileSync(path.join(OUT, name), 'utf8')).filter((r) => r.length > 1);
    const head = rows.shift() || [];
    return rows.map((r) => Object.fromEntries(head.map((h, i) => [h, r[i] !== '' && r[i] !== undefined && !isNaN(r[i]) ? Number(r[i]) : r[i]])));
  } catch { return []; }
}

http.createServer((req, res) => {
  const u = new URL(req.url, 'http://localhost');
  const send = (code, obj) => {
    res.writeHead(code, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
    res.end(JSON.stringify(obj, null, 2));
  };
  const sendHtml = () => {
    const file = path.join(ROOT, 'music_scada_dashboard.html');
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    fs.createReadStream(file).pipe(res);
  };
  switch (u.pathname) {
    case '/': return sendHtml();
    case '/api/dashboard': {
      const dashboard = readJson('dashboard.json');
      return dashboard ? send(200, { ...dashboard, node: state }) : send(503, { error: 'Analysis is not ready', node: state });
    }
    case '/api/status': return send(200, { node: state, analysis: readJson('status.json') });
    case '/api/latest': return send(200, readJson('latest.json'));
    case '/api/sessions': return send(200, readCsv('sessions.csv'));
    case '/api/alarms': return send(200, readCsv('alarms.csv'));
    case '/api/windows': {
      let w = readJson('windows.json') || [];
      const s = u.searchParams.get('session');
      if (s) w = w.filter((r) => r.session_id === s);
      return send(200, w.slice(-Math.min(parseInt(u.searchParams.get('limit') || '100', 10), 5000)));
    }
    case '/api/run': if (req.method === 'POST') { schedule('manual'); return send(202, { queued: true }); }
    // falls through
    default: return send(404, { error: 'not found' });
  }
}).listen(cfg.port, () => {
  console.log(`[node] watching ${DATA}`);
  console.log(`[node] API on http://localhost:${cfg.port}/api/status`);
  runAnalyzer('startup');
});
