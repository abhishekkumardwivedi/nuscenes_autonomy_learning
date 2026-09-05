const $ = (id) => document.getElementById(id);

const appState = {
  stages: [],
  runtime: null,
  config: null,
  selectedStage: 0,
  leftTab: 'visual',
  rightTab: 'overview',
  ws: null,
  pingTimer: null,
  reconnectTimer: null,
  reconnectDelay: 1000,
  pc: null,
  webrtcActive: false,
  selectedArtifacts: [],
  selectedSummary: null,
};

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  let data = null;
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) data = await res.json();
  else data = await res.text();
  if (!res.ok) {
    const msg = data?.detail || data?.message || `${res.status} ${res.statusText}`;
    throw new Error(msg);
  }
  return data;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function statusLabel(s) {
  return {
    not_started: 'Not started', running: 'Running', completed: 'Completed',
    failed: 'Failed', warning: 'Warning / stopped', stale: 'Stale',
  }[s] || s || 'Unknown';
}

function toast(message, kind = '') {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = message;
  $('toastContainer').appendChild(el);
  setTimeout(() => el.remove(), 5200);
}

function selectedMeta() {
  return appState.stages.find(s => s.number === appState.selectedStage) || appState.stages[0];
}

function runtimeFor(number) {
  if (!appState.runtime) return null;
  return appState.runtime.stages.find(s => s.stage === number) || null;
}

function renderStageNavigator() {
  const nav = $('stageNavigator');
  nav.innerHTML = '';
  for (const stage of appState.stages) {
    const runtime = runtimeFor(stage.number) || stage.runtime || { status: 'not_started' };
    const btn = document.createElement('button');
    btn.className = `stage-chip ${runtime.status} ${stage.number === appState.selectedStage ? 'selected' : ''}`;
    btn.title = `Stage ${String(stage.number).padStart(2,'0')} — ${stage.title}\n${stage.dataset}`;
    btn.innerHTML = `<span class="num">${String(stage.number).padStart(2,'0')}</span><span class="name">${escapeHtml(stage.alias.replaceAll('_',' '))}</span><span class="state-dot"></span>`;
    btn.onclick = () => selectStage(stage.number);
    nav.appendChild(btn);
  }
  renderFooterProgress();
}

function renderFooterProgress() {
  const el = $('pipelineFooter');
  el.innerHTML = '';
  for (const stage of appState.stages) {
    const runtime = runtimeFor(stage.number) || { status: 'not_started' };
    const bar = document.createElement('div');
    bar.className = `footer-stage ${runtime.status}`;
    bar.title = `${String(stage.number).padStart(2,'0')} ${stage.title}: ${statusLabel(runtime.status)}`;
    el.appendChild(bar);
  }
}

async function selectStage(number) {
  appState.selectedStage = number;
  const meta = selectedMeta();
  $('selectedStageLabel').textContent = `Stage ${String(number).padStart(2,'0')} — ${meta.alias}`;
  $('selectedDataset').textContent = meta.dataset;
  $('viewerTitle').textContent = `Stage ${String(number).padStart(2,'0')} — ${meta.title}`;
  $('overviewDataset').textContent = meta.dataset;
  $('overviewTitle').textContent = meta.title;
  $('overviewConcept').textContent = meta.concept;
  renderChips('inputsList', meta.inputs);
  renderChips('outputsList', meta.outputs);
  renderRuntimeCard();
  renderStageNavigator();
  populateCodeFiles(meta);
  await Promise.allSettled([loadLogs(), loadSummary(), loadArtifacts()]);
  if (appState.leftTab === 'code') await loadCode();
  renderVisualForSelection();
}

function renderChips(id, items) {
  $(id).innerHTML = (items || []).map(x => `<span class="info-chip">${escapeHtml(x)}</span>`).join('');
}

function renderRuntimeCard() {
  const runtime = runtimeFor(appState.selectedStage) || { status: 'not_started', progress: 0, current_step: 'Waiting' };
  $('runtimeStage').textContent = `Stage ${String(appState.selectedStage).padStart(2,'0')}`;
  $('runtimeStatus').textContent = statusLabel(runtime.status);
  $('runtimeStatus').className = `state-badge ${runtime.status}`;
  $('progressBar').style.width = `${Math.max(0, Math.min(100, runtime.progress || 0))}%`;
  $('progressText').textContent = `${runtime.progress || 0}%`;
  $('currentStep').textContent = runtime.current_step || 'Waiting';
  const failed = runtime.status === 'failed' || Boolean(runtime.error);
  $('errorCard').classList.toggle('hidden', !failed);
  if (failed) {
    $('errorTitle').textContent = runtime.status === 'failed' ? 'Stage failed' : 'Stage warning';
    $('errorMessage').textContent = runtime.error || 'Unknown error';
  }
}

function renderGlobalRuntime() {
  if (!appState.runtime) return;
  const busy = appState.runtime.busy;
  $('stopBtn').disabled = !busy;
  $('runToBtn').disabled = busy;
  $('runStageBtn').disabled = busy;
  $('resetBtn').disabled = busy;
  renderRuntimeCard();
  renderStageNavigator();
}

async function loadLogs() {
  try {
    const logs = await api(`/api/stage/${appState.selectedStage}/logs`);
    renderLogs(logs);
  } catch (e) {
    renderLogs([]);
  }
}

function renderLogs(logs) {
  const view = $('logView');
  view.innerHTML = '';
  for (const item of logs || []) appendLogLine(item, false);
  if ($('autoScrollLogs').checked) view.scrollTop = view.scrollHeight;
}

function appendLogLine(item, scroll = true) {
  if (!item) return;
  const view = $('logView');
  const div = document.createElement('div');
  div.className = `log-line ${item.level || ''}`;
  const ts = item.ts ? new Date(item.ts).toLocaleTimeString() : new Date().toLocaleTimeString();
  div.innerHTML = `<span class="time">${escapeHtml(ts)}</span>${escapeHtml(item.message || '')}`;
  view.appendChild(div);
  if (scroll && $('autoScrollLogs').checked) view.scrollTop = view.scrollHeight;
}

async function loadSummary() {
  appState.selectedSummary = null;
  try {
    const data = await api(`/api/stage/${appState.selectedStage}/summary`);
    appState.selectedSummary = data;
    $('outputJson').textContent = JSON.stringify(data, null, 2);
    renderTensorCards(data);
  } catch (e) {
    $('outputJson').textContent = 'No summary yet. Run this stage first.';
    renderTensorCards(null);
  }
}

function findTensorLike(value, path = '', out = []) {
  if (!value || typeof value !== 'object') return out;
  if (!Array.isArray(value) && ('shape' in value) && (('dtype' in value) || ('elements' in value))) {
    out.push({ name: path || 'tensor', info: value });
    return out;
  }
  if (Array.isArray(value)) {
    value.slice(0, 30).forEach((v, i) => findTensorLike(v, `${path}[${i}]`, out));
  } else {
    for (const [k, v] of Object.entries(value)) {
      findTensorLike(v, path ? `${path}.${k}` : k, out);
    }
  }
  return out;
}

function renderTensorCards(summary) {
  const root = $('tensorCards');
  const tensors = findTensorLike(summary || {});
  const runtime = runtimeFor(appState.selectedStage);
  if (runtime?.last_tensor && !tensors.some(t => t.name === 'Latest live tensor')) {
    tensors.unshift({ name: 'Latest live tensor', info: runtime.last_tensor });
  }
  if (!tensors.length) {
    root.innerHTML = '<div class="empty-state">Run this stage to see tensor shapes and statistics.</div>';
    return;
  }
  root.innerHTML = tensors.slice(0, 50).map(t => {
    const entries = Object.entries(t.info).slice(0, 16);
    return `<div class="tensor-card"><div class="tensor-name">${escapeHtml(t.name)}</div><div class="tensor-grid">${entries.map(([k,v]) => `<div class="tensor-kv"><strong>${escapeHtml(k)}</strong><br>${escapeHtml(Array.isArray(v) ? JSON.stringify(v) : String(v))}</div>`).join('')}</div></div>`;
  }).join('');
}

async function loadArtifacts() {
  try {
    const artifacts = await api(`/api/stage/${appState.selectedStage}/artifacts`);
    appState.selectedArtifacts = artifacts;
    renderArtifactGrid(artifacts);
  } catch (e) {
    appState.selectedArtifacts = [];
    renderArtifactGrid([]);
  }
}

function artifactUrl(item) {
  return `/api/artifact?path=${encodeURIComponent(item.relative_path)}&v=${item.mtime_ns || Date.now()}`;
}

function renderArtifactGrid(items) {
  const root = $('artifactGrid');
  if (!items.length) {
    root.innerHTML = '<div class="empty-state">No saved artifacts for this stage yet.</div>';
    return;
  }
  root.innerHTML = items.map(item => {
    if (item.kind === 'image') {
      return `<button class="artifact-card" data-path="${escapeHtml(item.relative_path)}"><img src="${artifactUrl(item)}" alt="${escapeHtml(item.name)}"><div class="artifact-meta">${escapeHtml(item.name)}</div></button>`;
    }
    return `<div class="artifact-card"><div class="artifact-meta">${escapeHtml(item.name)}<br>${Math.round(item.size/1024)} KB</div></div>`;
  }).join('');
  root.querySelectorAll('button.artifact-card').forEach((btn, idx) => {
    btn.onclick = () => {
      const imageItems = items.filter(i => i.kind === 'image');
      const item = items.find(i => i.relative_path === btn.dataset.path) || imageItems[0];
      if (item) showFallbackImage(artifactUrl(item), `Saved artifact: ${item.name}`);
      setLeftTab('visual');
    };
  });
}

function choosePreferredImage() {
  const meta = selectedMeta();
  const images = appState.selectedArtifacts.filter(x => x.kind === 'image');
  if (!images.length) return null;
  return images.find(x => x.name === meta.preferred_visual) || images[images.length - 1];
}

function renderVisualForSelection() {
  const current = appState.runtime?.current_stage;
  if (appState.webrtcActive && current === appState.selectedStage) {
    $('visualShell').className = 'visual-shell webrtc';
    $('visualModeBadge').textContent = 'WebRTC live';
    $('visualPlaceholder').classList.add('hidden');
    $('visualCaption').textContent = 'Live visual stream from the currently executing stage.';
    return;
  }
  const item = choosePreferredImage();
  if (item) {
    showFallbackImage(artifactUrl(item), `Saved output: ${item.name}`);
  } else if (current === appState.selectedStage && !appState.webrtcActive) {
    showFallbackImage(`/api/visual/current?v=${Date.now()}`, 'Live snapshot fallback (WebRTC unavailable or not connected).');
  } else {
    $('visualShell').className = 'visual-shell';
    $('visualPlaceholder').classList.remove('hidden');
    $('visualModeBadge').textContent = 'No visual';
  }
}

function showFallbackImage(src, caption) {
  const img = $('fallbackImage');
  img.onload = () => {
    $('visualShell').className = 'visual-shell fallback';
    $('visualPlaceholder').classList.add('hidden');
    $('visualModeBadge').textContent = 'Saved / HTTP visual';
  };
  img.onerror = () => {
    $('visualShell').className = 'visual-shell';
    $('visualPlaceholder').classList.remove('hidden');
  };
  img.src = src;
  $('visualCaption').textContent = caption || '';
}

function populateCodeFiles(meta) {
  const select = $('codeFileSelect');
  select.innerHTML = (meta.code_files || []).map(f => `<option value="${escapeHtml(f)}">${escapeHtml(f)}</option>`).join('');
}

async function loadCode(file = null) {
  const meta = selectedMeta();
  const chosen = file || $('codeFileSelect').value || meta.code_files?.[0];
  if (!chosen) return;
  $('codeView').innerHTML = '<span class="code-line">Loading code…</span>';
  try {
    const data = await api(`/api/stage/${appState.selectedStage}/code?file=${encodeURIComponent(chosen)}`);
    if ($('codeFileSelect').value !== data.file) $('codeFileSelect').value = data.file;
    $('codeLineCount').textContent = `${data.content.split('\n').length} lines`;
    renderCode(data.content);
  } catch (e) {
    $('codeView').textContent = `Could not load code: ${e.message}`;
  }
}

function renderCode(content) {
  const lines = content.split('\n');
  $('codeView').innerHTML = lines.map(line => `<span class="code-line">${highlightPythonLine(line)}</span>`).join('');
  $('codeView').dataset.raw = content;
}

function highlightPythonLine(line) {
  let s = escapeHtml(line);
  const hash = s.indexOf('#');
  let comment = '';
  if (hash >= 0) {
    comment = `<span class="py-comment">${s.slice(hash)}</span>`;
    s = s.slice(0, hash);
  }
  s = s.replace(/\b(def|class|return|if|elif|else|for|while|in|import|from|as|try|except|finally|with|lambda|yield|raise|True|False|None|and|or|not|is|async|await|pass|break|continue)\b/g, '<span class="py-keyword">$1</span>');
  s = s.replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="py-number">$1</span>');
  s = s.replace(/^\s*(@[A-Za-z_][\w.]*)/, '<span class="py-decorator">$1</span>');
  return s + comment;
}

function setLeftTab(name) {
  appState.leftTab = name;
  document.querySelectorAll('#leftTabs .tab').forEach(x => x.classList.toggle('active', x.dataset.tab === name));
  document.querySelectorAll('.viewer-content > .tab-page').forEach(x => x.classList.toggle('active', x.id === `left-${name}`));
  $('workspace').classList.toggle('code-focus', name === 'code');
  if (name === 'code') loadCode();
  if (name === 'graphs') loadArtifacts();
  if (name === 'outputs') loadSummary();
  if (name === 'visual') renderVisualForSelection();
}

function setRightTab(name) {
  appState.rightTab = name;
  document.querySelectorAll('#rightTabs .tab').forEach(x => x.classList.toggle('active', x.dataset.tab === name));
  document.querySelectorAll('.info-content > .tab-page').forEach(x => x.classList.toggle('active', x.id === `right-${name}`));
  if (name === 'logs') loadLogs();
  if (name === 'tensor') loadSummary();
  if (name === 'params') renderParams();
}

async function run(mode) {
  try {
    await api('/api/run', {
      method: 'POST',
      body: JSON.stringify({ target_stage: appState.selectedStage, mode }),
    });
    toast(`${mode === 'run_to' ? 'Running to' : 'Running'} Stage ${String(appState.selectedStage).padStart(2,'0')}`, 'good');
  } catch (e) {
    toast(e.message, 'bad');
  }
}

async function stopRun() {
  try { await api('/api/stop', { method: 'POST' }); toast('Stop requested.', 'warn'); }
  catch (e) { toast(e.message, 'bad'); }
}

async function resetPipeline() {
  if (!confirm('Reset in-memory stage cache? Saved output files remain on disk.')) return;
  try {
    await api('/api/reset', { method: 'POST' });
    appState.runtime = await api('/api/state');
    renderGlobalRuntime();
    toast('Pipeline cache reset. Saved artifacts were kept.', 'good');
  } catch (e) { toast(e.message, 'bad'); }
}

function connectWebSocket() {
  if (appState.ws && [WebSocket.CONNECTING, WebSocket.OPEN].includes(appState.ws.readyState)) return;
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${location.host}/ws`);
  appState.ws = ws;
  ws.onopen = () => {
    appState.reconnectDelay = 1000;
    $('connectionBanner').classList.add('hidden');
    setBackendStatus(true);
    clearInterval(appState.pingTimer);
    appState.pingTimer = setInterval(() => { if (ws.readyState === WebSocket.OPEN) ws.send('ping'); }, 20000);
  };
  ws.onmessage = (ev) => {
    try { handleEvent(JSON.parse(ev.data)); } catch (_) {}
  };
  ws.onclose = () => {
    setBackendStatus(false);
    $('connectionBanner').classList.remove('hidden');
    clearInterval(appState.pingTimer);
    clearTimeout(appState.reconnectTimer);
    appState.reconnectTimer = setTimeout(connectWebSocket, appState.reconnectDelay);
    appState.reconnectDelay = Math.min(10000, appState.reconnectDelay * 1.5);
  };
  ws.onerror = () => ws.close();
}

function handleEvent(event) {
  if (event.type === 'hello') {
    appState.runtime = event.state;
    renderGlobalRuntime();
    return;
  }
  if (event.state && event.state.stages) {
    appState.runtime = event.state;
  } else if (event.type === 'stage_state' && event.state && appState.runtime) {
    const idx = appState.runtime.stages.findIndex(x => x.stage === event.stage);
    if (idx >= 0) appState.runtime.stages[idx] = event.state;
    appState.runtime.current_stage = event.state.status === 'running' ? event.stage : appState.runtime.current_stage;
  } else if (event.type === 'pipeline_event' && event.state && appState.runtime) {
    const idx = appState.runtime.stages.findIndex(x => x.stage === event.stage);
    if (idx >= 0) appState.runtime.stages[idx] = event.state;
  }

  if (event.type === 'run_started') {
    if (!appState.runtime) return;
    appState.runtime.busy = true;
    appState.runtime.current_stage = event.start;
  }
  if (['run_completed','run_failed','run_stopped'].includes(event.type)) {
    if (event.state) appState.runtime = event.state;
    if (appState.runtime) appState.runtime.busy = false;
    if (event.type === 'run_failed') toast(event.error || 'Pipeline failed', 'bad');
    if (event.type === 'run_stopped') toast('Pipeline stopped.', 'warn');
    if (event.type === 'run_completed') toast(`Execution completed through Stage ${String(event.target).padStart(2,'0')}.`, 'good');
  }
  if (event.type === 'pipeline_reset' && event.state) appState.runtime = event.state;
  if (event.type === 'downstream_stale') toast(`Stages ${String(event.from_stage).padStart(2,'0')}+ marked stale because an upstream stage changed.`, 'warn');
  if (event.type === 'already_completed') toast(event.message, 'warn');

  if (event.type === 'pipeline_event' && event.stage === appState.selectedStage) {
    appendLogLine({ level: event.level, message: event.message });
    if (event.kind === 'tensor') {
      const rt = runtimeFor(event.stage);
      if (rt) rt.last_tensor = event.tensor_info;
      if (appState.rightTab === 'tensor') renderTensorCards(appState.selectedSummary || {});
    }
  }
  if (event.type === 'visual_updated' && event.stage === appState.selectedStage) {
    loadArtifacts().then(renderVisualForSelection);
  }
  if (event.type === 'run_completed' || event.type === 'run_failed') {
    loadSummary();
    loadArtifacts().then(renderVisualForSelection);
    loadLogs();
  }
  renderGlobalRuntime();
  renderVisualForSelection();
}

function setBackendStatus(ok) {
  const el = $('backendStatus');
  el.className = `status-pill ${ok ? 'good' : 'bad'}`;
  el.innerHTML = `<span class="dot"></span>${ok ? 'Backend connected' : 'Backend offline'}`;
}

function setWebRTCStatus(kind, text) {
  const el = $('webrtcStatus');
  el.className = `status-pill ${kind}`;
  el.innerHTML = `<span class="dot"></span>${escapeHtml(text)}`;
}

async function waitIceComplete(pc, timeoutMs = 3500) {
  if (pc.iceGatheringState === 'complete') return;
  await Promise.race([
    new Promise(resolve => {
      const f = () => { if (pc.iceGatheringState === 'complete') { pc.removeEventListener('icegatheringstatechange', f); resolve(); } };
      pc.addEventListener('icegatheringstatechange', f);
    }),
    new Promise(resolve => setTimeout(resolve, timeoutMs)),
  ]);
}

async function startWebRTC() {
  try {
    const health = await api('/api/health');
    if (!health.webrtc_available) throw new Error(health.webrtc_import_error || 'WebRTC packages not installed');
    const pc = new RTCPeerConnection();
    appState.pc = pc;
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.ontrack = (evt) => {
      $('webrtcVideo').srcObject = evt.streams[0];
    };
    pc.onconnectionstatechange = () => {
      if (pc.connectionState === 'connected') {
        appState.webrtcActive = true;
        setWebRTCStatus('good', 'WebRTC connected');
        renderVisualForSelection();
      } else if (['failed','disconnected','closed'].includes(pc.connectionState)) {
        appState.webrtcActive = false;
        setWebRTCStatus('warn', 'WebRTC fallback');
        renderVisualForSelection();
      }
    };
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await waitIceComplete(pc);
    const answer = await api('/api/webrtc/offer', {
      method: 'POST',
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
    });
    await pc.setRemoteDescription(answer);
    setWebRTCStatus('neutral', 'WebRTC connecting');
    setTimeout(() => {
      if (!appState.webrtcActive) {
        setWebRTCStatus('warn', 'WebRTC fallback');
        renderVisualForSelection();
      }
    }, 5000);
  } catch (e) {
    appState.webrtcActive = false;
    setWebRTCStatus('warn', 'WebRTC fallback');
    $('visualModeBadge').textContent = 'HTTP fallback';
  }
}

async function loadSystemInfo() {
  try {
    const data = await api('/api/system');
    const el = $('deviceStatus');
    if (data.cuda_available) {
      el.textContent = `${data.gpu_name || data.device} • ${data.allocated_mb || 0} MB alloc`;
      el.className = 'status-pill good';
    } else {
      el.textContent = `Device: ${data.device || 'CPU'}`;
      el.className = 'status-pill neutral';
    }
  } catch (_) {}
}

function renderParams() {
  $('paramsJson').textContent = JSON.stringify(appState.config || appState.runtime?.config || {}, null, 2);
}

function openSettings() {
  const c = appState.config || appState.runtime?.config || {};
  $('cfgDataroot').value = c.dataroot || '';
  $('cfgVersion').value = c.version || 'v1.0-mini';
  $('cfgScene').value = c.scene_index ?? 0;
  $('cfgSample').value = c.sample_index ?? -1;
  $('cfgHistory').value = c.history_frames ?? 4;
  $('cfgFuture').value = c.future_frames ?? 6;
  $('cfgDevice').value = c.device || 'auto';
  $('cfgTemporal').value = c.temporal_model || 'ema';
  $('cfgPlanner').value = c.planner_mode || 'classical';
  $('cfgVerbose').value = String(c.verbose ?? 2);
  $('settingsModal').classList.remove('hidden');
}

async function saveSettings() {
  const body = {
    dataroot: $('cfgDataroot').value.trim(), version: $('cfgVersion').value.trim(),
    scene_index: Number($('cfgScene').value), sample_index: Number($('cfgSample').value),
    history_frames: Number($('cfgHistory').value), future_frames: Number($('cfgFuture').value),
    device: $('cfgDevice').value.trim(), temporal_model: $('cfgTemporal').value,
    planner_mode: $('cfgPlanner').value, verbose: Number($('cfgVerbose').value),
  };
  try {
    appState.config = await api('/api/config', { method: 'PATCH', body: JSON.stringify(body) });
    appState.runtime = await api('/api/state');
    $('settingsModal').classList.add('hidden');
    renderGlobalRuntime(); renderParams();
    toast('Configuration saved; in-memory stage cache reset.', 'good');
  } catch (e) { toast(e.message, 'bad'); }
}

function toggleFullScreen() {
  const target = $('viewer-panel') || document.querySelector('.viewer-panel');
  document.querySelector('.viewer-panel').classList.toggle('fullscreen-panel');
}

function wireUi() {
  document.querySelectorAll('#leftTabs .tab').forEach(btn => btn.onclick = () => setLeftTab(btn.dataset.tab));
  document.querySelectorAll('#rightTabs .tab').forEach(btn => btn.onclick = () => setRightTab(btn.dataset.tab));
  $('runToBtn').onclick = () => run('run_to');
  $('runStageBtn').onclick = () => run('run_stage');
  $('stopBtn').onclick = stopRun;
  $('resetBtn').onclick = resetPipeline;
  $('retryStageBtn').onclick = () => run('run_stage');
  $('refreshSummaryBtn').onclick = loadSummary;
  $('refreshArtifactsBtn').onclick = () => loadArtifacts().then(renderVisualForSelection);
  $('clearLogViewBtn').onclick = () => $('logView').innerHTML = '';
  $('codeFileSelect').onchange = () => loadCode($('codeFileSelect').value);
  $('copyCodeBtn').onclick = async () => {
    try { await navigator.clipboard.writeText($('codeView').dataset.raw || ''); toast('Code copied.', 'good'); }
    catch (_) { toast('Clipboard access was not available.', 'warn'); }
  };
  $('settingsBtn').onclick = openSettings;
  $('closeSettingsBtn').onclick = () => $('settingsModal').classList.add('hidden');
  $('saveSettingsBtn').onclick = saveSettings;
  $('fullScreenBtn').onclick = toggleFullScreen;
  $('settingsModal').onclick = (e) => { if (e.target === $('settingsModal')) $('settingsModal').classList.add('hidden'); };
}

async function init() {
  wireUi();
  try {
    const [stages, state, config] = await Promise.all([
      api('/api/stages'), api('/api/state'), api('/api/config')
    ]);
    appState.stages = stages;
    appState.runtime = state;
    appState.config = config;
    renderParams();
    await selectStage(0);
    renderGlobalRuntime();
    setBackendStatus(true);
  } catch (e) {
    toast(`Dashboard initialization failed: ${e.message}`, 'bad');
    setBackendStatus(false);
  }
  connectWebSocket();
  startWebRTC();
  loadSystemInfo();
  setInterval(loadSystemInfo, 5000);
  // When WebRTC cannot traverse the RunPod network, this keeps the visual panel
  // responsive through the same HTTPS port with no additional infrastructure.
  setInterval(() => {
    if (!appState.webrtcActive && appState.runtime?.current_stage === appState.selectedStage) {
      showFallbackImage(`/api/visual/current?v=${Date.now()}`, 'Live HTTP snapshot fallback.');
    }
  }, 1200);
}

init();
