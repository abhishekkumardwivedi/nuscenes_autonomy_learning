/* Infrastructure UI stays separate from stage selection and educational rendering. */
(() => {
  const bytes = n => n == null ? 'N/A' : n >= 1024**3 ? `${(n/1024**3).toFixed(2)} GB` : `${(n/1024**2).toFixed(1)} MB`;
  const number = (n, unit = '') => n == null || !Number.isFinite(n) ? 'N/A' : `${n.toFixed(1)}${unit}`;
  let profiles = [], frames = [], previousOutput = null, lastTensorStage = null;
  let terminal = null, fit = null, socket = null, reconnectTimer = null, consoleWanted = false;
  let sessionId = sessionStorage.getItem('autonomy.console') || '';
  let series = [];
  const charts = [
    ['GPU utilization %', s => s.gpus?.[0]?.utilization, 100],
    ['VRAM used MB (GPU 0)', s => s.gpus?.[0]?.used == null ? null : s.gpus[0].used/1024**2],
    ['CPU %', s => s.cpu_percent, 100],
    ['RAM %', s => s.ram?.percent, 100],
    ['Torch allocated MB', s => s.torch?.allocated == null ? null : s.torch.allocated/1024**2],
    ['Torch reserved MB', s => s.torch?.reserved == null ? null : s.torch.reserved/1024**2],
  ];
  $('hardwareCharts').innerHTML = charts.map((c,i)=>`<figure><figcaption>${c[0]}</figcaption><canvas id="hardwareChart${i}" class="resource-chart" role="img" aria-label="${c[0]} over time"></canvas></figure>`).join('');
  function graph(canvas, points, maximum, bars = false) {
    if (!canvas || !canvas.clientWidth) return;
    const width = canvas.clientWidth, height = 110, ratio = window.devicePixelRatio || 1;
    canvas.width = width*ratio; canvas.height = height*ratio;
    const ctx = canvas.getContext('2d'); ctx.scale(ratio,ratio);
    ctx.fillStyle = '#080d19'; ctx.fillRect(0,0,width,height);
    const valid = points.filter(x=>x != null && Number.isFinite(x));
    const max = maximum || Math.max(1,...valid)*1.1;
    ctx.fillStyle = '#91a0b8'; ctx.font = '10px system-ui';
    ctx.fillText(valid.length ? number(valid.at(-1)) : 'N/A', 5,12);
    ctx.strokeStyle = '#22d3ee'; ctx.fillStyle = '#60a5fa'; ctx.lineWidth = 1.5;
    ctx.beginPath(); let connected = false;
    points.forEach((value,i)=> {
      if (value == null || !Number.isFinite(value)) { connected = false; return; }
      const x = 8+i*(width-16)/Math.max(1,points.length-1), y = height-8-value/max*(height-28);
      if (bars) ctx.fillRect(x, y, Math.max(2,(width-16)/points.length-3),height-8-y);
      else if (connected) ctx.lineTo(x,y); else ctx.moveTo(x,y);
      connected = true;
    });
    ctx.stroke();
  }
  function hardware(data) {
    series = data.history || [];
    const s = data.latest;
    if (!s) return;
    const rows = [
      ['CPU',number(s.cpu_percent,'%')], ['RAM',`${bytes(s.ram?.used)} / ${bytes(s.ram?.total)} (${number(s.ram?.percent,'%')})`],
      ['RAM available',bytes(s.ram?.available)], ['Storage',`${bytes(s.storage?.used)} / ${bytes(s.storage?.total)}`],
      ['Storage available',bytes(s.storage?.free)], ['Network RX / TX',`${bytes(s.network?.rx_bytes_sec)}/s / ${bytes(s.network?.tx_bytes_sec)}/s`],
      ['CUDA',s.torch?.cuda_available ? `Available · device ${s.torch.device}` : 'Unavailable'],
      ['Torch allocated / reserved',`${bytes(s.torch?.allocated)} / ${bytes(s.torch?.reserved)}`],
      ['Torch peak allocated / reserved',`${bytes(s.torch?.peak_allocated)} / ${bytes(s.torch?.peak_reserved)}`],
    ];
    for (const gpu of s.gpus || []) {
      rows.push([`GPU ${gpu.index}`,gpu.name],['GPU utilization',number(gpu.utilization,'%')],
        ['VRAM used / total / free',`${bytes(gpu.used)} / ${bytes(gpu.total)} / ${bytes(gpu.free)}`],
        ['Temperature / power / clock',`${number(gpu.temperature,' °C')} / ${number(gpu.power_mw == null ? null : gpu.power_mw/1000,' W')} / ${number(gpu.clock_mhz,' MHz')}`]);
    }
    if (!(s.gpus || []).length) rows.push(['NVML',s.nvml_error || 'N/A']);
    $('hardwareValues').innerHTML = `<dl class="metric-grid">${rows.map(([k,v])=>`<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`).join('')}</dl>`;
    charts.forEach((c,i)=>graph($(`hardwareChart${i}`), series.map(c[1]), c[2]));
  }
  function profileViews() {
    const selected = profiles.find(p=>p.stage===appState.selectedStage);
    $('stageProfile').innerHTML = selected ? `<dl class="metric-grid">
      <dt>Execution</dt><dd>${number(selected.elapsed_ms,' ms')}</dd>
      <dt>CPU average</dt><dd>${number(selected.cpu_percent_avg,'%')}</dd>
      <dt>Process CPU</dt><dd>${number(selected.process_cpu_percent,'%')}</dd>
      <dt>Torch before / after</dt><dd>${bytes(selected.torch_before?.allocated)} / ${bytes(selected.torch_after?.allocated)}</dd>
      <dt>Torch peak</dt><dd>${bytes(selected.torch_after?.peak_allocated)}</dd>
      <dt>Torch reserved / peak</dt><dd>${bytes(selected.torch_after?.reserved)} / ${bytes(selected.torch_after?.peak_reserved)}</dd>
      <dt>Stage output storage</dt><dd>${bytes(selected.output_memory_bytes)}</dd>
      ${(selected.gpus||[]).map(g=>`<dt>GPU ${g.index} average</dt><dd>${number(g.utilization_avg,'%')}</dd><dt>VRAM before / peak / after</dt><dd>${bytes(g.before)} / ${bytes(g.sampled_peak)} / ${bytes(g.after)}</dd>`).join('')}
      <dt>Started / ended</dt><dd>${escapeHtml(selected.started_at)}<br>${escapeHtml(selected.ended_at)}</dd>
      </dl><p>${escapeHtml(selected.notes)}</p><pre class="json-view small">${escapeHtml(JSON.stringify(selected,null,2))}</pre>` : 'N/A — run the selected stage to capture a profile.';
    $('stageComparison').innerHTML = `<table class="profile-table"><thead><tr><th>Stage</th><th>Time ms</th><th>GPU %</th><th>Peak VRAM</th><th>Torch peak</th><th>Outputs</th></tr></thead><tbody>${profiles.map(p=>`<tr><td><button class="small-btn" data-profile-stage="${p.stage}">${String(p.stage).padStart(2,'0')} ${escapeHtml(appState.stages.find(s=>s.number===p.stage)?.alias || '')}</button> ${escapeHtml(p.runtime_status || '')}</td><td>${number(p.elapsed_ms)}</td><td>${number(p.gpus?.[0]?.utilization_avg)}</td><td>${bytes(p.gpus?.[0]?.sampled_peak)}</td><td>${bytes(p.torch_after?.peak_allocated)}</td><td>${bytes(p.output_memory_bytes)}</td></tr>`).join('')}</tbody></table>`;
    $('stageComparison').querySelectorAll('[data-profile-stage]').forEach(b=>b.onclick=()=>selectStage(Number(b.dataset.profileStage)));
    graph($('stageTimeGraph'),profiles.map(p=>p.elapsed_ms),null,true);
  }
  async function scenes() {
    try {
      const list = await api('/api/scenes');
      $('sceneSelect').innerHTML = list.map(s=>`<option value="${s.index}">${escapeHtml(s.name)} (${s.frames} frames)</option>`).join('');
      if (appState.runtime?.config?.scene_index != null) $('sceneSelect').value = appState.runtime.config.scene_index;
      await loadFrames();
    } catch(e) {
      $('playbackStatus').textContent = e.message;
      $('sceneSelect').innerHTML = '<option value="">Dataset missing</option>';
    }
  }
  async function loadFrames() {
    frames = await api(`/api/scenes/${$('sceneSelect').value}/frames`);
    $('frameScrubber').max = Math.max(0,frames.length-1);
  }
  async function playbackCommand(action) {
    try {
      const result = await api('/api/playback',{method:'POST',body:JSON.stringify({action, scene:Number($('sceneSelect').value),
        frame:Number($('frameScrubber').value), target_stage:appState.selectedStage, speed:Number($('playbackSpeed').value)})});
      playbackView(result);
    } catch(e) { toast(e.message,'bad'); }
  }
  function playbackView(state) {
    if (!state) return;
    if (document.activeElement !== $('frameScrubber')) $('frameScrubber').value = state.frame;
    $('frameLabel').textContent = `Frame ${state.frame} / ${Math.max(0,frames.length-1)}${state.timestamp ? ' · '+new Date(state.timestamp/1000).toISOString().slice(11,23) : ''}`;
    $('playbackStatus').textContent = state.error ? state.error.split('\n').filter(Boolean).at(-1) : state.processing ? 'Computing frame…' : state.playing ? 'Playing · compute-paced' : state.cache_hit ? 'Cached frame' : 'Paused';
    $('playbackStatus').title = state.error || 'First pass computes each frame. Cached playback follows recorded timestamp spacing.';
  }
  async function connectConsole() {
    clearTimeout(reconnectTimer);
    const status = await api('/api/console');
    if (!status.enabled) { $('consoleStatus').textContent=status.message; return; }
    consoleWanted = true;
    if (!terminal) {
      terminal = new Terminal({cursorBlink:true,scrollback:5000,fontSize:12,theme:{background:'#080d19'}});
      fit = new FitAddon.FitAddon(); terminal.loadAddon(fit); terminal.open($('terminalHost'));
      terminal.onData(data=>{if(socket?.readyState===1)socket.send(JSON.stringify({type:'input',data}));});
      terminal.onResize(size=>{if(socket?.readyState===1)socket.send(JSON.stringify({type:'resize',...size}));});
      new ResizeObserver(()=>{if($('terminalHost').clientWidth && $('terminalHost').clientHeight) fit.fit();}).observe($('terminalHost'));
    }
    if (socket) { socket.onclose=null; socket.close(); }
    terminal.reset(); fit.fit();
    const scheme = location.protocol==='https:' ? 'wss':'ws';
    socket = new WebSocket(`${scheme}://${location.host}/ws/console?session=${encodeURIComponent(sessionId)}`);
    socket.onopen=()=>{
      $('consoleStatus').textContent='Console connected';
      fit.fit(); socket.send(JSON.stringify({type:'resize',cols:terminal.cols,rows:terminal.rows}));
      terminal.focus();
    };
    socket.onmessage=event=>{
      const msg=JSON.parse(event.data);
      if(msg.type==='session') {sessionId=msg.id;sessionStorage.setItem('autonomy.console',sessionId);}
      if(msg.type==='output')terminal.write(msg.data);
      if(msg.type==='exit') { consoleWanted=false;sessionId='';sessionStorage.removeItem('autonomy.console');$('consoleStatus').textContent='Shell exited. Connect to start another.'; }
    };
    socket.onclose=()=>{
      if(consoleWanted) {$('consoleStatus').textContent='Console reconnecting…';reconnectTimer=setTimeout(()=>connectConsole().catch(e=>{$('consoleStatus').textContent=e.message;}),1500);}
    };
  }
  $('connectConsoleBtn').onclick=()=>connectConsole().catch(e=>toast(e.message,'bad'));
  $('interruptConsoleBtn').onclick=()=>{if(socket?.readyState===1)socket.send(JSON.stringify({type:'input',data:'\u0003'}));};
  $('clearConsoleBtn').onclick=()=>terminal?.clear();
  $('sceneSelect').onchange=()=>loadFrames().then(()=>{$('frameScrubber').value=0;return playbackCommand('seek');}).catch(e=>toast(e.message,'bad'));
  $('playSceneBtn').onclick=()=>playbackCommand('play');
  $('pauseSceneBtn').onclick=()=>playbackCommand('pause');
  $('previousFrameBtn').onclick=()=>playbackCommand('previous');
  $('nextFrameBtn').onclick=()=>playbackCommand('next');
  $('frameScrubber').onchange=()=>playbackCommand('seek');
  $('frameScrubber').oninput=()=>{$('frameLabel').textContent=`Frame ${$('frameScrubber').value} / ${Math.max(0,frames.length-1)}`;};
  window.DashboardExtensions={tab(name){
    if(name==='console'&&!terminal)connectConsole().catch(e=>toast(e.message,'bad'));
    if(name==='console')requestAnimationFrame(()=>fit?.fit());
    if(name==='graphs'||name==='profiler')profileViews();
    if(name==='hardware')charts.forEach((c,i)=>graph($(`hardwareChart${i}`),series.map(c[1]),c[2]));
  }};
  async function poll() {
    try {
      const [hw,state,ps] = await Promise.all([api('/api/hardware'),api('/api/state'),api('/api/profiles')]);
      hardware(hw); profiles=ps; profileViews();
      appState.runtime=state; renderGlobalRuntime(); playbackView(state.playback);
      const runtime=runtimeFor(appState.selectedStage);
      const elapsed=runtime?.started_at ? ((runtime.ended_at ? Date.parse(runtime.ended_at):Date.now())-Date.parse(runtime.started_at))/1000:0;
      if(runtime?.started_at)$('progressText').textContent=`${runtime.progress||0}% · ${elapsed.toFixed(1)}s`;
      if(previousOutput!==state.config.output_dir){
        previousOutput=state.config.output_dir;
        appState.config=state.config;
        await Promise.allSettled([loadArtifacts(),loadSummary(),loadLogs()]);
        renderVisualForSelection();
      }
      if(appState.rightTab==='tensor') {
        const tensors=await api(`/api/stage/${appState.selectedStage}/tensors`);
        if(tensors.length)renderTensorCards(Object.fromEntries(tensors.map(t=>[t.name,t])));
      }
    } catch(e) { /* The existing WebSocket indicator owns connection status. */ }
    setTimeout(poll,1000); // No overlapping polls when the proxy is slow.
  }
  scenes(); poll();
})();
