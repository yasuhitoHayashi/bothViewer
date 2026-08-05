const $ = id => document.getElementById(id);
const count = Number(new URLSearchParams(location.search).get('count')) === 2 ? 2 : 1;
const API = 'http://127.0.0.1:5002';
let selected = null, sourceType = 'session', rawSourceId = null, manifests = [];
let index = 0, playing = false, animation = null, wallStart = 0, startUs = 0;
let globalStartNs = 0, totalDurationUs = 0, intervalUs = 33000;

$('modeTitle').textContent = count === 2 ? 'EVS 2台データ確認' : 'EVS 1台データ確認';
$('playbackGrid').style.setProperty('--count', count);
$('standaloneRaw').hidden = count !== 1;

async function json(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok || data.status === 'error') throw new Error(data.message || `HTTP ${response.status}`);
  return data;
}
function message(text) { $('message').textContent = text || ''; }

async function loadSessions() {
  try {
    message('');
    const data = await json(`${API}/recordings`);
    const kind = count === 2 ? 'evs_dual' : 'evs_single';
    const sessions = (data.sessions || []).filter(item => item.kind === kind);
    $('sessionList').replaceChildren();
    sessions.forEach(session => {
      const button = document.createElement('button');
      button.className = 'session-item';
      button.innerHTML = `<b>${session.session_id}</b><span>${session.duration_seconds.toFixed(2)} s · ${(session.evs_roles || []).join(' + ')}</span>`;
      button.addEventListener('click', () => selectSession(session.session_id));
      $('sessionList').append(button);
    });
    if (!sessions.length) message(count === 1 ? 'セッションがありません。上のボタンからRAWを直接開けます。' : '対象のEVSセッションがありません。');
  } catch (error) { message(error.message); }
}

function preparePlayer(title, detail, roles) {
  $('sessionTitle').textContent = title;
  $('detailText').textContent = detail;
  $('playbackGrid').replaceChildren();
  roles.forEach((role, i) => {
    const card = document.createElement('article');
    card.className = 'camera-card';
    card.innerHTML = `<div class="camera-head"><b>${role.label}</b><span>${role.source}</span></div><img id="evsImage${i}" class="camera-view" alt="${role.label}">`;
    $('playbackGrid').append(card);
  });
  $('timeline').max = Math.max(0, Math.ceil(totalDurationUs / intervalUs));
  $('timeline').value = 0;
  $('timeline').disabled = false;
  $('playButton').disabled = false;
  message('');
  render(0);
}

async function selectSession(id) {
  try {
    pause(); sourceType = 'session'; rawSourceId = null; selected = id;
    const detail = (await json(`${API}/recordings/${encodeURIComponent(id)}`)).session;
    const roles = count === 2 ? ['evs_a', 'evs_b'] : [detail.evs_sources?.evs ? 'evs' : Object.keys(detail.evs_sources || {})[0]];
    if (roles.some(role => !role)) throw new Error('EVSデータがありません。');
    manifests = await Promise.all(roles.map(role => json(`${API}/recordings/${encodeURIComponent(id)}/evs/${role}/playback`).then(data => data.playback)));
    const starts = manifests.map(item => item.started_utc_ns).filter(Boolean);
    globalStartNs = starts.length ? Math.min(...starts) : 0;
    manifests.forEach(item => { if (!item.started_utc_ns) item.started_utc_ns = globalStartNs; });
    intervalUs = manifests[0].interval_us || 33000;
    totalDurationUs = Math.max(...manifests.map(item => Math.round((item.started_utc_ns - globalStartNs) / 1000) + item.duration_us));
    preparePlayer(id, `${roles.join(' + ')} · ${(totalDurationUs / 1e6).toFixed(3)} s · host UTC同期`, roles.map(role => ({label: role.toUpperCase(), source: 'RAW'})));
  } catch (error) { message(error.message); }
}

async function openRawFile() {
  try {
    pause();
    message('RAWを解析しています。ファイルが大きい場合は少し時間がかかります…');
    const response = await fetch(`${API}/raw-files/dialog`, {method: 'POST'});
    const data = await response.json();
    if (!response.ok || data.status === 'error') throw new Error(data.message || `HTTP ${response.status}`);
    if (data.status === 'cancelled') { message(''); return; }
    const item = data.playback;
    sourceType = 'raw'; rawSourceId = item.source_id; selected = null;
    manifests = [{...item, role: 'raw', started_utc_ns: 0}];
    globalStartNs = 0; intervalUs = item.interval_us || 33000; totalDurationUs = item.duration_us;
    preparePlayer(item.filename, `単体RAW · ${(item.duration_us / 1e6).toFixed(3)} s · ${Number(item.event_count || 0).toLocaleString()} events · ${item.sensor_width}×${item.sensor_height}`, [{label: 'EVS', source: '単体RAW'}]);
  } catch (error) { message(error.message); }
}

function render(next) {
  if (!manifests.length) return;
  index = Math.max(0, Math.min(Number(next) || 0, Number($('timeline').max)));
  $('timeline').value = index;
  const elapsed = Math.min(totalDurationUs, index * intervalUs);
  manifests.forEach((manifest, i) => {
    const offset = Math.round((manifest.started_utc_ns - globalStartNs) / 1000);
    const roleElapsed = elapsed - offset;
    const segment = (manifest.segments || []).find(item => roleElapsed >= item.start_us && roleElapsed <= item.end_us) || (manifest.segments || [])[0] || {epoch: 0, start_us: 0};
    const roleCenter = Math.max(0, roleElapsed - segment.start_us);
    $(`evsImage${i}`).style.opacity = roleElapsed < 0 || roleElapsed > manifest.duration_us ? '.25' : '1';
    const query = `window_us=${$('windowUs').value}&width=960&palette=${$('eventPalette').value}`;
    $(`evsImage${i}`).src = sourceType === 'raw'
      ? `${API}/raw-files/${rawSourceId}/${roleCenter}.jpg?${query}`
      : `${API}/recordings/${encodeURIComponent(selected)}/evs/${manifest.role}/${segment.epoch}/${roleCenter}.jpg?${query}`;
  });
  $('timeText').textContent = `${(elapsed / 1e6).toFixed(3)} s`;
}

function pause() { playing = false; if (animation) cancelAnimationFrame(animation); animation = null; $('playButton').textContent = '▶ 再生'; }
function tick(now) { if (!playing) return; const target = startUs + (now - wallStart) * 1000; const next = Math.min(Number($('timeline').max), Math.floor(target / intervalUs)); if (next !== index) render(next); if (next >= Number($('timeline').max)) { pause(); return; } animation = requestAnimationFrame(tick); }
function toggle() { if (playing) { pause(); return; } if (index >= Number($('timeline').max)) render(0); playing = true; $('playButton').textContent = 'Ⅱ 一時停止'; wallStart = performance.now(); startUs = index * intervalUs; animation = requestAnimationFrame(tick); }

$('refreshButton').addEventListener('click', loadSessions);
$('openRawButton').addEventListener('click', openRawFile);
$('playButton').addEventListener('click', toggle);
$('timeline').addEventListener('input', () => { pause(); render($('timeline').value); });
$('windowUs').addEventListener('change', () => render(index));
$('eventPalette').addEventListener('change', () => render(index));
loadSessions();
