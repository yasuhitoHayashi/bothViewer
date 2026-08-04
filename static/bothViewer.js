const API = { evs: 'http://127.0.0.1:5001', frame: 'http://127.0.0.1:5002' };
let intervalTimer = null;
let recordingTimeout = null;
let recording = false;
let stoppingRecording = false;
let activeSessionId = null;
let triggerOptionsLoaded = false;
let previousFrameOnline = false;
let previousEvsOnline = false;
let previousAutoFallbackCount = 0;
let previousEvsPreviewFallbacks = 0;
let previousFramePreviewFallbacks = 0;
let capturePageActive = true;
let recordingSessions = [];
let selectedRecordingId = null;
let selectedRecordingPath = '';
let playbackManifest = null;
let playbackIndex = 0;
let playbackRunning = false;
let playbackAnimationFrame = null;
let playbackWallStart = 0;
let playbackTimelineStart = 0;
let lastPlaybackEvsMs = -Infinity;
let lastPlaybackEvsEpoch = null;

const $ = (id) => document.getElementById(id);

function showToast(message, type = 'success') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  $('toastRegion').appendChild(toast);
  window.setTimeout(() => toast.remove(), 3800);
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { data = { message: `HTTP ${response.status}` }; }
  if (!response.ok || data.status === 'error') throw new Error(data.message || `HTTP ${response.status}`);
  return data;
}

const post = (url, body) => requestJson(url, {
  method: 'POST',
  headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
  body: body === undefined ? undefined : JSON.stringify(body)
});

async function runAction(button, action) {
  button.disabled = true;
  try { await action(); } catch (error) { showToast(error.message || '操作に失敗しました。', 'error'); }
  finally { button.disabled = false; }
}

function value(id) { return $(id).value.trim(); }
function finiteNumber(id, label) {
  const rawValue = value(id);
  if (!rawValue) throw new Error(`${label}を入力してください。`);
  const number = Number(rawValue);
  if (!Number.isFinite(number)) throw new Error(`${label}を入力してください。`);
  return number;
}

function recordsDirectoryBesideHtml() {
  let path = decodeURIComponent(window.location.pathname || '');
  // file:///C:/... is exposed with a leading slash even on Windows.
  if (/^\/[A-Za-z]:\//.test(path)) path = path.slice(1);
  const directory = path.slice(0, path.lastIndexOf('/'));
  return directory ? `${directory}/records` : './records';
}

async function initializeSaveLocation() {
  $('save_location').value = recordsDirectoryBesideHtml();
  try {
    const data = await requestJson(`${API.frame}/recordings`);
    if (data.save_location) $('save_location').value = data.save_location;
  } catch { /* サーバー起動直後はHTML位置から求めた既定値を使う */ }
}

function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1000) return `${value} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let scaled = value;
  let unit = 'B';
  for (const candidate of units) {
    scaled /= 1000;
    unit = candidate;
    if (scaled < 1000) break;
  }
  return `${scaled.toFixed(scaled >= 100 ? 0 : scaled >= 10 ? 1 : 2)} ${unit}`;
}

function formatSessionDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '日時不明';
  return date.toLocaleString('ja-JP', {
    month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
    second: '2-digit', hour12: false
  });
}

function setActivePage(page) {
  const showData = page === 'data';
  capturePageActive = !showData;
  $('capturePage').hidden = showData;
  $('dataPage').hidden = !showData;
  $('captureTabButton').classList.toggle('active', !showData);
  $('dataTabButton').classList.toggle('active', showData);
  $('captureTabButton').toggleAttribute('aria-current', !showData);
  $('dataTabButton').toggleAttribute('aria-current', showData);
  if (showData) loadRecordings();
  else {
    pausePlayback();
    reconnectVideo('evsVideo', `${API.evs}/video_feed`);
    reconnectVideo('frameVideo', `${API.frame}/video_feed`);
  }
  syncCaptureActivity();
}

function isCaptureUiActive() {
  return capturePageActive && document.visibilityState === 'visible';
}

async function syncCaptureActivity() {
  const active = isCaptureUiActive();
  await Promise.allSettled([
    post(`${API.evs}/set_capture_active`, { active }),
    post(`${API.frame}/set_capture_active`, { active })
  ]);
}

function connectionStateLabel(connection) {
  if (connection.state === 'retry_paused') return '待機（撮影タブ外）';
  return connection.state || 'offline';
}

function renderRecordingList() {
  const query = value('recordingSearch').toLowerCase();
  const filtered = recordingSessions.filter((session) =>
    `${session.session_id} ${session.started_utc}`.toLowerCase().includes(query));
  $('recordingList').replaceChildren();
  if (!filtered.length) {
    const empty = document.createElement('div');
    empty.className = 'session-list-empty';
    empty.textContent = recordingSessions.length ? '条件に一致するセッションはありません。' : '保存済みセッションはありません。';
    $('recordingList').appendChild(empty);
    return;
  }
  for (const session of filtered) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `session-item${session.session_id === selectedRecordingId ? ' active' : ''}`;
    const heading = document.createElement('span');
    heading.className = 'session-item-heading';
    const title = document.createElement('strong');
    title.textContent = session.session_id;
    const date = document.createElement('time');
    date.textContent = formatSessionDate(session.started_utc);
    heading.append(title, date);
    const meta = document.createElement('span');
    meta.className = 'session-item-meta';
    meta.textContent = `${Number(session.duration_seconds || 0).toFixed(1)} s · ${session.saved_frames || 0} frames · edge ${session.reference_edges || 0}`;
    const badge = document.createElement('span');
    const lossTotal = Number(session.losses?.total || 0);
    badge.className = `quality-badge${lossTotal ? ' loss' : (!session.complete ? ' incomplete' : '')}`;
    badge.textContent = lossTotal ? `要確認 ${lossTotal}件` : (session.complete ? '欠損なし' : '記録未完了');
    button.append(heading, meta, badge);
    button.addEventListener('click', () => selectRecording(session.session_id));
    $('recordingList').appendChild(button);
  }
}

async function loadRecordings() {
  $('recordingRootMeta').textContent = '保存先を読み込み中…';
  try {
    const data = await requestJson(`${API.frame}/recordings`);
    recordingSessions = data.sessions || [];
    $('recordingRootMeta').textContent = `${data.save_location} · ${recordingSessions.length} セッション`;
    renderRecordingList();
    const selectedStillExists = recordingSessions.some((item) => item.session_id === selectedRecordingId);
    if (!selectedStillExists) selectedRecordingId = null;
    if (!selectedRecordingId && recordingSessions.length) {
      await selectRecording(recordingSessions[0].session_id);
    } else if (!recordingSessions.length) {
      $('dataEmpty').hidden = false;
      $('dataDetail').hidden = true;
    }
  } catch (error) {
    $('recordingRootMeta').textContent = '保存先を取得できません';
    $('recordingList').replaceChildren();
    const empty = document.createElement('div');
    empty.className = 'session-list-empty';
    empty.textContent = error.message;
    $('recordingList').appendChild(empty);
  }
}

function pausePlayback() {
  playbackRunning = false;
  if (playbackAnimationFrame !== null) window.cancelAnimationFrame(playbackAnimationFrame);
  playbackAnimationFrame = null;
  $('playbackPlayButton').textContent = '▶ 再生';
}

function resetPlayback() {
  pausePlayback();
  playbackManifest = null;
  playbackIndex = 0;
  lastPlaybackEvsMs = -Infinity;
  lastPlaybackEvsEpoch = null;
  $('playbackPlaceholder').hidden = false;
  $('playbackContent').hidden = true;
  $('playbackMessage').textContent = 'EVS RAWと保存フレームを同じ時刻で再生します。';
  $('preparePlaybackButton').hidden = false;
  $('playbackEvsImage').removeAttribute('src');
  $('playbackFrameImage').removeAttribute('src');
}

function playbackEvsSettings() {
  return {
    light: { fps: 5, maxEvents: 20000 },
    standard: { fps: 10, maxEvents: 50000 },
    dense: { fps: 15, maxEvents: 100000 },
    all: { fps: 0, maxEvents: 0 }
  }[$('playbackEvsLoad').value] || { fps: 10, maxEvents: 50000 };
}

function renderPlaybackFrame(index, forceEvs = false) {
  if (!playbackManifest?.frames?.length) return;
  playbackIndex = Math.max(0, Math.min(Number(index) || 0, playbackManifest.frames.length - 1));
  const frame = playbackManifest.frames[playbackIndex];
  const sessionId = playbackManifest.session_id;
  const windowUs = Number($('playbackWindow').value) || playbackManifest.event_window_us || 33000;
  $('playbackSlider').value = playbackIndex;
  $('playbackFrameImage').src = `${API.frame}/recordings/${encodeURIComponent(sessionId)}/preview/${encodeURIComponent(frame.filename)}`;
  const evsSettings = playbackEvsSettings();
  const evsIntervalMs = evsSettings.fps ? 1000 / evsSettings.fps : 0;
  const updateEvs = forceEvs || !playbackRunning || frame.stream_epoch !== lastPlaybackEvsEpoch ||
    frame.relative_ms - lastPlaybackEvsMs >= evsIntervalMs - 0.5;
  if (updateEvs) {
    $('playbackEvsImage').src = `${API.frame}/recordings/${encodeURIComponent(sessionId)}/events/${frame.stream_epoch}/${frame.raw_time_us}.png?window_us=${windowUs}&width=960&max_events=${evsSettings.maxEvents}`;
    lastPlaybackEvsMs = frame.relative_ms;
    lastPlaybackEvsEpoch = frame.stream_epoch;
  }
  $('playbackTime').textContent = `${(frame.relative_ms / 1000).toFixed(3)} / ${(playbackManifest.duration_ms / 1000).toFixed(3)} s`;
  $('playbackFrameMeta').textContent =
    `seq ${String(frame.sequence).padStart(6, '0')} · camera ID ${frame.camera_frame_id || '—'} · ` +
    `EVS ${frame.raw_time_us} µs · epoch ${frame.stream_epoch}`;
}

function playbackTick(now) {
  if (!playbackRunning || !playbackManifest) return;
  const speed = Number($('playbackSpeed').value) || 1;
  const targetMs = playbackTimelineStart + (now - playbackWallStart) * speed;
  const frames = playbackManifest.frames;
  let nextIndex = playbackIndex;
  while (nextIndex + 1 < frames.length && frames[nextIndex + 1].relative_ms <= targetMs) {
    nextIndex += 1;
  }
  if (nextIndex !== playbackIndex) renderPlaybackFrame(nextIndex);
  if (playbackIndex >= frames.length - 1 && targetMs >= playbackManifest.duration_ms) {
    pausePlayback();
    return;
  }
  playbackAnimationFrame = window.requestAnimationFrame(playbackTick);
}

function togglePlayback() {
  if (!playbackManifest?.frames?.length) return;
  if (playbackRunning) {
    pausePlayback();
    return;
  }
  if (playbackIndex >= playbackManifest.frames.length - 1) renderPlaybackFrame(0);
  playbackRunning = true;
  playbackWallStart = performance.now();
  playbackTimelineStart = playbackManifest.frames[playbackIndex].relative_ms;
  $('playbackPlayButton').textContent = 'Ⅱ 一時停止';
  playbackAnimationFrame = window.requestAnimationFrame(playbackTick);
}

async function preparePlayback() {
  if (!selectedRecordingId) throw new Error('再生するセッションを選択してください。');
  pausePlayback();
  $('playbackMessage').textContent = '同期時刻とEVS RAWを読み込み中…';
  $('preparePlaybackButton').hidden = true;
  try {
    const data = await requestJson(
      `${API.frame}/recordings/${encodeURIComponent(selectedRecordingId)}/playback`);
    if (selectedRecordingId !== data.playback.session_id) return;
    playbackManifest = data.playback;
    playbackIndex = 0;
    $('playbackSlider').max = Math.max(0, playbackManifest.frames.length - 1);
    $('playbackPlaceholder').hidden = true;
    $('playbackContent').hidden = false;
    renderPlaybackFrame(0, true);
  } catch (error) {
    $('playbackMessage').textContent = error.message;
    $('preparePlaybackButton').hidden = false;
    throw error;
  }
}

async function selectRecording(sessionId) {
  if (selectedRecordingId !== sessionId) resetPlayback();
  selectedRecordingId = sessionId;
  renderRecordingList();
  $('dataEmpty').hidden = false;
  $('dataEmpty').textContent = 'セッションを読み込み中…';
  $('dataDetail').hidden = true;
  try {
    const data = await requestJson(`${API.frame}/recordings/${encodeURIComponent(sessionId)}`);
    if (selectedRecordingId !== sessionId) return;
    const session = data.session;
    const frame = session.frame || {};
    const sync = session.synchronization || {};
    const losses = session.losses || {};
    const roi = session.recording_roi || {};
    selectedRecordingPath = session.path || '';
    $('dataSessionTitle').textContent = session.session_id;
    $('dataSessionPath').textContent = selectedRecordingPath;
    $('dataDurationMetric').textContent = `${Number(session.duration_seconds || 0).toFixed(1)} s`;
    $('dataFrameMetric').textContent = frame.saved_frames || 0;
    $('dataTriggerMetric').textContent = sync.reference_trigger_edges || 0;
    $('dataMatchedMetric').textContent = `${sync.matched || 0} / ${sync.reference_trigger_edges || 0}`;
    $('dataRoiMetric').textContent = roi.width && roi.height ? `${roi.width}×${roi.height}` : '—';
    $('dataLossMetric').textContent = losses.total || 0;
    $('dataLossMetricBox').classList.toggle('alert', Boolean(losses.total));
    $('dataPreviewMeta').textContent = `先頭 ${session.preview_images.length} 枚 / ${frame.saved_frames || 0}`;
    $('dataSizeMeta').textContent = `合計 ${formatBytes(session.total_size_bytes)}`;
    $('savedPreviewGrid').replaceChildren();
    for (const filename of session.preview_images) {
      const figure = document.createElement('div');
      figure.className = 'saved-preview';
      const image = document.createElement('img');
      image.loading = 'lazy';
      image.alt = `${session.session_id} · ${filename}`;
      image.src = `${API.frame}/recordings/${encodeURIComponent(session.session_id)}/preview/${encodeURIComponent(filename)}`;
      image.addEventListener('click', () => {
        $('dialogVideo').src = image.src;
        $('dialogTitle').textContent = image.alt;
        dialog.showModal();
      });
      const caption = document.createElement('span');
      caption.textContent = filename;
      figure.append(image, caption);
      $('savedPreviewGrid').appendChild(figure);
    }
    if (!session.preview_images.length) {
      const empty = document.createElement('div');
      empty.className = 'session-list-empty';
      empty.textContent = 'プレビュー可能なPGM画像がありません。';
      $('savedPreviewGrid').appendChild(empty);
    }
    $('dataFileList').replaceChildren(...session.files.map((file) => {
      const row = document.createElement('div');
      row.className = 'file-row';
      const path = document.createElement('span');
      path.textContent = file.path;
      const count = document.createElement('span');
      count.textContent = file.count === null || file.count === undefined ? '—' : `${file.count.toLocaleString()} ${file.path.includes('*') ? 'files' : file.path.endsWith('.csv') ? 'rows' : 'file'}`;
      const size = document.createElement('span');
      size.textContent = formatBytes(file.size_bytes);
      row.append(path, count, size);
      return row;
    }));
    $('dataEmpty').hidden = true;
    $('dataDetail').hidden = false;
  } catch (error) {
    $('dataEmpty').textContent = error.message;
    showToast(error.message, 'error');
  }
}

async function updateSave() {
  const saveLocation = value('save_location');
  if (!saveLocation) throw new Error('保存先フォルダを入力してください。');
  const payload = { save_location: saveLocation, save_filename: value('save_filename') };
  const results = await Promise.all([post(`${API.evs}/set_save`, payload), post(`${API.frame}/set_save`, payload)]);
  showToast(`保存設定を反映しました（EVS / Frame）`);
  return results;
}

async function setBias() {
  const data = await post(`${API.evs}/set_bias`, { bias_diff_on: finiteNumber('bias_on', 'Diff ON'), bias_diff_off: finiteNumber('bias_off', 'Diff OFF') });
  showToast(data.message);
}

async function reconnectEvs() {
  const data = await post(`${API.evs}/reconnect`);
  showToast(data.message);
  $('evsConnectionMeta').textContent = 'EVS · reconnecting';
}

async function setPreview() {
  const payload = {
    preset: value('preview_preset'),
    auto_degrade: $('preview_auto_degrade').checked
  };
  const frameResult = await post(`${API.frame}/set_preview`, { ...payload, persist: true });
  await post(`${API.evs}/set_preview`, { ...payload, persist: false });
  showToast(frameResult.message.replace('Frame', '両カメラの'));
}

async function setTrigger() {
  const data = await post(`${API.frame}/set_external_trigger`, {
    enabled: $('frame_trigger_enabled').checked,
    source: value('trigger_source'),
    activation: value('trigger_activation')
  });
  showToast(data.message);
  applyTriggerConfiguration(data.configuration);
}

function populateSelect(id, entries, selected) {
  const select = $(id);
  select.replaceChildren(...entries.map((entry) => {
    const option = document.createElement('option');
    option.value = option.textContent = entry;
    return option;
  }));
  if (entries.includes(selected)) select.value = selected;
}

function applyTriggerConfiguration(configuration = {}, updateInputs = true) {
  const enabled = Boolean(configuration.enabled);
  if (updateInputs) {
    $('frame_trigger_enabled').checked = enabled;
    if (configuration.source) $('trigger_source').value = configuration.source;
    if (configuration.activation) $('trigger_activation').value = configuration.activation;
  }
  $('fps_manual_value').disabled = enabled;
  $('fpsButton').disabled = enabled;
  $('fpsTriggerHelp').hidden = !enabled;
}

async function loadTriggerOptions() {
  try {
    const data = await requestJson(`${API.frame}/trigger_options`);
    populateSelect('trigger_source', data.sources || [], data.configuration?.source);
    populateSelect('trigger_activation', data.activations || [], data.configuration?.activation);
    applyTriggerConfiguration(data.configuration);
    triggerOptionsLoaded = Boolean(data.sources?.length && data.activations?.length);
  } catch { /* カメラ接続後の状態更新で再試行する */ }
}

async function setFrameMode(endpoint, mode, valueKey, valueId) {
  const payload = { mode };
  if (valueKey) payload[valueKey] = finiteNumber(valueId, '設定値');
  const data = await post(`${API.frame}/${endpoint}`, payload);
  showToast(data.message);
  await refreshFrameSettings();
}

async function setFramerate() {
  const fps = finiteNumber('fps_manual_value', 'フレームレート');
  if (fps <= 0) throw new Error('フレームレートは 0 より大きい値にしてください。');
  const data = await post(`${API.frame}/set_framerate`, { fps });
  showToast(data.message, data.persisted ? 'success' : 'error');
  await refreshFrameSettings();
}

async function setBandwidth() {
  const data = await post(`${API.frame}/set_bandwidth_preset`, {
    preset: value('bandwidth_preset')
  });
  showToast(data.message);
  await refreshFrameSettings();
}

function applyFpsCapabilities(data) {
  const minimum = Number(data.fps_min);
  const maximum = Number(data.fps_max);
  const increment = Number(data.fps_increment);
  if (Number.isFinite(minimum)) $('fps_manual_value').min = minimum;
  if (Number.isFinite(maximum)) $('fps_manual_value').max = maximum;
  if (Number.isFinite(increment) && increment > 0) $('fps_manual_value').step = increment;
  $('fpsRangeHelp').textContent = Number.isFinite(minimum) && Number.isFinite(maximum)
    ? `設定可能範囲 ${minimum.toFixed(2)}〜${maximum.toFixed(2)} fps`
    : '設定可能範囲を取得できません。';
}

function updateFpsReadout(configured, measured, exposure) {
  const configuredNumber = Number(configured);
  const measuredNumber = Number(measured);
  $('fpsActualHelp').textContent =
    `設定 ${Number.isFinite(configuredNumber) ? configuredNumber.toFixed(2) : '—'} fps · ` +
    `実測 ${Number.isFinite(measuredNumber) ? measuredNumber.toFixed(2) : '—'} fps`;
  const exposureNumber = Number(exposure);
  const periodUs = configuredNumber > 0 ? 1_000_000 / configuredNumber : Infinity;
  const exposureLimited = Number.isFinite(exposureNumber) && exposureNumber > periodUs;
  $('fpsActualHelp').classList.toggle('warning', exposureLimited);
  if (exposureLimited) {
    $('fpsActualHelp').textContent += ` · 露光 ${exposureNumber.toFixed(0)} µsが周期を超えています`;
  }
}

async function refreshFrameSettings() {
  try {
    const data = await requestJson(`${API.frame}/get_settings`);
    $('exposure_manual_value').value = data.exposure;
    $('gain_manual_value').value = data.gain;
    $('fps_manual_value').value = data.fps;
    applyFpsCapabilities(data);
    updateFpsReadout(data.fps, data.fps_measured, data.exposure);
  } catch { /* 起動直後やカメラ未接続時は状態表示に任せる */ }
}

function updateRecordingUi(active, finalizing = false) {
  recording = active;
  $('recordStatus').hidden = !active;
  $('recordButton').disabled = active || finalizing;
  $('stopButton').disabled = !active;
  $('bandwidthButton').disabled = active || finalizing;
}

async function startRecording() {
  await updateSave();
  const preflight = await Promise.all([
    requestJson(`${API.evs}/status`), requestJson(`${API.frame}/status`)
  ]);
  const pixelFormat = preflight[1].camera_settings?.PixelFormat || '';
  const externallyTriggered = Boolean(preflight[1].trigger_configuration?.enabled);
  const frameHasData = preflight[1].frame_ready &&
    preflight[1].last_frame_status === 'FrameStatus.Complete';
  if (!preflight[0].streaming || !preflight[0].trigger_in || !preflight[1].streaming ||
      (!externallyTriggered && !frameHasData)) {
    throw new Error('両カメラとEVS Trigger Inを確認し、フリーラン時は映像受信後に録画してください。');
  }
  if (!/Bayer..8/.test(pixelFormat)) {
    throw new Error(`フレームカメラが8-bit Bayerではありません（${pixelFormat || '不明'}）。`);
  }
  const now = new Date().toISOString().replace(/[-:]/g, '');
  const label = value('save_filename').replace(/[^A-Za-z0-9_.-]+/g, '-').replace(/^-+|-+$/g, '');
  const random = Math.random().toString(36).slice(2, 8);
  activeSessionId = `${now}${label ? `_${label}` : ''}_${random}`;
  await post(`${API.evs}/start_recording`, { session_id: activeSessionId });
  try {
    await post(`${API.frame}/start_recording`, { session_id: activeSessionId });
  } catch (error) {
    await Promise.allSettled([post(`${API.evs}/stop_recording`)]);
    activeSessionId = null;
    throw error;
  }
  updateRecordingUi(true);
  $('sessionLabel').textContent = activeSessionId;
  showToast('計測セッションを開始しました。');
}

async function stopRecording() {
  // Frameは受付停止後すぐ応答し、画像キューだけをバックグラウンドで排出する。
  // そのためEVSを余分なExposureActiveを蓄積する前に停止できる。
  stoppingRecording = true;
  try {
    const frameResult = await Promise.allSettled([post(`${API.frame}/stop_recording`)]);
    const evsResult = await Promise.allSettled([post(`${API.evs}/stop_recording`)]);
    updateRecordingUi(false);
    const failed = [...frameResult, ...evsResult].find((result) => result.status === 'rejected');
    if (failed) throw failed.reason;
    $('sessionLabel').textContent = `完了 · ${activeSessionId || ''}`;
    activeSessionId = null;
    showToast('計測セッションを保存しました。');
  } finally {
    stoppingRecording = false;
  }
}

async function startIntervalRecording() {
  if (intervalTimer !== null) throw new Error('インターバル録画はすでに実行中です。');
  const intervalSec = finiteNumber('interval_sec', '録画間隔');
  const durationSec = finiteNumber('duration_sec', '録画時間');
  if (intervalSec <= 0 || durationSec <= 0) throw new Error('録画間隔と録画時間は 0 より大きい値にしてください。');
  if (durationSec > intervalSec) throw new Error('録画時間は録画間隔以下にしてください。');
  await startRecording();
  recordingTimeout = window.setTimeout(stopRecordingSafely, durationSec * 1000);
  intervalTimer = window.setInterval(async () => {
    try {
      await startRecording();
      recordingTimeout = window.setTimeout(stopRecordingSafely, durationSec * 1000);
    } catch (error) { showToast(error.message, 'error'); }
  }, intervalSec * 1000);
  showToast(`${intervalSec} 秒間隔のインターバル録画を開始しました。`);
}

async function stopRecordingSafely() { try { await stopRecording(); } catch (error) { showToast(error.message, 'error'); } }

async function stopIntervalRecording() {
  if (intervalTimer !== null) window.clearInterval(intervalTimer);
  if (recordingTimeout !== null) window.clearTimeout(recordingTimeout);
  intervalTimer = recordingTimeout = null;
  if (recording) await stopRecording();
  else showToast('インターバル録画を停止しました。');
}

async function refreshStatus() {
  const checks = await Promise.allSettled([
    requestJson(`${API.evs}/status`),
    requestJson(`${API.frame}/status`)
  ]);
  const evsOnline = checks[0].status === 'fulfilled' && checks[0].value.streaming;
  const frameOnline = checks[1].status === 'fulfilled' && checks[1].value.streaming;
  ['evsStatusDot', 'evsCardDot'].forEach((id) => $(id).classList.toggle('online', evsOnline));
  ['frameStatusDot', 'frameCardDot'].forEach((id) => $(id).classList.toggle('online', frameOnline));
  const evsConnection = checks[0].status === 'fulfilled' ? checks[0].value.connection || {} : {};
  const frameConnection = checks[1].status === 'fulfilled' ? checks[1].value.connection || {} : {};
  $('evsConnectionMeta').textContent = `EVS · ${connectionStateLabel(evsConnection)} · 復旧${evsConnection.successful_reconnections || 0}`;
  $('frameConnectionMeta').textContent = `FRAME · ${connectionStateLabel(frameConnection)} · 復旧${frameConnection.successful_reconnections || 0}`;
  if (evsOnline && !previousEvsOnline) reconnectVideo('evsVideo', `${API.evs}/video_feed`);
  if (frameOnline && !previousFrameOnline) {
    reconnectVideo('frameVideo', `${API.frame}/video_feed`);
    refreshFrameSettings();
    loadTriggerOptions();
  }
  previousEvsOnline = evsOnline;
  previousFrameOnline = frameOnline;
  if (frameOnline && !triggerOptionsLoaded) loadTriggerOptions();
  if (checks[0].status === 'fulfilled' && checks[1].status === 'fulfilled') {
    const backendRecording = Boolean(checks[0].value.recording || checks[1].value.recording);
    const frameFinalizing = Boolean(checks[1].value.recording_finalizing);
    const recordingWasActive = recording;
    updateRecordingUi(backendRecording, frameFinalizing);
    if (recordingWasActive && !backendRecording && activeSessionId && !stoppingRecording) {
      showToast('バックエンド再起動により録画が停止しました。保存済みセグメントを確認してください。', 'error');
      $('sessionLabel').textContent = `中断 · ${activeSessionId}`;
      activeSessionId = null;
    }
    const evsQuality = checks[0].value.recording_quality || {};
    const frameQuality = checks[1].value.recording_quality || {};
    const triggerMonitor = checks[0].value.trigger_monitor || {};
    const frameTriggerMonitor = checks[1].value.external_trigger_monitor || {};
    const evsPreview = checks[0].value.preview || {};
    const framePreview = checks[1].value.preview || {};
    const evsProcessing = checks[0].value.processing || {};
    const bandwidth = checks[1].value.bandwidth || {};
    const recordingRoi = checks[1].value.recording_roi || {};
    if (framePreview.requested_preset && document.activeElement !== $('preview_preset')) {
      $('preview_preset').value = framePreview.requested_preset;
    }
    if (document.activeElement !== $('preview_auto_degrade')) {
      $('preview_auto_degrade').checked = Boolean(framePreview.auto_degrade);
    }
    const previewReduced =
      evsPreview.effective_preset !== evsPreview.requested_preset ||
      framePreview.effective_preset !== framePreview.requested_preset ||
      Boolean(evsProcessing.overloaded);
    $('previewStatusHelp').classList.toggle('warning', previewReduced);
    $('previewStatusHelp').textContent =
      `EVS ${evsPreview.label || '—'} ${Number(evsPreview.measured_fps || 0).toFixed(1)}/${Number(evsPreview.target_fps || 0).toFixed(0)}fps ` +
      `(${Number(evsPreview.encode_duration_ms || 0).toFixed(1)}ms, 遅延${Number(evsProcessing.decode_lag_ms || 0).toFixed(1)}ms, ` +
      `${(Number(evsProcessing.event_rate_per_second || 0) / 1_000_000).toFixed(2)}Mev/s, 間引き${evsPreview.skipped_source_frames || 0}) · ` +
      `Frame ${framePreview.label || '—'} ${Number(framePreview.measured_fps || 0).toFixed(1)}/${Number(framePreview.target_fps || 0).toFixed(0)}fps ` +
      `(${Number(framePreview.encode_duration_ms || 0).toFixed(1)}ms, 間引き${framePreview.skipped_source_frames || 0})`;
    if ((evsPreview.auto_degrade_count || 0) > previousEvsPreviewFallbacks ||
        (framePreview.auto_degrade_count || 0) > previousFramePreviewFallbacks) {
      showToast('処理負荷が高いため、記録を優先してライブ表示品質を自動的に下げました。', 'error');
    }
    previousEvsPreviewFallbacks = evsPreview.auto_degrade_count || 0;
    previousFramePreviewFallbacks = framePreview.auto_degrade_count || 0;
    if (bandwidth.preset && document.activeElement !== $('bandwidth_preset')) {
      $('bandwidth_preset').value = bandwidth.preset;
    }
    $('bandwidthHelp').textContent =
      `${Number(bandwidth.limit_bytes_per_second || 0) / 1_000_000} MB/s · ` +
      `自動復帰 ${bandwidth.auto_fallback_count || 0}回`;
    if ((bandwidth.auto_fallback_count || 0) > previousAutoFallbackCount) {
      showToast(`不完全フレームを検出し、帯域を「${bandwidth.preset}」へ下げました。`, 'error');
    }
    previousAutoFallbackCount = bandwidth.auto_fallback_count || 0;
    updateFpsReadout(
      checks[1].value.camera_settings?.AcquisitionFrameRate,
      checks[1].value.measured_fps,
      checks[1].value.camera_settings?.ExposureTime
    );
    $('completeMetric').textContent = frameQuality.complete || 0;
    $('savedMetric').textContent = frameQuality.saved || 0;
    $('incompleteMetric').textContent = frameQuality.incomplete || 0;
    $('gapMetric').textContent = frameQuality.frame_id_missing || 0;
    $('dropMetric').textContent = frameQuality.queue_drops || 0;
    $('triggerMetric').textContent = checks[1].value.camera_settings?.LineInverter
      ? (evsQuality.falling_edges || 0)
      : (evsQuality.rising_edges || 0);
    const referenceIsFalling = Boolean(checks[1].value.camera_settings?.LineInverter);
    const triggerRate = Number(referenceIsFalling
      ? triggerMonitor.falling_hz : triggerMonitor.rising_hz) || 0;
    const triggerPeriod = referenceIsFalling
      ? triggerMonitor.falling_period_ms : triggerMonitor.rising_period_ms;
    $('triggerActivityDot').classList.toggle('online', Boolean(triggerMonitor.active));
    $('triggerActivityText').textContent = !triggerMonitor.enabled
      ? '無効' : (triggerMonitor.active ? '受信中' : '待機中');
    $('triggerRateText').textContent = triggerRate
      ? `${triggerMonitor.active ? '' : '最終 '}${triggerRate.toFixed(2)} Hz · ${Number(triggerPeriod).toFixed(2)} ms`
      : '— Hz · — ms';
    $('triggerCountsText').textContent =
      `${triggerMonitor.rising_edges || 0} / ${triggerMonitor.falling_edges || 0}`;
    const lastPolarity = triggerMonitor.last_polarity === 1
      ? '立上り' : (triggerMonitor.last_polarity === 0 ? '立下り' : '未受信');
    const edgeAgeValue = triggerMonitor.last_edge_age_ms;
    const edgeAge = Number(edgeAgeValue);
    $('triggerLastText').textContent = edgeAgeValue !== null && edgeAgeValue !== undefined && Number.isFinite(edgeAge)
      ? `${lastPolarity} · ${edgeAge.toFixed(0)} ms前` : lastPolarity;
    const frameTriggerEnabled = Boolean(frameTriggerMonitor.enabled);
    const frameTriggerActive = Boolean(frameTriggerMonitor.active);
    const frameTriggerAgeValue = frameTriggerMonitor.last_frame_age_ms;
    const frameTriggerAge = Number(frameTriggerAgeValue);
    const frameTriggerAgeText = frameTriggerAgeValue !== null && frameTriggerAgeValue !== undefined && Number.isFinite(frameTriggerAge)
      ? ` · ${frameTriggerAge.toFixed(0)} ms前` : '';
    const activationSymbol = {
      RisingEdge: '↑ RisingEdge', FallingEdge: '↓ FallingEdge', AnyEdge: '↕ AnyEdge'
    }[frameTriggerMonitor.activation] || frameTriggerMonitor.activation || '—';
    $('frameTriggerPanel').classList.toggle('inactive', !frameTriggerEnabled);
    $('frameTriggerActivityDot').classList.toggle('online', frameTriggerActive);
    $('frameTriggerActivityText').textContent = !frameTriggerEnabled
      ? 'フリーラン' : (frameTriggerActive ? `受信中${frameTriggerAgeText}` : '入力待ち');
    $('frameTriggerInputText').textContent = frameTriggerEnabled
      ? `${frameTriggerMonitor.source || '—'} · ${activationSymbol}` : '外部Trigger OFF';
    const frameTriggerHz = Number(frameTriggerMonitor.measured_hz) || 0;
    $('frameTriggerFramesText').textContent =
      `${frameTriggerMonitor.complete_frames || 0} / ${frameTriggerMonitor.incomplete_frames || 0} · ` +
      (frameTriggerHz ? `${frameTriggerHz.toFixed(2)} Hz` : '— Hz');
    const evsReferenceHz = referenceIsFalling
      ? Number(triggerMonitor.falling_hz) : Number(triggerMonitor.rising_hz);
    const frameToEvsActive = frameTriggerEnabled && frameTriggerActive && Boolean(triggerMonitor.active);
    $('frameToEvsTriggerText').textContent = frameTriggerEnabled
      ? `${frameTriggerMonitor.evs_output_line || 'Line0'} ExposureActive → EVS · ` +
        (frameToEvsActive ? `${(evsReferenceHz || 0).toFixed(2)} Hz` : '待機')
      : 'Line0 ExposureActive → EVS';
    $('queueMetric').textContent = frameQuality.queue_depth || 0;
    $('saveFpsMetric').textContent = Number(frameQuality.save_fps || 0).toFixed(1);
    $('writeRateMetric').textContent = Number(
      frameQuality.write_megabytes_per_second || 0).toFixed(1);
    $('writeFailureMetric').textContent = frameQuality.write_failures || 0;
    $('elapsedMetric').textContent = Number(frameQuality.elapsed_seconds || evsQuality.elapsed_seconds || 0).toFixed(1);
    $('frameReconnectMetric').textContent = frameConnection.successful_reconnections || 0;
    $('evsReconnectMetric').textContent = evsConnection.successful_reconnections || 0;
    $('roiMetric').textContent = recordingRoi.width && recordingRoi.height
      ? `${recordingRoi.width}×${recordingRoi.height}` : '—';
    $('incompleteMetricBox').classList.toggle('alert', Boolean(frameQuality.incomplete));
    $('gapMetricBox').classList.toggle('alert', Boolean(frameQuality.frame_id_missing));
    $('dropMetricBox').classList.toggle('alert', Boolean(frameQuality.queue_drops));
    $('writeFailureMetricBox').classList.toggle('alert', Boolean(frameQuality.write_failures));
    const sessionId = frameQuality.session_id || evsQuality.session_id;
    applyTriggerConfiguration(checks[1].value.trigger_configuration, false);
    if (sessionId) {
      activeSessionId = sessionId;
      $('sessionLabel').textContent = frameFinalizing ? `保存完了処理中 · ${sessionId}` : sessionId;
    }
  }
}

const actions = {
  updateSave, setBias, reconnectEvs, setPreview, setTrigger,
  exposureOnce: () => setFrameMode('set_exposure', 'Once'),
  exposureContinuous: () => setFrameMode('set_exposure', 'Continuous'),
  setExposureManual: () => setFrameMode('set_exposure', 'Manual', 'value', 'exposure_manual_value'),
  gainOnce: () => setFrameMode('set_gain', 'Once'),
  gainContinuous: () => setFrameMode('set_gain', 'Continuous'),
  setGainManual: () => setFrameMode('set_gain', 'Manual', 'value', 'gain_manual_value'),
  wbOnce: () => setFrameMode('set_whitebalance', 'Once'),
  wbContinuous: () => setFrameMode('set_whitebalance', 'Continuous'),
  setFramerate, setBandwidth, startRecording, stopRecording,
  startInterval: startIntervalRecording,
  stopInterval: stopIntervalRecording
};

document.querySelectorAll('[data-action]').forEach((button) => {
  button.addEventListener('click', () => runAction(button, actions[button.dataset.action]));
});

$('captureTabButton').addEventListener('click', () => setActivePage('capture'));
$('dataTabButton').addEventListener('click', () => setActivePage('data'));
$('recordingSearch').addEventListener('input', renderRecordingList);
$('refreshRecordingsButton').addEventListener('click', () =>
  runAction($('refreshRecordingsButton'), loadRecordings));
$('copySessionPathButton').addEventListener('click', async () => {
  if (!selectedRecordingPath) return;
  try {
    await navigator.clipboard.writeText(selectedRecordingPath);
    showToast('セッションのパスをコピーしました。');
  } catch {
    showToast('パスをコピーできませんでした。表示されたパスを選択してください。', 'error');
  }
});
$('preparePlaybackButton').addEventListener('click', () =>
  runAction($('preparePlaybackButton'), preparePlayback));
$('playbackPlayButton').addEventListener('click', togglePlayback);
$('playbackSlider').addEventListener('input', () => {
  pausePlayback();
  renderPlaybackFrame(Number($('playbackSlider').value), true);
});
$('playbackSpeed').addEventListener('change', () => {
  if (!playbackRunning || !playbackManifest) return;
  playbackWallStart = performance.now();
  playbackTimelineStart = playbackManifest.frames[playbackIndex].relative_ms;
});
$('playbackWindow').addEventListener('change', () => renderPlaybackFrame(playbackIndex, true));
$('playbackEvsLoad').addEventListener('change', () => {
  lastPlaybackEvsMs = -Infinity;
  renderPlaybackFrame(playbackIndex, true);
});
$('playbackOpacity').addEventListener('input', () => {
  $('playbackEvsImage').style.opacity = $('playbackOpacity').value;
});

const dialog = $('videoDialog');
document.querySelectorAll('.stream-stage img').forEach((image) => {
  image.addEventListener('click', () => {
    $('dialogVideo').src = image.src;
    $('dialogTitle').textContent = image.alt;
    dialog.showModal();
  });
});

function reconnectVideo(id, baseUrl) {
  const image = $(id);
  image.src = `${baseUrl}?reconnect=${Date.now()}`;
}

$('evsVideo').addEventListener('error', () => {
  if (isCaptureUiActive()) {
    window.setTimeout(() => {
      if (isCaptureUiActive()) reconnectVideo('evsVideo', `${API.evs}/video_feed`);
    }, 1500);
  }
});
$('frameVideo').addEventListener('error', () => {
  if (isCaptureUiActive()) {
    window.setTimeout(() => {
      if (isCaptureUiActive()) reconnectVideo('frameVideo', `${API.frame}/video_feed`);
    }, 1500);
  }
});
document.addEventListener('visibilitychange', syncCaptureActivity);
window.addEventListener('pagehide', () => {
  capturePageActive = false;
  syncCaptureActivity();
});
document.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', (event) => { if (event.target === dialog) dialog.close(); });
dialog.addEventListener('close', () => { $('dialogVideo').removeAttribute('src'); });

const initialMode = new URLSearchParams(window.location.search).get('mode');
if (initialMode === 'review') setActivePage('data');
syncCaptureActivity();
initializeSaveLocation();
updateRecordingUi(false);
refreshFrameSettings();
loadTriggerOptions();
refreshStatus();
window.setInterval(refreshStatus, 3000);
window.setInterval(syncCaptureActivity, 5000);
