// OCTO-EARTH helpers shared by the two cinematic views (update p37): screen labels, camera presets with smooth
// transitions + an optional automatic "cinematic" camera, the right-hand HUD panel and rendering with the panel
// offset (same approach as the Weather & life view).
import * as THREE from './vendor/three.module.js';

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
export const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
export const sm = (k) => { k = clamp(k, 0, 1); return k * k * (3 - 2 * k); };
export function fmtE(x, d = 2) {
  if (!Number.isFinite(x)) return 'N/A';
  if (x === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(x)));
  if (e >= -2 && e < 5) return x.toLocaleString('en-US', { maximumFractionDigits: d });
  return `${(x / 10 ** e).toFixed(d)}e${e}`;
}

/** DOM labels that follow 3-D points. colors: { key: '#rrggbb' } */
export function createLabels(container, canvas, camera, colors) {
  const L = {}, _v = new THREE.Vector3(), S = { on: true };
  for (const [k, col] of Object.entries(colors)) {
    const d = document.createElement('div'); d.className = 'blabel mono small povlbl cinelbl'; d.style.color = col; d.style.display = 'none';
    container.appendChild(d); L[k] = d;
  }
  function set(k, pos, text) {
    const el = L[k]; if (!el) return;
    if (!S.on || !pos || !text) { el.style.display = 'none'; return; }
    _v.copy(pos).project(camera);
    if (_v.z > 1 || Math.abs(_v.x) > 1.02 || Math.abs(_v.y) > 1.02) { el.style.display = 'none'; return; }
    if (el.textContent !== text) el.textContent = text; el.style.display = 'block';
    const w = canvas.clientWidth, h = canvas.clientHeight;
    el.style.left = `${clamp((_v.x * 0.5 + 0.5) * w + 8, 2, w - el.offsetWidth - 2)}px`; el.style.top = `${clamp((-_v.y * 0.5 + 0.5) * h - 6, 2, h - el.offsetHeight - 2)}px`;
  }
  function declutter() {
    const placed = [];
    for (const el of Object.values(L)) {
      if (el.style.display === 'none') continue;
      let y = parseFloat(el.style.top); const x = parseFloat(el.style.left), w = el.offsetWidth, hh = el.offsetHeight;
      for (let n = 0; n < 8; n++) { const hit = placed.find((b) => x < b.x + b.w && x + w > b.x && y < b.y + b.h && y + hh > b.y); if (!hit) break; y = hit.y + hit.h + 2; }
      el.style.top = `${y}px`; placed.push({ x, y, w, h: hh });
    }
  }
  function hide() { for (const el of Object.values(L)) el.style.display = 'none'; }
  return { set, hide, declutter, state: S, els: L };
}

/**
 * Camera presets with smooth transitions. presets: { name: { pos: [x,y,z], target: [x,y,z] } } (functions allowed:
 * () => ({pos, target}) for presets that follow moving objects). Dragging the view switches the automatic camera off.
 */
export function createCineCamera(camera, controls, presets, onAutoChange) {
  const C = { auto: true, tr: null, current: null };
  const p0 = new THREE.Vector3(), t0 = new THREE.Vector3(), p1 = new THREE.Vector3(), t1 = new THREE.Vector3();
  controls.addEventListener('start', () => { if (C.auto || C.tr) { C.auto = false; C.tr = null; if (onAutoChange) onAutoChange(false); } });
  const resolve = (name) => { const p = presets[name]; return typeof p === 'function' ? p() : p; };
  /** Fly to a preset over 'dur' seconds (0 = jump). */
  function go(name, dur = 1.8) {
    const p = resolve(name); if (!p) return; C.current = name;
    if (dur <= 0) { camera.position.set(...p.pos); controls.target.set(...p.target); C.tr = null; controls.update(); return; }
    C.tr = { name, k: 0, dur }; p0.copy(camera.position); t0.copy(controls.target);
  }
  function update(dt) {
    if (!C.tr) {                    // automatic camera: stay on the current preset (it may follow a moving object)
      if (C.auto && C.current && typeof presets[C.current] === 'function') { const p = resolve(C.current); camera.position.set(...p.pos); controls.target.set(...p.target); }
      return;
    }
    const p = resolve(C.tr.name); C.tr.k = Math.min(1, C.tr.k + dt / C.tr.dur); const k = sm(C.tr.k);
    p1.set(...p.pos); t1.set(...p.target);
    camera.position.lerpVectors(p0, p1, k); controls.target.lerpVectors(t0, t1, k);
    if (C.tr.k >= 1) C.tr = null;
  }
  function setAuto(on) { C.auto = !!on; if (onAutoChange) onAutoChange(C.auto); }
  return { go, update, setAuto, state: C, presets };
}

/** Render with the view shifted left so the subject stays clear of the right-hand HUD panel. */
export function renderShifted(renderer, scene, camera, canvas, controls, panel) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  camera.aspect = w / Math.max(1, h);
  const shift = panel && !panel.hidden && w > 700 ? Math.min(180, 0.12 * w) : 0;
  camera.setViewOffset(w, h, shift, 0, w, h);
  camera.updateProjectionMatrix(); controls.update(); camera.updateMatrixWorld();
  renderer.render(scene, camera);
}

/** HUD helpers (same look as the Weather & life panel). */
export const hudRow = (label, value, tip = '', cls = '') => `<div class="crow ${cls}" title="${esc(tip)}"><span class="cl">${label}</span><span class="cv">${value}</span></div>`;
export const hudBar = (frac, col = '#7fc8ff') => `<span class="cbar"><i style="width:${(100 * clamp(frac, 0, 1)).toFixed(1)}%;background:${col}"></i></span>`;
