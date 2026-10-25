// OCTO-EARTH "failed attempts" page (update p38): four INDEPENDENT models, viewed one at a time.
// One WebGL renderer and one animation loop; each model is created on first view and only the active model is updated
// and drawn. Physics numbers: failedPhysics.js (mirrored in failed_models.py).
import * as THREE from './vendor/three.module.js';
import { createLaserView } from './laserScene.js';
import { createHydrogenScoopView } from './hydrogenScene.js';
import { createMagnetView } from './magnetScene.js';
import { createDysonView } from './dysonScene.js';
import { DV_REQ, F_REQ, DYSON, K } from './failedPhysics.js';
import { fmtE, esc } from './cineKit.js';

const $ = (id) => document.getElementById(id);
const canvas = $('scene'), center = $('center');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));

function dysonWrapped() {
  const v = createDysonView(renderer, canvas, center);
  v.key = 'dyson';
  v.info = {
    title: 'Dyson beam on a reflector held by 8 arms',
    idea: 'A Dyson swarm collects sunlight and sends one beam to a giant reflector held by Earth\u2019s 8 arms. The light pressure on the reflector is carried through the arms into Earth, and the reflected beam is absorbed by Earth.',
    fails: [`At full power (${fmtE(DYSON.P, 2)} W) the reflector load overloads the arms: they break in a cascade and the reflector is torn loose.`,
      `Even with intact arms the push is ${fmtE(DYSON.F, 2)} N = ${(100 * DYSON.frac).toFixed(1)} % of the ${fmtE(F_REQ, 2)} N needed.`,
      `The absorbed beam would heat Earth to ~${fmtE(DYSON.Tsurf, 0)} K (same problem as model 1).`],
  };
  return v;
}
const MODELS = [
  { label: 'Laser push', make: () => createLaserView(renderer, canvas, center) },
  { label: 'Hydrogen harvesting', make: () => createHydrogenScoopView(renderer, canvas, center) },
  { label: 'Magnetic arms', make: () => createMagnetView(renderer, canvas, center) },
  { label: 'Dyson beam', make: dysonWrapped },
];
const views = [null, null, null, null];
let cur = -1;
const get = (i) => views[i] || (views[i] = MODELS[i].make());

$('faDv').textContent = DV_REQ.toLocaleString('en-US', { maximumFractionDigits: 0 });
$('faF').textContent = fmtE(F_REQ, 2);

function show(i) {
  i = ((i % 4) + 4) % 4;
  if (i === cur) return;
  let v;
  try { v = get(i); } catch (e) { showErr(e); return; }
  if (cur >= 0 && views[cur]) { views[cur].setActive(false); if (views[cur].hideLabels) views[cur].hideLabels(); }
  cur = i; v.setActive(true);
  for (const b of document.querySelectorAll('#faTabs button')) b.classList.toggle('on', +b.dataset.m === i);
  $('faTitle').textContent = `${i + 1}. ${v.info.title}`;
  $('faIdea').textContent = v.info.idea;
  $('faFails').innerHTML = v.info.fails.map((f) => `<li>${esc(f)}</li>`).join('');
  $('faWhich').textContent = `model ${i + 1} of 4: ${MODELS[i].label}`;
  if (location.hash !== `#${i + 1}`) history.replaceState(null, '', `#${i + 1}`);
  syncBar();
}
function syncBar() {
  const v = views[cur]; if (!v) return;
  const S = v.S, t = !S.started ? 'Pause' : S.ended ? 'Pause' : S.running ? 'Pause' : 'Resume';
  if ($('faPause').textContent !== t) $('faPause').textContent = t;
  $('faPause').disabled = !S.started || !!S.ended;
}
function showErr(e) { console.error(e); const d = $('faErr'); d.hidden = false; d.textContent = `Error: ${e && e.message ? e.message : e}`; }

document.querySelector('#faTabs').addEventListener('click', (e) => { const b = e.target.closest('button[data-m]'); if (b) show(+b.dataset.m); });
$('faPrev').onclick = () => show(cur - 1);
$('faNext').onclick = () => show(cur + 1);
$('faStart').onclick = () => { views[cur].start(); syncBar(); };
$('faPause').onclick = () => { views[cur].togglePause(); syncBar(); };
$('faReset').onclick = () => { views[cur].reset(); syncBar(); };
$('faInfoHide').onclick = () => { const b = $('faInfoBody'); b.hidden = !b.hidden; $('faInfoHide').textContent = b.hidden ? '+' : '-'; };
window.addEventListener('keydown', (e) => {
  if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) return;
  if (e.key >= '1' && e.key <= '4') show(+e.key - 1);
  else if (e.key === 'ArrowRight') show(cur + 1);
  else if (e.key === 'ArrowLeft') show(cur - 1);
  else if (e.key === ' ') { e.preventDefault(); views[cur].togglePause(); syncBar(); }
  else if (e.key === 'r' || e.key === 'R') { views[cur].reset(); syncBar(); }
});

let lastW = 0, lastH = 0;
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (w !== lastW || h !== lastH) { lastW = w; lastH = h; renderer.setSize(w, h, false); }
}
let last = performance.now(), fpsN = 0, fpsT = 0;
function frame(now) {
  const dt = Math.min(0.1, Math.max(0, (now - last) / 1000)); last = now;
  resize();
  const v = views[cur];
  if (v) { try { v.update(dt); v.render(); } catch (e) { showErr(e); return; } }
  fpsN++; fpsT += dt; if (fpsT > 1) { $('faFps').textContent = `${Math.round(fpsN / fpsT)} fps`; fpsN = 0; fpsT = 0; syncBar(); }
  requestAnimationFrame(frame);
}
const h0 = parseInt((location.hash || '#1').slice(1), 10);
show(Number.isFinite(h0) && h0 >= 1 && h0 <= 4 ? h0 - 1 : 0);
requestAnimationFrame(frame);

// test hook: views, switching and fast-forward (advances the active model by 'sec' seconds of animation time)
window.__fa = {
  views, show, get, cur: () => cur, K,
  ff(sec) { const v = views[cur]; let t = 0; while (t < sec) { v.update(0.1); t += 0.1; } v.render(); syncBar(); return v.S; },
};
