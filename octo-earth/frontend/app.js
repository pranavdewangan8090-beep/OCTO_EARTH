// OCTO-EARTH viewer.
// The browser does NO physics. It draws states recorded by the Python engine:
//   * Earth: one recorded state (position, velocity before and after the event) per ejection event,
//     joined by a cubic Hermite curve in polar coordinates (accuracy checked by Python check 13).
//     The curve uses the velocity AFTER the event at its start and BEFORE the next event at its end,
//     so the recoil kink is shown, never smoothed away. Gaps larger than 2.5 event spacings are NOT
//     interpolated: Earth is held at the last recorded state and a notice is shown.
//   * Moon (both models): its own recorded stream, 8 states per event (fast model: exact Earth-Moon
//     two-body orbit; high-fidelity: N-body). Its current orbit is drawn as a thin ellipse.
//   * Asteroid: docked to Earth on the launch ring's axis (equidistant from all 8 arms, 2 R_E long; see launcher.js).
//   * Packets: positions computed in Python (two-body solar gravity) at fixed times after release.
//   * Mars (both models): its own recorded state in every row (position + velocity), joined by the same
//     polar Hermite curve. Its start orbit (faint) and current orbit (dashed) show how Earth changes it.
//     The safe / caution distances around Mars are drawn at TRUE scale (markers are enlarged).
//   * Earth POV view (pov.js): true-scale close-up around Earth fed by the same recorded states; its
//     environment layers are display-only standard models and the ring/arms are an illustration.
import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import { createPov, LAYERS as POV_LAYERS, DEFAULT_LAYERS as POV_DEFAULTS } from './pov.js';
import { createCollectorView } from './collector.js';
import { createWeatherView } from './weather.js';
import { createOvExtras, ledgerPanelHtml, drawClimateChart, climateAtDec } from './ovextras.js';
import { createPlanets, NAMES as PL_NAMES } from './planets.js';
import { GRAB_Z, ARM_L, RING_R as LRING_R } from './launcher.js';

const $ = (id) => document.getElementById(id);
const S = 10;                       // scene units per AU
const MOON_X = 100;                 // Moon distance exaggeration (drawing only)
const AU = 1.495978707e11, YEAR_S = 31557600, DAY_Y = 1 / 365.25;
const KMS2AUY = 1000 * YEAR_S / AU; // km/s -> AU/yr
const M_EARTH = 5.97216787e24, M_MOON = 7.3458e22;
const EARTH_R = 0.3, AST_RMAX = 0.17;   // marker radii [scene units] (not to scale)
const AST_OFFSET_AU = EARTH_R * 1.25 / S;  // packet paths start at the grab point under the asteroid marker, on the ring axis (visual offset, = EARTH_R x GRAB_Z)
const GM_EARTH_KM = 398600.4418;    // km^3/s^2
const PERSIST_WALL_S = 0.6;         // packet paths stay visible at least this long (wall clock)
// row layout (see _viewer_rows in octoearth_rl.py)
const R_T = 0, R_X = 1, R_VPRE = 4, R_VPOST = 7, R_DM = 10, R_U = 11, R_VREL = 12, R_FATE = 13, R_MAST = 14,
  R_EREM = 15, R_NFRAG = 16, R_DV = 17, R_EDRAW = 18, R_EHAT = 19, R_MARS = 22, R_MARSV = 25, R_DMARS = 28, R_SHIELD = 29,
  R_PAV = 30, R_PIN = 31, R_ETA = 32, R_PLAUNCH = 33;   // beam power available at the arms, microwaves absorbed, capture efficiency, launcher power [W]
const GM_SUN = 1.32712440018e20, GM_MARS = 4.282837e13;
const SHIELD_TXT = { 0: "agent's command", 1: 'forced coasting', 2: 'forced full thrust', 3: 'forced braking (thrust reversed)' };
const FATE = { 0: 'none', 1: 'leaves the Solar System', 2: 'stays bound to the Sun', 3: 'FELL BACK to Earth', 4: 'hit the Moon' };

// ------------------------------------------------------------------ formatting helpers
const ok = (v) => v !== null && v !== undefined && Number.isFinite(+v);
const pct = (v, d = 1) => ok(v) ? (100 * v).toFixed(d) + ' %' : 'N/A';
const num = (v, d = 4) => ok(v) ? (+v).toFixed(d) : 'N/A';
const sci = (v, d = 2) => ok(v) ? (+v).toExponential(d).replace('e+', 'e') : 'N/A';
const int = (v) => ok(v) ? Math.round(v).toLocaleString('en-US') : 'N/A';
const esc = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
function fmtSpeed(y) {
  if (y < 1) return (y * 365.25).toFixed(y * 365.25 < 10 ? 1 : 0) + (Math.abs(y * 365.25 - 1) < 0.05 ? ' day' : ' days');
  return (y < 10 ? y.toFixed(1) : Math.round(y)) + (Math.abs(y - 1) < 0.05 ? ' year' : ' years');
}
function fmtYear(t) { return 'Year ' + (t < 100 ? t.toFixed(3) : t.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })); }
function dirWords(d) {
  if (!d) return '—';
  const [r, t, n] = d; const a = [[Math.abs(t), t >= 0 ? 'prograde' : 'retrograde'], [Math.abs(r), r >= 0 ? 'outward' : 'inward'], [Math.abs(n), n >= 0 ? 'north' : 'south']];
  a.sort((x, y) => y[0] - x[0]);
  return a[1][0] > 0.25 ? `${a[0][1]} + ${a[1][1]}` : a[0][1];
}

// ------------------------------------------------------------------ three.js scene
const canvas = $('scene');
// powerPreference 'high-performance' asks laptops with two graphics chips for the dedicated (e.g. NVIDIA) one
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
let PIXEL_RATIO = Math.min(2, window.devicePixelRatio || 1);
renderer.setPixelRatio(PIXEL_RATIO);
// Which chip is the browser really drawing with? Software renderers (SwiftShader, llvmpipe, Microsoft Basic Render)
// mean the browser does NOT use the graphics card at all -> very low frame rate -> the whole viewer lags.
const GL_GPU = (() => {
  try {
    const gl = renderer.getContext(), ext = gl.getExtension('WEBGL_debug_renderer_info');
    const name = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    return { name: String(name || 'unknown'), software: /swiftshader|llvmpipe|softpipe|basic render|software/i.test(String(name)) };
  } catch (e) { return { name: 'unknown', software: false }; }
})();
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05070c);
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 5000);
// default view is shifted slightly so the Sun sits right of centre: the compact HUD covers the top-left corner
const HOME_X = -2.5;
camera.position.set(HOME_X, -40, 34);
const controls = new OrbitControls(camera, canvas);
controls.target.set(HOME_X, 0, 0);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.45));
scene.add(new THREE.PointLight(0xffffff, 3, 0, 0));
const sun = new THREE.Mesh(new THREE.SphereGeometry(0.6, 32, 16), new THREE.MeshBasicMaterial({ color: 0xffcc33 }));
scene.add(sun);

function circlePts(rAU, n = 256) {
  const p = []; for (let i = 0; i <= n; i++) { const a = i / n * Math.PI * 2; p.push(new THREE.Vector3(Math.cos(a) * rAU * S, Math.sin(a) * rAU * S, 0)); } return p;
}
function ring(rAU, color, opacity) {
  const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(circlePts(rAU)), new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
  scene.add(l); return l;
}
ring(1.0, 0x4fa3ff, 0.18);
let targetRing = ring(2.7, 0x3ddc84, 0.65);
let targetAU = 2.7;
function setTargetRing(au) { if (au === targetAU) return; targetAU = au; scene.remove(targetRing); targetRing = ring(au, 0x3ddc84, 0.65); }

const earth = new THREE.Mesh(new THREE.SphereGeometry(EARTH_R, 24, 12), new THREE.MeshStandardMaterial({ color: 0x4fa3ff, emissive: 0x0a2a55 }));
scene.add(earth);
// overview details (textured Earth, launcher ring + 8 arms, collector satellites, the 12 simulated collectors + grey ghosts,
// the ring of space mirrors)
const OVX = createOvExtras(scene, S, earth, EARTH_R);
let OVX_STATE = { frozen: false };
const moon = new THREE.Mesh(new THREE.SphereGeometry(0.17, 16, 8), new THREE.MeshStandardMaterial({ color: 0xdddddd, emissive: 0x555555 }));
scene.add(moon);
// the Moon's current (osculating) orbit around Earth, from the simulated Earth-Moon state, drawn at the same x100 scale
const moonOrbPos = new Float32Array(129 * 3);
const moonOrbGeom = new THREE.BufferGeometry(); moonOrbGeom.setAttribute('position', new THREE.BufferAttribute(moonOrbPos, 3)); moonOrbGeom.setDrawRange(0, 0);
const moonOrbit = new THREE.Line(moonOrbGeom, new THREE.LineBasicMaterial({ color: 0xbbbbbb, transparent: true, opacity: 0.45 }));
moonOrbit.frustumCulled = false; scene.add(moonOrbit);
const moonLinkGeom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
const moonLink = new THREE.Line(moonLinkGeom, new THREE.LineBasicMaterial({ color: 0xbbbbbb, transparent: true, opacity: 0.3 }));
moonLink.frustumCulled = false; scene.add(moonLink);
const asteroid = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 1), new THREE.MeshStandardMaterial({ color: 0xc08a58, emissive: 0x3a2412, flatShading: true }));
scene.add(asteroid);
// Mars: marker (enlarged), start orbit (faint), current osculating orbit (dashed), true-scale safety rings, Earth-Mars line
const MARS_R = 0.22;
const mars = new THREE.Mesh(new THREE.SphereGeometry(MARS_R, 20, 10), new THREE.MeshStandardMaterial({ color: 0xff6a45, emissive: 0x8a2a14 }));
scene.add(mars);
function ellipseLine(color, opacity, dashed) {
  const pos = new Float32Array(257 * 3), g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setDrawRange(0, 0);
  const m = dashed ? new THREE.LineDashedMaterial({ color, dashSize: 0.3, gapSize: 0.22, transparent: true, opacity }) : new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  const l = new THREE.Line(g, m); l.frustumCulled = false; scene.add(l); l.userData.pos = pos; return l;
}
const marsOrb0 = ellipseLine(0xff6a4d, 0.22, false), marsOrbNow = ellipseLine(0xff6a4d, 0.65, true);
function unitCircle(color, opacity) {
  const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(circlePts(1 / S, 96)), new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
  l.frustumCulled = false; scene.add(l); l.visible = false; return l;
}
const marsSafeRing = unitCircle(0xff4d4d, 0.9), marsCautRing = unitCircle(0xffcf4d, 0.7);
const emLineGeom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
const emLine = new THREE.Line(emLineGeom, new THREE.LineBasicMaterial({ color: 0xff9a80, transparent: true, opacity: 0.8 }));
emLine.frustumCulled = false; emLine.visible = false; scene.add(emLine);
// text labels that follow the bodies (HTML overlay)
const LBL = {};
for (const [k, c] of [['sun', '#ffcc33'], ['earth', '#7cc4ff'], ['moon', '#dddddd'], ['ast', '#e0a46a'], ['mars', '#ff8a6a'], ['emd', '#ffb39a'], ['pl0', '#c8c2b8'], ['pl1', '#f3d9a0'], ['pl2', '#e6c49a']]) {
  const d = document.createElement('div'); d.className = 'blabel mono small'; d.style.color = c; d.style.display = 'none';
  $('center').appendChild(d); LBL[k] = d;
}
// awayFrom (optional 3-D point): put the label on the side of the body facing away from that point
function placeLabel(k, pos3, text, dx = 8, dy = -8, awayFrom = null) {
  const el = LBL[k];
  if (!text || !OV.labels) { el.style.display = 'none'; return; }
  _lv.copy(pos3).project(camera);
  if (_lv.z > 1 || Math.abs(_lv.x) > 1 || Math.abs(_lv.y) > 1) { el.style.display = 'none'; return; }   // off-screen
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (el.textContent !== text) el.textContent = text;
  el.style.display = 'block';
  let x = (_lv.x * 0.5 + 0.5) * w, y = (-_lv.y * 0.5 + 0.5) * h;
  const lw = el.offsetWidth, lh = el.offsetHeight;
  if (awayFrom) {
    _lw.copy(awayFrom).project(camera);
    const ax = (_lw.x * 0.5 + 0.5) * w, ay = (-_lw.y * 0.5 + 0.5) * h;
    x += x < ax ? -Math.abs(dx) - lw : Math.abs(dx);
    y += y < ay ? -Math.abs(dy) - lh : Math.abs(dy);
  } else { x += dx; y += dy; }
  el.style.left = `${Math.max(2, Math.min(w - lw - 2, x))}px`; el.style.top = `${Math.max(2, Math.min(h - lh - 2, y))}px`;
}
const _lw = new THREE.Vector3();
function resolveLabels() {        // shift overlapping labels apart vertically (readability only)
  const placed = [];
  const dv = $('dvLabel');            // the recoil label stays where it is; body labels move around it
  if (dv.style.display !== 'none' && dv.style.left) placed.push({ x: parseFloat(dv.style.left), y: parseFloat(dv.style.top), w: dv.offsetWidth, h: dv.offsetHeight });
  for (const k of ['sun', 'earth', 'moon', 'ast', 'mars', 'emd', 'pl0', 'pl1', 'pl2']) {
    const el = LBL[k]; if (el.style.display === 'none') continue;
    let x = parseFloat(el.style.left), y = parseFloat(el.style.top);
    const w = el.offsetWidth, h = el.offsetHeight;
    for (let it = 0; it < 6; it++) {
      const hit = placed.find((b) => x < b.x + b.w && x + w > b.x && y < b.y + b.h && y + h > b.y);
      if (!hit) break;
      y = hit.y + hit.h + 2;
    }
    el.style.top = `${y}px`; placed.push({ x, y, w, h });
  }
}
const _lv = new THREE.Vector3();
const pov = createPov(renderer, canvas, $('center'));
const col = createCollectorView(renderer, canvas, $('center'));
const wx = createWeatherView(renderer, canvas, $('center'));
const povOn = () => $('viewMode').value === 'pov' || $('viewMode').value === 'beam';   // both use the Earth close-up
const colOn = () => $('viewMode').value === 'collector';
const wxOn = () => $('viewMode').value === 'weather';                                  // weather & life view
const closeUp = () => povOn() || colOn() || wxOn();                                    // any view other than the overview
const beamOn = () => $('viewMode').value === 'beam';
// ------------------------------------------------------------------ solar collectors + microwave beams (overview)
const SOL = () => (V.meta && V.meta.solar) || (V.init && V.init.solar) || null;
function powerAt(q) {             // beam power state of the event at row q (columns 30-33 of _viewer_rows)
  const sol = SOL();
  if (!sol || !sol.on) return { on: false };
  if (!q || q.length <= R_PLAUNCH) return { on: true, avail: sol.P_elec_1au_W, inW: 0, eta: sol.eta_1au, launch: 0 };
  return { on: true, avail: q[R_PAV], inW: q[R_PIN], eta: q[R_ETA], launch: q[R_PLAUNCH] };
}
let CURPOW = { on: false }, BEAM_I = 0;
const OVTHROW = { launching: false, dir: null };   // overview arms: are packets being launched right now, and where to
const NBELT = 1440;
const beltOvPos = new Float32Array(NBELT * 3), beltOvGeom = new THREE.BufferGeometry();
beltOvGeom.setAttribute('position', new THREE.BufferAttribute(beltOvPos, 3));
const beltOv = new THREE.Points(beltOvGeom, new THREE.PointsMaterial({ color: 0xffd27a, size: 0.05, transparent: true, opacity: 0.55, depthWrite: false }));
beltOv.visible = false; scene.add(beltOv);
const beamOvPos = new Float32Array(16 * 3), beamOvGeom = new THREE.BufferGeometry();
beamOvGeom.setAttribute('position', new THREE.BufferAttribute(beamOvPos, 3));
const beamOv = new THREE.LineSegments(beamOvGeom, new THREE.LineBasicMaterial({ color: 0xb48cff, transparent: true, opacity: 0.5, depthWrite: false, blending: THREE.AdditiveBlending }));
beamOv.visible = false; beamOv.frustumCulled = false; scene.add(beamOv);
let beltOvKey = '';
function updateOvSolar(dt, epos) {
  const sol = SOL(), on = !!(sol && sol.on) && !closeUp();
  const pw = CURPOW, tgt = !pw.on ? 0 : pw.inW > 0 ? 1 : 0.15;
  BEAM_I += (tgt - BEAM_I) * Math.min(1, dt * 4);
  beltOv.visible = on && OV.collectors;
  if (beltOv.visible) {
    const key = `${sol.orbit_au}|${sol.half_width_deg}`;
    if (key !== beltOvKey) {
      beltOvKey = key; const hw = sol.half_width_deg * Math.PI / 180;
      for (let k = 0; k < NBELT; k++) {
        const lon = 2 * Math.PI * ((k * 0.6180339887) % 1), lat = Math.asin(Math.sin(hw) * (2 * ((k * 0.7548776662) % 1) - 1));
        beltOvPos.set([sol.orbit_au * Math.cos(lat) * Math.cos(lon) * S, sol.orbit_au * Math.cos(lat) * Math.sin(lon) * S, sol.orbit_au * Math.sin(lat) * S], 3 * k);
      }
      beltOvGeom.attributes.position.needsUpdate = true;
    }
  }
  beamOv.visible = on && OV.beams && pw.on;
  if (beamOv.visible) {
    // 8 representative beams (one per arm) from belt points on Earth's side of the Sun to Earth
    const ang0 = Math.atan2(epos.y, epos.x);
    for (let k = 0; k < 8; k++) {
      const a = ang0 + (k - 3.5) * 0.22, lat = (k % 2 ? -1 : 1) * 0.3 * sol.half_width_deg * Math.PI / 180, R = sol.orbit_au * S;
      beamOvPos.set([R * Math.cos(lat) * Math.cos(a), R * Math.cos(lat) * Math.sin(a), R * Math.sin(lat), epos.x, epos.y, epos.z], 6 * k);
    }
    beamOvGeom.attributes.position.needsUpdate = true; beamOv.material.opacity = 0.08 + 0.6 * BEAM_I;
  }
}
const recoil = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 1.4, 0x7fe08a, 0.35, 0.2);
recoil.visible = false; scene.add(recoil);

// past trail (solid) – row positions up to the display time, plus the live point
const MAXTRAIL = 400000;
const TRAIL_SUB = 4;                // the past path follows the same Hermite curve as Earth (4 points per row interval)
const trailPos = new Float32Array((MAXTRAIL + 1) * 3), trailCol = new Float32Array((MAXTRAIL + 1) * 3);
const trailGeom = new THREE.BufferGeometry(); trailGeom.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
trailGeom.setAttribute('color', new THREE.BufferAttribute(trailCol, 3)); trailGeom.setDrawRange(0, 0);
// colour encodes simulated time: dark blue at year 0 -> light cyan at the end of the mission
const trailLine = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.6 }));
const trailDots = new THREE.Points(trailGeom, new THREE.PointsMaterial({ vertexColors: true, size: 0.06, transparent: true, opacity: 0.7 }));
trailLine.frustumCulled = trailDots.frustumCulled = false;
scene.add(trailLine); scene.add(trailDots);
// computed ahead (physics done, not yet displayed) – dim
const AHEADMAX = 4000;
const aheadPos = new Float32Array(AHEADMAX * 3);
const aheadGeom = new THREE.BufferGeometry(); aheadGeom.setAttribute('position', new THREE.BufferAttribute(aheadPos, 3)); aheadGeom.setDrawRange(0, 0);
const ahead = new THREE.Points(aheadGeom, new THREE.PointsMaterial({ color: 0x8090a8, size: 0.035, transparent: true, opacity: 0.3 }));
ahead.frustumCulled = false; scene.add(ahead);
// coasting (osculating) orbit – dashed
const coastPos = new Float32Array(257 * 3);
const coastGeom = new THREE.BufferGeometry(); coastGeom.setAttribute('position', new THREE.BufferAttribute(coastPos, 3)); coastGeom.setDrawRange(0, 0);
const coast = new THREE.Line(coastGeom, new THREE.LineDashedMaterial({ color: 0x7cc4ff, dashSize: 0.35, gapSize: 0.25, transparent: true, opacity: 0.6 }));
coast.frustumCulled = false; scene.add(coast);
// packets: line segments with per-vertex colour (brightness fades with wall-clock age)
const MAXPK = 600, MAXSEG = MAXPK * 25;
const pkPos = new Float32Array(MAXSEG * 6), pkCol = new Float32Array(MAXSEG * 6);
const pkGeom = new THREE.BufferGeometry();
pkGeom.setAttribute('position', new THREE.BufferAttribute(pkPos, 3)); pkGeom.setAttribute('color', new THREE.BufferAttribute(pkCol, 3)); pkGeom.setDrawRange(0, 0);
const pkLines = new THREE.LineSegments(pkGeom, new THREE.LineBasicMaterial({ vertexColors: true }));
pkLines.frustumCulled = false; scene.add(pkLines);
const headPos = new Float32Array(MAXPK * 3), headCol = new Float32Array(MAXPK * 3);
const headGeom = new THREE.BufferGeometry();
headGeom.setAttribute('position', new THREE.BufferAttribute(headPos, 3)); headGeom.setAttribute('color', new THREE.BufferAttribute(headCol, 3)); headGeom.setDrawRange(0, 0);
const heads = new THREE.Points(headGeom, new THREE.PointsMaterial({ size: 0.12, vertexColors: true }));
heads.frustumCulled = false; scene.add(heads);

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight, pr = renderer.getPixelRatio();
  if (canvas.width !== Math.floor(w * pr) || canvas.height !== Math.floor(h * pr)) {
    renderer.setSize(w, h, false); camera.aspect = w / Math.max(1, h); camera.updateProjectionMatrix();
  }
}

// ------------------------------------------------------------------ viewer state
const V = {
  run: -1, mode: 'idle', label: '', fidelity: '', policy: null, done: true, final: null, meta: null,
  rows: [], cumPk: [], cumDv: [], since: 0, moon: [], msince: 0, packets: [], psince: 0, pstrideRx: 1,
  decisions: [], trailSrv: null, disp: 0, playing: true, trailUpTo: -1, trailN: 0, gap: false, waiting: false,
  evRow: -1, evDec: 0, lastEj: -1, ejScan: -1, flags: {}, wallStart: performance.now(), produced: 0, povScan: -1,
  marsMin: Infinity, marsSh: 0, marsShT: -1e9, marsZone: 0, mPrev: null, mPrev2: null, mPassT: -1e9,
};
let speedY = 2;                        // simulated years per wall-clock second
let DOCS = null;                       // guide texts from /api/results
function speedFromSlider(v) { return DAY_Y * Math.pow(500 * 365.25, v / 1000); }
function updSpeed() { speedY = speedFromSlider(+$('speed').value); $('speedLbl').textContent = fmtSpeed(speedY); }
$('speed').addEventListener('input', updSpeed); updSpeed();

function resetViewer() {
  Object.assign(V, { rows: [], cumPk: [], cumDv: [], since: 0, moon: [], msince: 0, packets: [], psince: 0, decisions: [], bsince: 0, beltPend: {},
    trailSrv: null, disp: 0, trailUpTo: -1, trailN: 0, gap: false, waiting: false, evRow: -1, evDec: 0, lastEj: -1, ejScan: -1,
    flags: {}, wallStart: performance.now(), meta: null, povScan: -1, marsMin: Infinity, marsSh: 0, marsShT: -1e9, marsZone: 0, mPrev: null, mPrev2: null, mPassT: -1e9, plMin: null, plWarn: null });
  trailGeom.setDrawRange(0, 0); aheadGeom.setDrawRange(0, 0); coastGeom.setDrawRange(0, 0); pkGeom.setDrawRange(0, 0);
  headGeom.setDrawRange(0, 0); recoil.visible = false; $('evList').innerHTML = ''; evLines.length = 0;
}

function bsearch(arr, t) {          // last index with arr[i][0] <= t, or -1
  let lo = 0, hi = arr.length - 1, ans = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (arr[m][0] <= t) { ans = m; lo = m + 1; } else hi = m - 1; }
  return ans;
}

// cubic Hermite in (rho, theta, z); p in length unit L, v in L per time unit, h in the same time unit
function polarHermite(p0, v0, p1, v1, h, s) {
  const pol = (p, v) => { const r2 = p[0] * p[0] + p[1] * p[1], r = Math.sqrt(r2);
    return [r, Math.atan2(p[1], p[0]), p[2], (p[0] * v[0] + p[1] * v[1]) / r, (p[0] * v[1] - p[1] * v[0]) / r2, v[2]]; };
  const a = pol(p0, v0), b = pol(p1, v1);
  let th1 = b[1]; while (th1 - a[1] < -Math.PI) th1 += 2 * Math.PI; while (th1 - a[1] > Math.PI) th1 -= 2 * Math.PI;
  const s2 = s * s, s3 = s2 * s;
  const h00 = 2 * s3 - 3 * s2 + 1, h10 = s3 - 2 * s2 + s, h01 = -2 * s3 + 3 * s2, h11 = s3 - s2;
  const d00 = 6 * s2 - 6 * s, d10 = 3 * s2 - 4 * s + 1, d01 = -6 * s2 + 6 * s, d11 = 3 * s2 - 2 * s;
  const H = (f0, f1, g0, g1) => h00 * f0 + h10 * h * g0 + h01 * f1 + h11 * h * g1;
  const D = (f0, f1, g0, g1) => (d00 * f0 + d10 * h * g0 + d01 * f1 + d11 * h * g1) / h;
  const r = H(a[0], b[0], a[3], b[3]), th = H(a[1], th1, a[4], b[4]), z = H(a[2], b[2], a[5], b[5]);
  const dr = D(a[0], b[0], a[3], b[3]), dth = D(a[1], th1, a[4], b[4]), dz = D(a[2], b[2], a[5], b[5]);
  const c = Math.cos(th), sn = Math.sin(th);
  return { p: [r * c, r * sn, z], v: [dr * c - r * sn * dth, dr * sn + r * c * dth, dz] };
}
const rowP = (q) => [q[R_X], q[R_X + 1], q[R_X + 2]];
const rowV = (q, k) => [q[k] * KMS2AUY, q[k + 1] * KMS2AUY, q[k + 2] * KMS2AUY];

function eventSpacing() { return V.meta ? 1 / V.meta.ejections_per_year : 0.125; }
function earthAt(t) {             // -> {p [AU], v [AU/yr], i, gap, hold}
  const R = V.rows; if (!R.length) return null;
  const i = bsearch(R, t);
  if (i < 0) return { p: rowP(R[0]), v: rowV(R[0], R_VPRE), i: 0, hold: true };
  if (i >= R.length - 1) return { p: rowP(R[i]), v: rowV(R[i], R_VPOST), i, hold: true };
  const a = R[i], b = R[i + 1], h = b[R_T] - a[R_T];
  if (h > 2.5 * eventSpacing() + 1e-9 || h <= 0) return { p: rowP(a), v: rowV(a, R_VPOST), i, gap: true };
  const s = Math.max(0, Math.min(1, (t - a[R_T]) / h));
  const o = polarHermite(rowP(a), rowV(a, R_VPOST), rowP(b), rowV(b, R_VPRE), h, s);
  return { p: o.p, v: o.v, i };
}
function moonAt(t) {              // relative position [km] or null
  const M = V.moon; if (!M.length) return null;
  const i = bsearch(M, t); if (i < 0 || i >= M.length - 1) return null;
  const a = M[i], b = M[i + 1], h = b[0] - a[0];
  if (h <= 0 || h > 2.5 * eventSpacing() / 8 + 1e-9) return null;
  const k = YEAR_S / 1000; // m/s -> km/yr
  return polarHermite([a[1], a[2], a[3]], [a[4] * k, a[5] * k, a[6] * k], [b[1], b[2], b[3]], [b[4] * k, b[5] * k, b[6] * k], h, (t - a[0]) / h).p;
}
function moonStateAt(t) {         // nearest recorded Moon state at or before t: {r:[km], v:[km/s]}
  const M = V.moon; if (!M.length) return null;
  const i = bsearch(M, t); if (i < 0 || i >= M.length) return null;
  const a = M[i]; if (t - a[0] > 2.5 * eventSpacing() / 8 + 1e-9) return null;
  return { r: [a[1], a[2], a[3]], v: [a[4] / 1000, a[5] / 1000, a[6] / 1000] };
}
function drawMoonOrbit(center, ms, mAst) {   // osculating Earth-Moon ellipse (km) -> scene, x MOON_X
  if (!ms) { moonOrbGeom.setDrawRange(0, 0); return; }
  const mu = GM_EARTH_KM * (1 + M_MOON / M_EARTH + (mAst || 0) / M_EARTH);
  const r = ms.r, v = ms.v, rm = Math.hypot(...r), v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  const h = [r[1] * v[2] - r[2] * v[1], r[2] * v[0] - r[0] * v[2], r[0] * v[1] - r[1] * v[0]], hm = Math.hypot(...h);
  const eps = v2 / 2 - mu / rm; if (eps >= 0) { moonOrbGeom.setDrawRange(0, 0); return; }
  const a = -mu / (2 * eps);
  const ev = [(v[1] * h[2] - v[2] * h[1]) / mu - r[0] / rm, (v[2] * h[0] - v[0] * h[2]) / mu - r[1] / rm, (v[0] * h[1] - v[1] * h[0]) / mu - r[2] / rm];
  const e = Math.hypot(...ev);
  const P = e > 1e-6 ? ev.map((x) => x / e) : r.map((x) => x / rm);
  const hh = h.map((x) => x / hm);
  const Q = [hh[1] * P[2] - hh[2] * P[1], hh[2] * P[0] - hh[0] * P[2], hh[0] * P[1] - hh[1] * P[0]];
  const b = a * Math.sqrt(Math.max(0, 1 - e * e)), k = 1000 * MOON_X / AU * S;
  for (let j = 0; j <= 128; j++) {
    const E = j / 128 * 2 * Math.PI, x = a * (Math.cos(E) - e), y = b * Math.sin(E);
    moonOrbPos[3 * j] = center.x + (x * P[0] + y * Q[0]) * k; moonOrbPos[3 * j + 1] = center.y + (x * P[1] + y * Q[1]) * k;
    moonOrbPos[3 * j + 2] = center.z + (x * P[2] + y * Q[2]) * k;
  }
  moonOrbGeom.attributes.position.needsUpdate = true; moonOrbGeom.setDrawRange(0, 129);
}
function placeMoon(center, rel) {         // rel = Moon minus Earth [km]
  if (!rel) { moon.visible = false; moonLink.visible = false; placeLabel('moon', null, ''); return; }
  const k = 1000 * MOON_X / AU * S;
  moon.position.set(center.x + rel[0] * k, center.y + rel[1] * k, center.z + rel[2] * k); moon.visible = true;
  const lp = moonLinkGeom.attributes.position; lp.setXYZ(0, center.x, center.y, center.z); lp.setXYZ(1, moon.position.x, moon.position.y, moon.position.z);
  lp.needsUpdate = true; moonLink.visible = true;
  placeLabel('moon', moon.position, `Moon · ${int(Math.hypot(...rel))} km`, 8, 4, center);
}
function placeAsteroid(center, dir, frac, t) {
  if (!(frac > 0)) { asteroid.visible = false; placeLabel('ast', center, frac === 0 ? 'asteroid used up' : '', 10, 14); return; }
  const ra = Math.max(0.06, AST_RMAX * Math.cbrt(frac));
  asteroid.scale.setScalar(ra); asteroid.visible = true;
  // docked on the launch ring's AXIS (ecliptic north of Earth, update p36): its Earth-facing side rests on the grab point
  // GRAB_Z Earth radii from Earth's centre, the same distance from all 8 arm roots (marker enlarged)
  asteroid.position.set(center.x, center.y, center.z + EARTH_R * GRAB_Z + 0.9 * ra);
  asteroid.rotation.z = t * 3;
  placeLabel('ast', asteroid.position, `asteroid (docked) · ${(100 * frac).toFixed(frac < 0.1 ? 2 : 1)} % left`, 8, 6, center);
}
// ------------------------------------------------------------------ Mars
const OV = { marsOrbit: true, marsSafe: true, coast: true, ahead: true, labels: true, packets: true, collectors: true, beams: true, tracked: true, mirrors: true, arms: true, planets: true, planetOrbits: true };   // overview layer toggles
// ------------------------------------------------------------------ Mercury, Venus, Jupiter (update p35; see planets.js)
const PLAN = createPlanets(scene, S);
let PLANETS = null;          // {d: [AU] x3, min: [AU] x3} for the HUD
const plSafe = () => ((V.meta && V.meta.planets) || (V.init && V.init.planets) || {}).safe_au || [0.0197, 0.0483, 0.353];
function hidePlanets() { PLAN.hide(); for (let k = 0; k < 3; k++) placeLabel('pl' + k, null, ''); PLANETS = null; }
function placePlanets(states, earthP) {
  const d = PLAN.update(states, earthP, { on: OV.planets, orbits: OV.planetOrbits });
  if (!d) { hidePlanets(); return; }
  const safe = plSafe(), mins = V.mode === 'playback' ? (V.plMin = V.plMin || [Infinity, Infinity, Infinity]) : [NaN, NaN, NaN];
  for (let k = 0; k < 3; k++) {
    if (V.mode === 'playback') mins[k] = Math.min(mins[k], d[k]);
    placeLabel('pl' + k, PLAN.meshes[k].position, `${PL_NAMES[k]}${Number.isFinite(d[k]) ? ` · ${num(d[k], 2)} AU from Earth` : ''}`, 8, -8, sun.position);
    if (V.mode === 'playback' && Number.isFinite(d[k]) && d[k] < 3 * safe[k] && !(V.plWarn || [])[k]) {
      (V.plWarn = V.plWarn || [])[k] = true;
      ev(V.disp, `Earth within 3x the safe distance of ${PL_NAMES[k]}: ${num(d[k], 4)} AU (safe limit ${num(safe[k], 4)} AU).`, d[k] < safe[k] ? 'bad' : 'mars');
    }
  }
  PLANETS = { d, min: mins };
}
const rowM = (q) => [q[R_MARS], q[R_MARS + 1], q[R_MARS + 2]];
const rowMV = (q) => [q[R_MARSV] * KMS2AUY, q[R_MARSV + 1] * KMS2AUY, q[R_MARSV + 2] * KMS2AUY];
function marsAt(t) {               // -> {p [AU], v [AU/yr]} from the recorded Mars states (no ejection kinks: Mars is only pulled)
  const R = V.rows; if (!R.length || R[0].length <= R_SHIELD) return null;
  const i = bsearch(R, t);
  if (i < 0) return { p: rowM(R[0]), v: rowMV(R[0]) };
  if (i >= R.length - 1) return { p: rowM(R[i]), v: rowMV(R[i]) };
  const a = R[i], b = R[i + 1], h = b[R_T] - a[R_T];
  if (h > 2.5 * eventSpacing() + 1e-9 || h <= 0) return { p: rowM(a), v: rowMV(a) };
  return polarHermite(rowM(a), rowMV(a), rowM(b), rowMV(b), h, Math.max(0, Math.min(1, (t - a[R_T]) / h)));
}
const muMarsAU = () => (GM_SUN + GM_MARS) * YEAR_S * YEAR_S / (AU * AU * AU);
function orbitOf(p, v, mu) {
  const r = Math.hypot(...p), v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  const h = [p[1] * v[2] - p[2] * v[1], p[2] * v[0] - p[0] * v[2], p[0] * v[1] - p[1] * v[0]], hm = Math.hypot(...h);
  const eps = v2 / 2 - mu / r;
  const ev = [(v[1] * h[2] - v[2] * h[1]) / mu - p[0] / r, (v[2] * h[0] - v[0] * h[2]) / mu - p[1] / r, (v[0] * h[1] - v[1] * h[0]) / mu - p[2] / r];
  return { a: eps < 0 ? -mu / (2 * eps) : Infinity, e: Math.hypot(...ev), ev, h, hm, r };
}
function drawEllipse(line, el, p) {
  const g = line.geometry, pos = line.userData.pos;
  if (!(el.e < 0.99) || !Number.isFinite(el.a)) { g.setDrawRange(0, 0); return; }
  const P = el.e > 1e-9 ? el.ev.map((x) => x / el.e) : p.map((x) => x / el.r);
  const W = el.h.map((x) => x / el.hm), Q = [W[1] * P[2] - W[2] * P[1], W[2] * P[0] - W[0] * P[2], W[0] * P[1] - W[1] * P[0]];
  const b = el.a * Math.sqrt(1 - el.e * el.e);
  for (let k = 0; k <= 256; k++) { const E = k / 256 * 2 * Math.PI, x = el.a * (Math.cos(E) - el.e), y = b * Math.sin(E); for (let c = 0; c < 3; c++) pos[k * 3 + c] = (x * P[c] + y * Q[c]) * S; }
  g.attributes.position.needsUpdate = true; g.setDrawRange(0, 257);
  if (line.material.isLineDashedMaterial) line.computeLineDistances();
}
let marsFrame = 0;
function hideMars() {
  mars.visible = false; marsOrb0.visible = marsOrbNow.visible = false; marsSafeRing.visible = marsCautRing.visible = emLine.visible = false;
  placeLabel('mars', null, ''); placeLabel('emd', null, '');
}
const marsSafeAU = () => (V.meta && V.meta.mars_safe_au) || (V.init && V.init.mars_safe_au) || 0.0246;
const marsCautAU = () => (V.meta && V.meta.mars_caution_au) || (V.init && V.init.mars_caution_au) || 0.049;
function marsCls(d) { return d < marsSafeAU() ? 'bad' : d < marsCautAU() ? 'warn' : 'good'; }
function placeMars(ms, earthP, start) {   // ms = {p, v} [AU, AU/yr]; start = {p, v} for the start orbit
  if (!ms) { hideMars(); return null; }
  mars.position.set(ms.p[0] * S, ms.p[1] * S, ms.p[2] * S); mars.visible = true;
  const el = orbitOf(ms.p, ms.v, muMarsAU());
  marsOrb0.visible = marsOrbNow.visible = OV.marsOrbit;
  if (OV.marsOrbit) {
    if (start && marsOrb0.userData.key !== V.run + ':' + start.p[0]) { marsOrb0.userData.key = V.run + ':' + start.p[0]; drawEllipse(marsOrb0, orbitOf(start.p, start.v, muMarsAU()), start.p); }
    if ((marsFrame++ % 6) === 0) drawEllipse(marsOrbNow, el, ms.p);
  }
  const d = Math.hypot(ms.p[0] - earthP[0], ms.p[1] - earthP[1], ms.p[2] - earthP[2]);
  const near = OV.marsSafe && d < 0.3;
  marsSafeRing.visible = marsCautRing.visible = emLine.visible = near;
  if (near) {
    marsSafeRing.position.copy(mars.position); marsSafeRing.scale.setScalar(marsSafeAU() * S);
    marsCautRing.position.copy(mars.position); marsCautRing.scale.setScalar(marsCautAU() * S);
    const lp = emLineGeom.attributes.position; lp.setXYZ(0, earthP[0] * S, earthP[1] * S, earthP[2] * S); lp.setXYZ(1, mars.position.x, mars.position.y, mars.position.z); lp.needsUpdate = true;
    emLine.material.color.set(d < marsSafeAU() ? 0xff4d4d : d < marsCautAU() ? 0xffcf4d : 0x7fe08a);
    _mid.set((earthP[0] * S + mars.position.x) / 2, (earthP[1] * S + mars.position.y) / 2, (earthP[2] * S + mars.position.z) / 2);
    placeLabel('emd', _mid, `Earth–Mars ${num(d, 4)} AU (safe ≥ ${num(marsSafeAU(), 4)})`, 6, 6);
  } else placeLabel('emd', null, '');
  placeLabel('mars', mars.position, `Mars · ${num(d, 3)} AU from Earth`, 8, -8, sun.position);
  return { d, el };
}
const _mid = new THREE.Vector3();
function muAU() { return (V.meta ? V.meta.mu : 1.3271284e20) * YEAR_S * YEAR_S / (AU * AU * AU); }
function elements(p, v) {
  const mu = muAU(), r = Math.hypot(...p), v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  const h = [p[1] * v[2] - p[2] * v[1], p[2] * v[0] - p[0] * v[2], p[0] * v[1] - p[1] * v[0]], hm = Math.hypot(...h);
  const eps = v2 / 2 - mu / r;
  const ev = [(v[1] * h[2] - v[2] * h[1]) / mu - p[0] / r, (v[2] * h[0] - v[0] * h[2]) / mu - p[1] / r, (v[0] * h[1] - v[1] * h[0]) / mu - p[2] / r];
  const e = Math.hypot(...ev);
  return { a: eps < 0 ? -mu / (2 * eps) : Infinity, e, inc: Math.acos(Math.max(-1, Math.min(1, h[2] / hm))) * 180 / Math.PI, eps, r, speed: Math.sqrt(v2) / KMS2AUY, ev, h, hm };
}

// ------------------------------------------------------------------ event log
const evLines = [];
function ev(t, text, cls = '') {
  evLines.unshift(`<div class="${cls}"><b>${fmtYear(t)}</b>: ${text}</div>`);
  if (evLines.length > 60) evLines.length = 60;
  evDirty = true;
}
let evDirty = false;
function processEvents(t, i) {
  const m0 = V.meta ? V.meta.m_ast0 : 1, E0 = V.meta ? V.meta.E0 : 1;
  let perPacket = speedY <= 1.0, shown = 0, fb = 0, mh = 0;
  for (let k = V.evRow + 1; k <= i && k < V.rows.length; k++) {
    const q = V.rows[k];
    if (q.length > R_SHIELD) marsEvents(q);
    if (q.length > R_PIN && q[R_PIN] > 0 && !V.flags.beam) {
      V.flags.beam = true;
      ev(q[R_T], `first microwave power absorbed by the 8 arm rectennas: ${sci(q[R_PIN])} W (capture ${pct(q[R_ETA])}); the launcher draws ${sci(q[R_PLAUNCH])} W – ${pct(Math.min(1, q[R_PIN] * (SOL() ? SOL().rx : 1) / Math.max(1, q[R_PLAUNCH])), 0)} of it from the beams.`, 'sp');
    }
    if (q[R_DM] > 0) {
      if (!V.flags.first) { V.flags.first = true; ev(q[R_T], `first packet launched: ${sci(q[R_DM])} kg at ${int(q[R_U] / 1000)} km/s. Earth recoils by ${num(q[R_DV], 3)} m/s.`, 'sp'); }
      if (q[R_FATE] === 3) fb++;
      if (q[R_FATE] === 4) mh++;
      if (perPacket && shown < 12) {
        shown++;
        ev(q[R_T], `ejected ${sci(q[R_DM])} kg at ${int(q[R_U] / 1000)} km/s → ${FATE[q[R_FATE]]}; Earth Δv ${num(q[R_DV], 3)} m/s; asteroid left ${sci(q[R_MAST], 4)} kg (${pct(q[R_MAST] / m0, 2)})`, 'pk');
      }
      if (q[R_MAST] <= 0 && !V.flags.mass0) { V.flags.mass0 = true; ev(q[R_T], 'asteroid completely used up – no reaction mass left (resources exhausted).', 'bad'); }
      if (q[R_EREM] <= 1e-9 * E0 && !V.flags.e0) { V.flags.e0 = true; ev(q[R_T], 'propulsion energy budget used up (resources exhausted).', 'bad'); }
    }
    V.evRow = k;
  }
  if (perPacket && shown >= 12) ev(t, `(more packets this frame not listed – slow down to see each one)`, 'dim');
  if (fb) ev(t, `${fb} packet(s) FELL BACK to Earth: launched slower than escape speed (11.2 km/s), so no net push and energy wasted.`, 'bad');
  if (mh) ev(t, `${mh} packet(s) hit the Moon (momentum transferred to the Moon).`, 'sp');
  while (V.evDec < V.decisions.length && V.decisions[V.evDec].t_years <= t + 1e-9) {
    const d = V.decisions[V.evDec], prev = V.evDec > 0 ? V.decisions[V.evDec - 1] : null;
    const t0 = d.t_years - d.years;
    if (d.dm_kg > 0 && !perPacket) {
      ev(d.t_years, `years ${int(t0)}–${int(d.t_years)}: ${int(d.n_packets)} packets of ${sci(d.dm_kg)} kg at ${int(d.u_ms / 1000)} km/s, ` +
        `thrust ${dirWords(d.d_rtn)} (prograde cosine ${num(d.d_rtn[1], 2)}); Δv ${num(d.dv_ms, 1)} m/s; a = ${num(d.a_au)} AU, e = ${num(d.e)}; ` +
        `asteroid left ${pct(d.m_ast_frac)}, energy left ${pct(d.energy_frac)}`);
    }
    if (d.dm_kg <= 0 && (!prev || prev.dm_kg > 0) && V.policy !== 'none') ev(t0, 'agent stops ejecting (coasting).', 'sp');
    if (d.dm_kg > 0 && prev && prev.dm_kg <= 0) ev(t0, 'ejections resume.', 'sp');
    if (d.moon_bound === false && !V.flags.moon) { V.flags.moon = true; ev(d.t_years, 'MOON LOST – no longer bound to Earth.', 'bad'); }
    if (d.reason) {
      const txt = DOCS && DOCS.reason_docs[d.reason] ? DOCS.reason_docs[d.reason] : d.reason;
      ev(d.t_years, `MISSION END – ${d.reason}: ${txt} Final a = ${num(d.a_au)} AU, e = ${num(d.e)}.`, d.reason === 'success' ? 'ok' : 'bad');
    }
    V.evDec++;
  }
  if (V.done && V.final && V.final.validation && !V.flags.valid && V.rows.length && t >= V.rows[V.rows.length - 1][R_T] - 1e-9) {
    V.flags.valid = true; const vv = V.final.validation;
    ev(t, `stability check (${int(vv.years)} years without thrust, high-fidelity): ${vv.passed ? 'PASSED' : 'FAILED'} – worst |a − target| ${num(vv.max_abs_da_au)} AU, worst e ${num(vv.max_e)}, Moon ${vv.moon_bound ? 'bound' : 'LOST'}.`, vv.passed ? 'ok' : 'bad');
  }
  if (evDirty) { evDirty = false; $('evList').innerHTML = evLines.slice(0, 10).join(''); }
}

function marsEvents(q) {            // planetary-safety events from the recorded Mars distance and shield mode
  const d = q[R_DMARS], t = q[R_T], safe = marsSafeAU(), caut = marsCautAU();
  if (!(d > 0)) return;
  if (d < (V.marsMin ?? Infinity)) V.marsMin = d;
  const zone = d < safe ? 2 : d < caut ? 1 : 0;
  if (zone !== V.marsZone) {
    if (zone === 2) ev(t, `EARTH INSIDE THE MARS-SAFE DISTANCE: ${num(d, 4)} AU < ${num(safe, 4)} AU (Mars's tide / pull on the Moon above the limits).`, 'bad');
    else if (zone === 1 && V.marsZone === 0) ev(t, `Earth inside the shield's caution distance of Mars: ${num(d, 4)} AU (safe limit ${num(safe, 4)} AU).`, 'mars');
    else if (zone === 0) ev(t, `Earth left the Mars caution zone (${num(d, 4)} AU).`, 'mars');
    V.marsZone = zone;
  }
  const a = V.mPrev2, b = V.mPrev;                 // local minimum of the distance = a close pass
  if (a !== null && b !== null && b < a && d > b && b < 0.1 && (t - V.mPassT > 20 || b <= V.marsMin + 1e-12)) {
    V.mPassT = t; ev(t, `close pass of Mars: ${num(b, 4)} AU (${int(b * AU / 1000)} km) – ${b >= safe ? 'outside' : 'INSIDE'} the safe distance ${num(safe, 4)} AU. Both planets' orbits are bent by their mutual pull.`, b >= safe ? 'mars' : 'bad');
  }
  V.mPrev2 = b; V.mPrev = d;
  const sh = q[R_SHIELD];
  if (sh !== V.marsSh) {
    if (t - V.marsShT > 10 || (sh && !V.marsSh)) {
      ev(t, sh ? `Mars shield overrides the agent: ${SHIELD_TXT[sh]}. Its forecast showed the agent's command would bring Earth within ${num(caut, 4)} AU of Mars. (No energy or momentum is created – only the packet size changes.)`
        : 'Mars shield: control returned to the agent.', 'mars');
      V.marsShT = t;
    }
    V.marsSh = sh;
  }
}

// ------------------------------------------------------------------ drawing helpers
function setTrailVertex(n, p, t) {
  trailPos[n * 3] = p[0] * S; trailPos[n * 3 + 1] = p[1] * S; trailPos[n * 3 + 2] = p[2] * S;
  const f = Math.max(0, Math.min(1, (t ?? 0) / (V.meta ? V.meta.mission_years : 10000)));
  trailCol[n * 3] = 0.10 + 0.55 * f; trailCol[n * 3 + 1] = 0.22 + 0.66 * f; trailCol[n * 3 + 2] = 0.50 + 0.50 * f;
}
function updateTrail(i, live) {
  if (V.trailN === 0 && V.trailSrv && V.trailSrv.length && V.rows.length) {
    const t0 = V.rows[0][R_T];     // joined mid-run: decimated path before the first received row
    for (const p of V.trailSrv) { if (V.trailN >= MAXTRAIL || p[0] >= t0) break; setTrailVertex(V.trailN++, p.slice(1), p[0]); }
  }
  const sub = V.rows.length * TRAIL_SUB > MAXTRAIL * 0.9 ? 1 : TRAIL_SUB;
  for (let k = V.trailUpTo + 1; k <= i; k++) {
    if (k > 0 && sub > 1) {
      const a = V.rows[k - 1], b = V.rows[k], h = b[R_T] - a[R_T];
      if (h > 0 && h <= 2.5 * eventSpacing() + 1e-9) {
        for (let j = 1; j < sub && V.trailN < MAXTRAIL; j++) setTrailVertex(V.trailN++, polarHermite(rowP(a), rowV(a, R_VPOST), rowP(b), rowV(b, R_VPRE), h, j / sub).p, a[R_T] + h * j / sub);
      }
    }
    if (V.trailN < MAXTRAIL) setTrailVertex(V.trailN++, rowP(V.rows[k]), V.rows[k][R_T]);
    V.trailUpTo = k;
  }
  setTrailVertex(V.trailN, live, V.disp);
  trailGeom.attributes.position.needsUpdate = true; trailGeom.attributes.color.needsUpdate = true; trailGeom.setDrawRange(0, V.trailN + 1);
}
let aheadT = 0;
function updateAhead(i) {
  const now = performance.now(); if (now - aheadT < 300) return; aheadT = now;
  const n = V.rows.length - 1 - i; if (n <= 0) { aheadGeom.setDrawRange(0, 0); return; }
  const st = Math.max(1, Math.ceil(n / AHEADMAX)); let c = 0;
  for (let k = i + 1; k < V.rows.length && c < AHEADMAX; k += st) { const q = V.rows[k]; aheadPos[c * 3] = q[R_X] * S; aheadPos[c * 3 + 1] = q[R_X + 1] * S; aheadPos[c * 3 + 2] = q[R_X + 2] * S; c++; }
  aheadGeom.attributes.position.needsUpdate = true; aheadGeom.setDrawRange(0, c);
}
let coastFrame = 0;
function updateCoast(p, v, el) {
  if ((coastFrame++ % 6) !== 0) return;
  if (!(el.e < 0.99) || !Number.isFinite(el.a)) { coastGeom.setDrawRange(0, 0); return; }
  const P = el.e > 1e-9 ? el.ev.map((x) => x / el.e) : p.map((x) => x / el.r);
  const W = el.h.map((x) => x / el.hm);
  const Q = [W[1] * P[2] - W[2] * P[1], W[2] * P[0] - W[0] * P[2], W[0] * P[1] - W[1] * P[0]];
  const b = el.a * Math.sqrt(1 - el.e * el.e);
  for (let k = 0; k <= 256; k++) {
    const E = k / 256 * 2 * Math.PI, x = el.a * (Math.cos(E) - el.e), y = b * Math.sin(E);
    for (let c = 0; c < 3; c++) coastPos[k * 3 + c] = (x * P[c] + y * Q[c]) * S;
  }
  coastGeom.attributes.position.needsUpdate = true; coastGeom.setDrawRange(0, 257); coast.computeLineDistances();
}
function astOffsetDir(i) {        // direction of the latest ejection at or before row i
  for (let k = Math.min(i, V.rows.length - 1); k >= 0 && k > i - 4000; k--) { const q = V.rows[k]; if (q[R_DM] > 0) return [q[R_EHAT], q[R_EHAT + 1], q[R_EHAT + 2]]; }
  const q = V.rows[Math.max(0, Math.min(i, V.rows.length - 1))]; if (!q) return [1, 0, 0];
  const v = [q[R_VPRE], q[R_VPRE + 1], q[R_VPRE + 2]], m = Math.hypot(...v) || 1; return v.map((x) => -x / m);
}
function rtnOf(e, p, v) {         // components of the inertial unit vector e in the radial / along-track / normal frame of (p, v)
  const rn = Math.hypot(...p), r = p.map((x) => x / rn);
  const h = [p[1] * v[2] - p[2] * v[1], p[2] * v[0] - p[0] * v[2], p[0] * v[1] - p[1] * v[0]], hn = Math.hypot(...h) || 1, n = h.map((x) => x / hn);
  const t = [n[1] * r[2] - n[2] * r[1], n[2] * r[0] - n[0] * r[2], n[0] * r[1] - n[1] * r[0]];
  const d = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  return [d(e, r), d(e, t), d(e, n)];
}
function astRtn(i) {               // throw direction of the latest ejection, in that moment's orbital frame
  for (let k = Math.min(i, V.rows.length - 1); k >= 0 && k > i - 4000; k--) { const q = V.rows[k]; if (q[R_DM] > 0) return rtnOf([q[R_EHAT], q[R_EHAT + 1], q[R_EHAT + 2]], rowP(q), rowV(q, R_VPRE)); }
  return [0, -1, 0];
}
const PKCOL = { 1: [1.0, 0.6, 0.3], 2: [1.0, 0.88, 0.3], 3: [1.0, 0.35, 0.35], 4: [0.75, 0.75, 0.75] };
let pkNote = '';
function updatePackets(t) {
  const off = V.meta ? V.meta.offsets_days : [0];
  const persist = Math.max(PERSIST_WALL_S * speedY, 2 * DAY_Y);
  // drop records that can no longer be visible
  let cut = 0; while (cut < V.packets.length && V.packets[cut][0] + 1.0 + persist < t) cut++;
  if (cut > 200) V.packets.splice(0, cut);
  const lo = bsearch(V.packets, t - 1.0 - persist) + 1, hi = bsearch(V.packets, t);
  const cand = Math.max(0, hi - lo + 1);
  const step = Math.max(1, Math.ceil(cand / MAXPK));
  pkNote = (step > 1 || V.pstrideRx > 1) ? `Packets: drawing 1 of every ${step * V.pstrideRx} at this speed (all are in the physics).` : '';
  let ns = 0, nh = 0;
  for (let j = lo; j <= hi && j >= 0; j += step) {
    const q = V.packets[j]; const tp = q[0], fate = q[1], n = q[4];
    const ageD = (t - tp) * 365.25; if (ageD < 0) continue;
    const lifeD = off[n - 1] || 0;
    const fade = ageD <= lifeD ? 1 : 1 - (ageD - lifeD) / (persist * 365.25);
    if (fade <= 0.02) continue;
    const col = PKCOL[fate] || [1, 1, 1];
    // visual offset: shift the path so it starts at the drawn asteroid (AST_OFFSET_AU from Earth's centre)
    let ox = 0, oy = 0, oz = 0;
    oz = AST_OFFSET_AU;        // the grab point lies on the ring axis (ecliptic north of Earth's centre)
    if (n === 1) {             // fell back / hit Moon: marker at Earth
      if (nh < MAXPK) { headPos.set([(q[5] + ox) * S, (q[6] + oy) * S, (q[7] + oz) * S], nh * 3); headCol.set(col.map((c) => c * fade), nh * 3); nh++; }
      continue;
    }
    const aD = Math.min(ageD, lifeD);
    let last = null;
    for (let k = 0; k < n - 1 && ns < MAXSEG; k++) {
      if (off[k] > aD) break;
      const f = off[k + 1] <= aD ? 1 : (aD - off[k]) / (off[k + 1] - off[k]);
      const b0 = 5 + 3 * k, b1 = b0 + 3;
      const x0 = q[b0] + ox, y0 = q[b0 + 1] + oy, z0 = q[b0 + 2] + oz;
      const x1 = q[b0] + (q[b1] - q[b0]) * f + ox, y1 = q[b0 + 1] + (q[b1 + 1] - q[b0 + 1]) * f + oy, z1 = q[b0 + 2] + (q[b1 + 2] - q[b0 + 2]) * f + oz;
      pkPos.set([x0 * S, y0 * S, z0 * S, x1 * S, y1 * S, z1 * S], ns * 6);
      const c = col.map((v) => v * fade * 0.85);
      pkCol.set([...c, ...c], ns * 6); ns++; last = [x1, y1, z1];
    }
    if (last && ageD <= lifeD && nh < MAXPK) { headPos.set(last.map((x) => x * S), nh * 3); headCol.set(col, nh * 3); nh++; }
  }
  pkGeom.attributes.position.needsUpdate = true; pkGeom.attributes.color.needsUpdate = true; pkGeom.setDrawRange(0, ns * 2);
  headGeom.attributes.position.needsUpdate = true; headGeom.attributes.color.needsUpdate = true; headGeom.setDrawRange(0, nh);
}
function trimMoon(t) { let c = 0; while (c < V.moon.length - 2 && V.moon[c + 1][0] < t - 0.05) c++; if (c > 500) V.moon.splice(0, c); }

// ------------------------------------------------------------------ HUD
function tip(key) {
  const o = DOCS && DOCS.outputs[key]; if (!o) return '';
  return esc(`${o.label} [${o.unit}] — ${o.measures}\nHow: ${o.how}\nType: ${o.kind}\nDesirable: ${o.better}\nMeaning for success: ${o.success}`);
}
function hrow(key, label, value, cls = '') { return `<div class="row ${cls}" title="${tip(key)}"><span>${label}</span><span>${value}</span></div>`; }
function decAt(t) { let d = null; for (let k = V.decisions.length - 1; k >= 0; k--) if (V.decisions[k].t_years <= t + 1e-9) { d = V.decisions[k]; break; } return d; }
// the decision record whose interval CONTAINS t (records are written at the END of each interval)
function decFor(t) { const D = V.decisions; if (!D.length) return null; let lo = 0, hi = D.length - 1; if (D[hi].t_years < t - 1e-9) return null;
  while (lo < hi) { const m = (lo + hi) >> 1; if (D[m].t_years >= t - 1e-9) hi = m; else lo = m + 1; } return D[lo]; }
function climNow(t) {             // climate at display time t from the recorded climate track
  const cl = V.meta && V.meta.climate; if (V.mode !== 'playback' || !cl) return null;
  const d = decFor(t), c = d ? climateAtDec(d, t) : { T: cl.T0, sun: NaN, mirW: 0 };
  return c ? { ...c, mirRel: (c.mirW || 0) / cl.earth_sun_W, ok: Math.abs(c.T - cl.target) <= cl.tol, cl, dec: d } : null;
}
function climRows(t) {
  const c = climNow(t); if (!c || !c.cl.on) return '';
  return hrow('T_surface', 'surface temperature', `${num(c.T, 1)} K <span class="dim">(target ${c.cl.target} ± ${c.cl.tol})</span>`, c.ok ? '' : 'warn') +
    hrow('mirror_light', 'sunlight + space mirrors', `${num(c.sun, 2)}× + ${num(c.mirRel, 2)}× today`);
}
/** Everything the weather view needs, from the recorded climate (or today's Earth when nothing is playing). */
function weatherState() {
  const t = V.mode === 'playback' ? V.disp : 0, c = climNow(t), cl = (V.meta && V.meta.climate) || null, d = c ? c.dec : null;
  return { has: !!c, t, T: c ? c.T : 288, sun: c && Number.isFinite(c.sun) ? c.sun : 1, mirRel: c ? c.mirRel : 0,
    capRel: d ? d.mirror_cap_rel : cl && cl.mirrors ? cl.mirrors.cap_rel : 1, Tno: d ? d.T_no_mirror_K : NaN, noTxt: d ? d.life_no_mirror : '',
    rAU: earth.position.length() / S || 1, md: cl ? cl.mirrors : null, life: cl && cl.life ? { dT: cl.life.dT, T_ice: cl.life.T_ice, T_grow: cl.life.T_grow } : undefined,
    playing: V.mode === 'playback' && V.playing, speedY };
}
function decActive(t) { for (const d of V.decisions) if (d.t_years > t + 1e-9) return d; return null; }
let hudMore = false;
function setText(id, txt) { const e = $(id); if (e.textContent !== txt) e.textContent = txt; }
function setHtml(id, h) { const e = $(id); if (e._h !== h) { e._h = h; e.innerHTML = h; } }
function marsRows(t, full) {          // Mars / planetary-safety telemetry from the recorded Mars state
  if (!MARS || !Number.isFinite(MARS.d)) return V.mode === 'training' ? '' : `<div class="dim small">Mars: no recorded state yet.</div>`;
  const d = MARS.d, safe = marsSafeAU(), caut = marsCautAU();
  let h = hrow('mars_dist', 'distance to Mars', `${num(d, 4)} AU`, marsCls(d));
  if (!full) {
    if (MARS.shield) h += hrow('shield', 'Mars shield', SHIELD_TXT[MARS.shield], 'warn');
    return h;
  }
  const mn = Math.min(V.marsMin ?? Infinity, d);
  h += hrow('mars_min_dist', 'closest so far', Number.isFinite(mn) ? `${num(mn, 4)} AU` : 'N/A', Number.isFinite(mn) ? marsCls(mn) : '');
  h += hrow('mars_safe', 'safe / caution distance', `${num(safe, 4)} / ${num(caut, 4)} AU`);
  const dm = d * AU, tide = (GM_MARS / dm ** 3) / (4.9028e12 / 3.844e8 ** 3);
  h += hrow('mars_tide', "Mars's tide on Earth now", `${sci(tide, 2)} × Moon's`, tide > 0.01 ? 'bad' : '');
  const dec = decAt(t);
  h += hrow('shield', 'shield now / overrides', `${SHIELD_TXT[MARS.shield || 0]} · ${int(dec ? dec.shield_overrides_cum : 0)}`, MARS.shield ? 'warn' : '');
  if (MARS.el && Number.isFinite(MARS.el.a)) {
    const r0 = V.rows[0], el0 = r0 && r0.length > R_SHIELD && r0[R_T] === 0 ? orbitOf(rowM(r0), rowMV(r0), muMarsAU()) : null;
    h += hrow('mars_orbit', "Mars's orbit a, e", `${num(MARS.el.a, 4)} AU, ${num(MARS.el.e, 4)}` + (el0 ? ` <span class="dim">(Δa ${MARS.el.a - el0.a >= 0 ? '+' : ''}${num(MARS.el.a - el0.a, 4)})</span>` : ''));
  }
  return h;
}
function planetRows(full) {           // Mercury / Venus / Jupiter telemetry (update p35)
  if (!PLANETS || !OV.planets) return '';
  const safe = plSafe();
  let h = '';
  for (let k = 0; k < 3; k++) {
    const d = PLANETS.d[k], cls = d < safe[k] ? 'bad' : d < 3 * safe[k] ? 'warn' : '';
    h += hrow('planet_dist', `distance to ${PL_NAMES[k]}`, `${num(d, 3)} AU` + (full && Number.isFinite(PLANETS.min[k]) ? ` <span class="dim">(closest ${num(PLANETS.min[k], 3)}, safe ${num(safe[k], 3)})</span>` : ''), cls);
  }
  return h;
}
function compactHud(t, st, el, i) {
  let h = '';
  if (V.mode === 'idle') {
    h += `<div>Nothing running – press <b>▶</b> below (or Space).</div><div class="dim small">Showing the simulated start state (year 0).</div>`;
    h += marsRows(t, false);
  } else if (V.mode === 'training') {
    h += `<div>LIVE TRAINING · ${esc(V.label)}</div><div class="dim small">${fmtYear(t)} simulated (replay)</div>`;
    if (V.final) h += hrow('a', 'last episode a', `${num(V.final.a_au)} AU`) + `<div class="row"><span>outcome</span><span>${esc(V.final.reason)}</span></div>`;
  } else if (st && el) {
    const q = V.rows[Math.min(i, V.rows.length - 1)];
    h += `<div class="dim">${esc(V.label)}</div>`;
    h += hrow('a', 'orbit size a', `${num(el.a)} AU <span class="dim">→ ${V.meta.target_au}</span>`);
    h += hrow('e', 'eccentricity e', num(el.e));
    h += hrow('m_ast', 'asteroid left', pct(q[R_MAST] / V.meta.m_ast0), q[R_MAST] <= 0 ? 'warn' : '');
    h += marsRows(t, false);
    h += climRows(t);
    if (CURPOW.on) h += hrow('power_beam', 'beam power in', CURPOW.inW > 0 ? `${sci(CURPOW.inW)} W` : '<span class="dim">idle</span>');
    if (povOn() && pov.P.env) {
      const e = pov.P.env;
      h += hrow('env_sw', 'solar wind', `${num(e.n_cm3, 2)} /cm³ · ${int(e.v_kms)} km/s`) + hrow('env_mp', 'magnetopause', `${num(e.r0_re, 1)} R_E`);
    }
    if (V.done && V.final && t >= V.rows[V.rows.length - 1][R_T] - 1e-9)
      h += `<div class="${V.final.success ? 'pass' : 'fail'}">END: ${esc(V.final.reason)}${V.final.validation ? ` · stability ${V.final.validation.passed ? 'passed' : 'FAILED'}` : ''}</div>`;
  }
  return h;
}
function renderTransport(t) {
  setText('tLabel', V.mode === 'playback' ? `${fmtYear(t)} simulated` : V.mode === 'training' ? `${fmtYear(t)} simulated · replay of training episode ${((V.label || '').match(/episode (\d+)/) || [])[1] || ''}${TRAIN_EP ? ` (training is at episode ${TRAIN_EP})` : ''} · replay ${fmtSpeed(trainReplaySpeed())}/s` : 'Year 0 (start state)');
  setText('runLbl', V.mode === 'playback' ? `${V.label} · ${V.playing ? (V.waiting ? 'waiting for physics' : 'playing') : 'paused'} · wall ${((performance.now() - V.wallStart) / 1000).toFixed(0)} s`
    : V.mode === 'training' ? `${V.label} · ${V.playing ? 'playing' : 'PAUSED'} · ⟲ = jump to the newest episode · speed slider = replay speed` : 'choose a mission in ▶ Mission, or press ▶');
  const mp = $('marsPill'), safe = marsSafeAU();
  let txt = `Mars-safe ≥ ${num(safe, 4)} AU`, cls = 'pill';
  if (V.mode === 'playback' && MARS && Number.isFinite(MARS.d)) {
    const mn = Math.min(V.marsMin ?? Infinity, MARS.d);
    txt = `Mars: closest ${num(mn, 4)} AU ${mn >= safe ? '(safe)' : '(TOO CLOSE)'}`; cls = 'pill ' + (mn < safe ? 'bad' : MARS.d < marsCautAU() ? 'warn' : 'ok');
  }
  if (mp.textContent !== txt) mp.textContent = txt;
  if (mp.className !== cls) mp.className = cls;
  const cp = $('climatePill'), c = climNow(t), cl = V.meta && V.meta.climate;
  let ct = cl && !cl.on ? 'climate model off' : 'climate –', cc = 'pill';
  if (c && c.cl.on) { ct = `${num(c.T, 1)} K${c.mirRel > 1e-3 ? ` · ${num(c.mirRel, 2)}×` : ''}`; cc = 'pill ' + (c.ok ? 'ok' : Math.abs(c.T - c.cl.target) <= c.cl.livable ? 'warn' : 'bad'); }
  if (cp.textContent !== ct) cp.textContent = ct;
  if (cp.className !== cc) cp.className = cc;
}
function renderHud(t, st, el, i) {
  renderTransport(t);
  if (!hudMore) { setHtml('hud', compactHud(t, st, el, i)); return; }
  const wall = (performance.now() - V.wallStart) / 1000;
  let h = '';
  const modeTxt = V.mode === 'idle' ? 'Nothing running – choose a policy and press ▶ Play' :
    V.mode === 'training' ? `LIVE TRAINING · ${esc(V.label)} · replayed from its recorded physics (positions + velocities every 3/8 year); the newest finished episode follows` : `MISSION PLAYBACK · ${esc(V.label)}`;
  h += `<div>${modeTxt}</div>`;
  if (V.mode === 'playback') {
    const dayOfYear = Math.floor((t % 1) * 365.25);
    h += `<div title="Simulated time since the start of the mission. It is NOT wall-clock time."><span class="big">${fmtYear(t)}</span> <span class="dim">(day ${dayOfYear}) simulated</span></div>`;
    h += `<div class="dim" title="Real time since this playback started, and the time multiplier.">wall clock ${wall.toFixed(0)} s · speed ${fmtSpeed(speedY)}/s · ${V.playing ? 'playing' : 'PAUSED'}</div>`;
  }
  if (povOn()) h += povHud();
  if (V.mode === 'training' && V.final) {
    const f = V.final;
    h += `<div class="sec">Result of this exploring episode</div>` + hrow('a', 'final a', `${num(f.a_au)} AU`) + hrow('e', 'final e', num(f.e)) +
      hrow('m_ast', 'asteroid used', pct(f.mass_used_frac)) + hrow('E_used', 'energy used', pct(f.energy_used_frac)) +
      `<div class="row"><span>outcome</span><span>${esc(f.reason)} after ${int(f.t_years)} yr</span></div>`;
  }
  if (V.mode === 'playback' && st && el && povOn()) {      // compact summary so the HUD does not cover the close-up
    const q = V.rows[Math.min(i, V.rows.length - 1)], mr = moonAt(t);
    h += `<div class="sec">Mission (simulated state)</div>`;
    h += hrow('a', 'size a', `${num(el.a)} AU <span class="dim">(target ${V.meta.target_au})</span>`) + hrow('e', 'eccentricity e', num(el.e));
    h += hrow('speed', 'orbital speed', `${num(el.speed, 3)} km/s`) + hrow('packets', 'packets ejected', int(V.cumPk[i] ?? 0));
    h += hrow('m_ast', 'asteroid left', pct(q[R_MAST] / V.meta.m_ast0)) + hrow('moon', 'Earth–Moon distance', mr ? `${int(Math.hypot(...mr))} km` : 'no data');
    h += `<div class="sec">Mars</div>` + marsRows(t, true);
    h += planetRows(true);
    h += `<div class="dim small">Full orbit telemetry: switch View to the overview.</div>`;
  } else if (V.mode === 'playback' && st && el) {
    const q = V.rows[Math.min(i, V.rows.length - 1)], m0 = V.meta.m_ast0, E0 = V.meta.E0;
    const mast = q[R_MAST], eRem = q[R_EREM];
    const Memb = M_EARTH + M_MOON + mast;
    const epsSI = el.eps * (AU / YEAR_S) ** 2;
    h += `<div class="sec">Orbit (from the simulated state)</div>`;
    h += hrow('a', 'size a', `${num(el.a)} AU <span class="dim">(target ${V.meta.target_au} ± ${V.meta.tol_a_au})</span>`);
    h += hrow('e', 'eccentricity e', `${num(el.e)} <span class="dim">(≤ ${V.meta.tol_e})</span>`);
    h += hrow('inc', 'inclination', `${num(el.inc, 3)}°`);
    h += hrow('r', 'distance from Sun', `${num(el.r)} AU`);
    h += hrow('speed', 'orbital speed', `${num(el.speed, 3)} km/s`);
    h += hrow('E_orb', 'orbital energy', `${sci(epsSI * Memb, 4)} J`);
    h += `<div class="sec">Propulsion</div>`;
    const lastEj = V.lastEj >= 0 ? V.rows[V.lastEj] : null;
    h += hrow('packets', 'packets ejected', int(V.cumPk[i] ?? 0));
    h += hrow('recoil', 'last packet', lastEj ? `${sci(lastEj[R_DM])} kg @ ${int(lastEj[R_U] / 1000)} km/s` : '—');
    h += hrow('recoil', 'recoil per packet', lastEj ? `${num(lastEj[R_DV], 3)} m/s` : '—');
    const dvc = V.cumDv[i] ?? 0;
    h += hrow('dv', 'total Δv delivered', `${dvc < 1000 ? num(dvc, 1) : int(dvc)} m/s <span class="dim">(need ≈ 11,700)</span>`);
    const da = decActive(t);
    h += hrow('thrust', 'thrust (interval avg.)', da ? `${sci(da.thrust_N)} N ${da.dm_kg > 0 ? dirWords(da.d_rtn) : '(coasting)'}` : '—');
    h += `<div class="sec">Resources</div>`;
    h += hrow('m_ast', 'asteroid left', `${sci(mast, 3)} kg (${pct(mast / m0)})`, mast <= 0 ? 'warn' : '');
    h += hrow('E_used', 'energy used', `${sci(E0 - eRem, 3)} J (${pct((E0 - eRem) / E0)})`, eRem <= 0 ? 'warn' : '');
    h += `<div class="sec">Moon</div>`;
    {
      const mr = moonAt(t);
      h += hrow('moon', 'Earth–Moon distance', mr ? `${int(Math.hypot(...mr))} km` : 'no data at this time');
      const d = decAt(t);
      h += hrow('moon', 'Moon bound', d ? (d.moon_bound ? 'yes' : '<span class="warn">NO</span>') : 'yes (start)');
      h += `<div class="dim small">${V.fidelity === 'hifi' ? 'N-body: Sun, Earth(+asteroid), Moon and fragments all pull on each other.' : 'Fast model: exact Earth–Moon two-body orbit; the Sun\'s tide on the Moon is neglected.'}</div>`;
      if (d && d.conservation) {
        h += `<div class="sec">Numerical accuracy (end of last decision)</div>`;
        h += hrow('cons_E', 'energy error', sci(d.conservation.energy_residual_rel, 1));
        h += hrow('cons_P', 'momentum error', sci(d.conservation.momentum_residual_rel, 1));
        h += hrow('cons_M', 'mass error', sci(d.conservation.mass_residual_rel, 1));
      }
    }
    {
      const c = climNow(t);
      if (c && c.cl.on) {
        h += `<div class="sec">Climate (two-box model; agent aims the space mirrors)</div>` + climRows(t);
        if (c.dec) h += hrow('waste_heat_earth', 'waste heat reaching Earth', `${sci(c.dec.waste_heat_earth_W)} W`) +
          hrow('insol_set', "agent's light set-point", `${num(c.dec.insol_set_rel, 3)}× today`) +
          hrow('years_outside', 'years outside livable band', num(c.dec.years_outside_livable, 0));
        h += `<div class="dim small">Details: ▦ Ledger panel (rail on the left).</div>`;
      }
    }
    h += `<div class="sec">Mars &amp; planetary safety</div>` + marsRows(t, true) + planetRows(true);
    const d = decAt(t);
    h += `<div class="sec">Training score</div>` + hrow('return', 'cumulative reward', d ? num(d.cum_reward, 2) + ' <span class="dim">(not a physical quantity)</span>' : '—');
    if (V.done && V.final && t >= V.rows[V.rows.length - 1][R_T] - 1e-9) {
      h += `<div class="${V.final.success ? 'pass' : 'fail'}">END: ${esc(V.final.reason)}${V.final.validation ? ` · stability check ${V.final.validation.passed ? 'passed' : 'FAILED'}` : ''}</div>`;
    }
  }
  setHtml('hud', h);
}
function povHud() {
  const P = pov.P, e = P.env; if (!e) return '<div class="sec">Earth POV</div><div class="dim">waiting for a simulated state…</div>';
  let h = `<div class="sec">Environment at Earth (standard models, estimated)</div>`;
  h += hrow('env_r', 'distance from Sun', `${num(P.rAU)} AU`);
  h += hrow('env_sw', 'solar wind', `${num(e.n_cm3, 2)} protons/cm³ · ${int(e.v_kms)} km/s`);
  h += hrow('env_dp', 'solar-wind pressure', `${num(e.dp_npa, 3)} nPa`);
  h += hrow('env_imf', 'interplanetary field', `${num(e.imf_nt, 2)} nT <span class="dim">(spiral ${num(e.spiral_deg, 0)}°)</span>`);
  h += hrow('env_mp', 'magnetopause (nose)', `${num(e.r0_re, 1)} R_E = ${int(e.r0_re * 6371)} km`);
  h += hrow('env_bs', 'bow shock (nose)', `${num(e.rbs_re, 1)} R_E`);
  h += hrow('env_bfield', "Earth's field", `31,200 nT at surface · ${num(31200 / e.r0_re ** 3, 1)} nT at nose`);
  h += hrow('env_hill', 'Hill sphere radius', `${int(e.hill_re)} R_E = ${num(e.hill_re * 6371 / 1e6, 2)} million km`);
  h += hrow('env_gsun', 'Sun pull on Earth', `${sci(P.aSun, 3)} m/s²`);
  h += hrow('env_gmoon', 'Moon pull on Earth', P.moonPos ? `${sci(P.aMoon, 3)} m/s²` : 'N/A');
  h += hrow('env_insol', 'sunlight here', `${int(e.insolation_wm2)} W/m² <span class="dim">(${num(e.insolation_wm2 / 1361, 3)}× today)</span>`);
  h += `<div class="dim small">Display-only models evaluated at the simulated Sun distance; they do not change the simulated orbit. Hover a row for details.</div>`;
  return h;
}
function renderPower(i) {
  const box = $('power'), want = UI.power !== false && (beamOn() || colOn());
  box.hidden = !want; if (!want) return;
  const nt = $('notes'); box.style.top = (nt.style.display === 'none' || nt.hidden ? 8 : nt.offsetTop + nt.offsetHeight + 6) + 'px';
  const sol = SOL();
  if (!sol || !sol.on) { setHtml('powerBody', '<div class="dim">Solar collectors are switched off for these settings (Parameters → Solar collectors). Launch energy comes only from the starting reserve.</div>'); return; }
  const pw = CURPOW, L = sol.L_sun_W, nA = sol.n_arms || 8;
  const pin = pw.inW || 0, pel = pin * sol.rx, plaunch = pw.launch || 0, pke = plaunch * sol.launch_eff;
  const pmwArm = sol.P_mw_W * (pw.eta || 0);       // microwaves that reach the dishes when all collectors beam
  const lg = (v) => v > 0 ? Math.max(2, Math.min(100, 100 * (Math.log10(v) - 21) / (Math.log10(L) - 21))) : 0;
  const row = (lbl, v, col, tipTxt, sub = '') => `<div class="prow" title="${esc(tipTxt)}"><span class="pl">${lbl}</span><span class="pb"><i style="width:${lg(v)}%;background:${col}"></i></span><span class="pv">${v > 0 ? sci(v) + ' W' : '0'}</span></div>` + (sub ? `<div class="psub">${sub}</div>` : '');
  let h = '';
  if (colOn()) {
    const n = sol.count, one = (v) => sci(v / n);
    h += `<div class="pone" title="The belt's numbers divided by the number of collectors. In the model each collector is a flat thin-film sheet; the satellite shape is an illustration."><b>This collector</b> (1 of ${(n / 1e6).toFixed(0)} million)<br>`
      + `size ~${int(sol.each_side_km)} km · mass ~${sci(sol.mass_kg / n, 1)} kg<br>catches ${one(sol.L_captured_W)} W of sunlight → ${one(sol.P_pv_W)} W electric → ${one(sol.P_mw_W)} W microwaves<br>`
      + `${pw.inW > 0 ? `<span style="color:#d8c0ff">beaming now: ~${sci(pw.inW / Math.max(1e-30, pw.eta || 1) / n)} W of microwaves (the launcher needs ${pct(Math.min(1, pw.inW / Math.max(1, sol.P_mw_W * (pw.eta || 1))), 0)} of the belt)</span>` : '<span class="dim">stand-by: no launch right now, pilot beam only</span>'}</div>`;
  }
  h += `<div class="dim small">Where the launcher's power comes from <b>now</b> (event at ${fmtYear(V.rows[i] ? V.rows[i][R_T] : 0)}). Bars: log scale.</div>`;
  h += row('Sun', L, '#ffcc33', 'Total power radiated by the Sun (luminosity).');
  h += row(`belt ${pct(sol.fraction, 0)}`, sol.L_captured_W, '#ffd27a', 'Sunlight intercepted by the collector belt.', `${(sol.count / 1e6).toFixed(0)} M collectors at ${sol.orbit_au} AU, ~${int(sol.each_side_km)} km each · ${sci(sol.mass_kg, 1)} kg`);
  h += row(`→ solar cells ${pct(sol.pv, 0)}`, sol.P_pv_W, '#e6c060', 'Electric power from the collectors\' solar cells.');
  h += row(`→ microwaves ${pct(sol.mw, 0)}`, sol.P_mw_W, '#b48cff', `Microwave power transmitted (${sol.freq_ghz} GHz, λ = ${num(100 * sol.wavelength_m, 1)} cm).`);
  h += row(`→ hits dishes ${pct(pw.eta, 1)}`, pmwArm, '#a07cff', `Fraction caught by the ${nA} rectennas (Ø ${int(sol.dr_km)} km) – Goubau beam formula; falls as Earth moves away.`);
  h += row(`→ electricity ${pct(sol.rx, 0)}`, pw.avail, '#8fd0ff', 'Electric power AVAILABLE at the arms if every collector beams (it is only beamed when needed).');
  h += row('absorbed now', pin, '#d8c0ff', 'Microwave power the 8 dishes absorb in the current event (only what the launcher needs).', pin > 0 ? `${sci(pin / nA)} W per arm · ${sci(pin / nA / sol.rectenna_area_m2, 1)} W/m² = ${sci(pin / nA / sol.rectenna_area_m2 / 1361, 1)}× sunlight on each dish` : 'idle: no launch in this event (beams on stand-by)');
  h += row('launcher', plaunch, '#7fe08a', 'Electric power drawn by the launcher (beams first, reserve only for any shortfall).', plaunch > 0 ? `${pct(Math.min(1, pel / plaunch), 0)} from beams · ${pct(Math.max(0, 1 - pel / plaunch), 0)} from the reserve` : '');
  h += row('→ packet motion', pke, '#3ddc84', `Kinetic energy given to the packets (launch efficiency ${pct(sol.launch_eff, 0)}).`);
  const heat = (pin - pel) + (plaunch - pke);
  h += `<div class="psub warn" title="Rectenna + launcher losses end up as heat in the arms. Not simulated.">heat in the arms: ${sci(heat)} W = ${sci(heat / sol.earth_sunlight_W, 1)}× all sunlight on Earth – far beyond any real material (estimate, not simulated)</div>`;
  h += `<div class="psub dim">Microwave push on Earth not applied (tiny: ≈ P/c). Energy is conserved: launcher energy = reserve spent + beam electricity.</div>`;
  setHtml('powerBody', h);
}
function renderNotes(st) {
  const n = [];
  if (colOn()) {
    const sol = SOL();
    if (!sol || !sol.on) n.push('Solar collectors are switched off for these settings (Parameters → Solar collectors): nothing to show in this view.');
    else {
      n.push(`ILLUSTRATION: in the model each collector is a flat thin-film sheet ~${int(sol.each_side_km)} km across (${(sol.count / 1e6).toFixed(0)} million of them, ${pct(sol.fraction, 0)} of the Sun's output); here it is drawn as a satellite, not to scale.`);
      n.push('From the simulation: the directions to the Sun and to Earth (Earth\'s current position, collector 50° ahead of Earth in the belt) and the beam brightness (power recorded for the current launch; dim = stand-by).');
      n.push('Neighbouring collectors are drawn far apart; the real ones nearly touch and form a continuous band. Microwaves are invisible – colour and wavefronts are illustrations. Thruster puffs and nav lights are decoration.');
    }
    if (V.mode === 'idle') n.push('Showing the simulated start state (year 0). Press ▶ to see the beam power during launches.');
    showNotes(n); return;
  }
  if (povOn() && pov.P.env) {
    const P = pov.P;
    n.push('Earth POV: true scale, 1 unit = 1 Earth radius (6,371 km). Drag to rotate, scroll to zoom.');
    if (P.layers.arms) n.push(`Ring and arms are an ILLUSTRATION – no hardware is simulated. For each recorded ejection ALL 8 arms grab the same single packet and throw it together in its true direction (one packet per throw). At high playback speed several recorded launches share one animated throw; the animation adds ≈0.7 s of wall-clock delay. The rock is drawn enlarged (a real 1e18 kg packet is ~100 km across). Geometry (update p36): every arm is ${ARM_L} Earth radii long; the asteroid sits on the ring\'s axis, its Earth-facing side on the grab point ${GRAB_Z} R_E from Earth\'s centre, so all 8 arm roots (ring radius ${LRING_R} R_E) are the same ${Math.hypot(GRAB_Z, LRING_R).toFixed(2)} R_E from it; the joint throw carries the packet ~2,000 km before release.`);
    if (P.layers.packets) n.push('Packet streaks: recorded direction and fate; motion slowed down for visibility.');
    if (P.rotFrozen) n.push("Earth's daily rotation is not animated above 2 days/s (it would be a blur).");
    if (P.astEnlarged) n.push('Asteroid drawn enlarged (its true size for the remaining mass would be below one pixel).');
    if (P.layers.wind) n.push('Solar-wind particles move at a slowed visual speed (real: 400 km/s); their number ∝ density.');
    if (P.windHidden) n.push('Solar-wind particles hidden while zoomed out beyond 250 R_E (they are drawn only within ~100 R_E of Earth).');
    if (P.layers.field || P.layers.magnetopause || P.layers.belts) n.push('Field, boundaries and belts: standard simplified models (dipole, Shue 1998, schematic belts) – not simulated dynamics.');
    if (P.layers.gravwell) n.push('Gravity well: effective potential of Earth + Moon + Sun tide (Hill approximation), height log-scaled.');
    if (P.power && (P.layers.beams || P.layers.collectors)) n.push('Power beams: brightness follows the beam power recorded by the physics for the current event; beam width and dishes are TRUE scale. Microwaves are invisible – colour, wavefronts and arm pulses are illustrations. The beams\' push on Earth is NOT simulated (estimate in the Power panel); the share of the arms\' heat that reaches Earth is part of the climate model.');
    if (P.mirrors && P.mirrors.on) n.push('Space mirrors: ring at its TRUE distance (20 Earth radii) in the plane perpendicular to the sunlight; mirrors drawn ~100× enlarged (48 drawn for ~1 million). The share aimed at Earth and the light follow the mirror light recorded by the climate model (the agent\'s action a4).');
    if (P.beamEnlarged && P.layers.beams) n.push('Beams drawn wider than true scale while zoomed out (kept ≥ 3 px wide); zoom in for the true 1,000 km width.');
    if (P.beamStandby && P.layers.beams) n.push('Beams dim = stand-by pilot beam: no launch in this event, no power flowing (press ▶ to watch launches).');
    if (P.power && P.layers.beams) n.push('The collector satellites at the far end of the beams are drawn ~15× enlarged (press C for a close-up of one).');
    if (P.power && P.layers.collectors) n.push(`Collector belt: true directions from Earth, distances scaled like the Sun's; each dot stands for many collectors (each really ~${int(P.solar.each_side_km)} km wide).`);
    if (P.layers.mars && P.marsPos) n.push('Mars: true direction; drawn at its true distance up to 40,000 R_E (label = true distance). Safety spheres at true scale (zoom out).');
    if (V.mode === 'training') n.push('Training view: replay of the newest finished training episode (fast model, with exploration noise) from positions + velocities recorded every 3/8 year; Earth moves between them by the same interpolation as a playback. The Moon is drawn at its final offset; packets, beams, Mars, Mercury, Venus and Jupiter are not shown.');
    if (V.mode === 'idle') n.push('Showing the simulated start state (year 0).');
    if (st && st.gap) n.push('Data gap: Earth is held at the last recorded state (no interpolation across a gap).');
    if (V.waiting) n.push('Waiting for the physics engine…');
    showNotes(n);
    return;
  }
  if (V.mode === 'playback') {
    if (V.fidelity === 'fast') n.push('Fast model (what the agent trained on): Moon simulated as an exact two-body orbit around Earth; the Sun\'s tidal pull on the Moon is neglected.');
    if (st && st.gap) n.push('Data gap: Earth is held at the last recorded state (no interpolation across a gap).');
    if (V.waiting) n.push('Waiting for the physics engine…');
    if (pkNote) n.push(pkNote);
    n.push(`Not to scale: Earth, Sun and asteroid markers enlarged; Moon distance ×100; packet paths shifted ${AST_OFFSET_AU.toFixed(3)} AU north to start at the grab point under the asteroid marker (on the launch ring's axis).`);
    n.push('Packet paths: two-body solar gravity, drawn for ≤ 1 year or 60 AU.');
  }
  if (V.mode !== 'training' && SOL() && SOL().on && OV.collectors) n.push(`Collector belt: ~900 satellites drawn for ${(SOL().count / 1e6).toFixed(0)} million (each really ~${int(SOL().each_side_km)} km wide, drawn enlarged); they orbit the Sun every ~${int(365.25 * (SOL().orbit_period_yr || 0.354))} days${OVX_STATE.frozen ? ' – motion frozen at this playback speed (slow down to see it)' : ''}.`);
  if (V.mode === 'playback' && OV.tracked && V.meta && V.meta.belt) n.push('Solid silver collectors with a thin orbit circle: 12 SIMULATED collectors with station keeping (they stay within a few hundred km of their slots – too small to see here). Faint grey ghosts: the same collectors simulated WITHOUT station keeping, at their true positions (they slide along the orbit out of their slots). Numbers: ▦ Ledger panel.');
  if (V.mode !== 'training' && OV.arms) n.push('Earth\'s launcher ring and 8 arms: drawn to the same (enlarged) scale as the Earth marker; while packets are launched they throw together along the recorded launch direction (one throw stands for many packets).');
  if (V.mode === 'playback' && OV.mirrors && climNow(V.disp) && climNow(V.disp).cl.mirrors && climNow(V.disp).cl.mirrors.on) n.push('Space mirrors (white squares around Earth): drawn at 4.5 Earth-marker radii (real orbit: 20 Earth radii); the share whose rays hit Earth follows the mirror light recorded by the climate model. Close-up: Weather view (key W).');
  if (V.mode !== 'training' && CURPOW.on && (OV.collectors || OV.beams)) n.push('Solar collectors (gold dots, 0.5 AU) and power beams (violet, one line per arm) are drawn schematically; their power is simulated as the launcher\'s energy supply only.');
  if (V.mode !== 'training' && MARS) n.push('Mars: simulated in both models (pulls Earth and the Moon, is pulled back). Marker enlarged; the red / yellow rings around Mars (shown within 0.3 AU) are the TRUE-scale safe and caution distances.');
  if (V.mode !== 'training' && PLANETS) n.push('Mercury, Venus, Jupiter (update p35): simulated in both models (they pull Earth, the Moon, Mars and the fragments and are pulled back). Drawn exactly at the recorded state of each ejection event and moved between events along their two-body orbit from that state. Markers enlarged. Saturn and beyond are not modelled. Use "whole system (to Jupiter)" in Mission > Overview to zoom out.');
  if (V.mode === 'training') n.push('Training view: positions only (Mars, Mercury, Venus, Jupiter not drawn).');
  showNotes(n);
}
let ledgerWall = 0;
function renderLedger() {             // rail drawer "▦ Ledger": climate, energy ledger, belt, enablers, problems (2 updates per second)
  if (UI.drawer !== 'dLedger') return;
  const now = performance.now(); if (now - ledgerWall < 500) return; ledgerWall = now;
  const t = V.mode === 'playback' ? V.disp : 0, dec = V.mode === 'playback' ? decFor(t) || (V.decisions.length ? V.decisions[V.decisions.length - 1] : null) : null;
  const cum = {};
  if (dec) for (const d of V.decisions) { if (d.t_years > dec.t_years + 1e-9) break; for (const [k, v] of Object.entries(d.ledger || {})) if (typeof v === 'number') cum[k] = (cum[k] || 0) + v; }
  const box = $('ledgerBody'), st = box.scrollTop;
  setHtml('ledgerBody', ledgerPanelHtml({ dec, t, cum, meta: V.meta, enablers: DOCS ? DOCS.enablers : [], problems: DOCS ? DOCS.problems : [], frozen: OVX_STATE.frozen }));
  box.scrollTop = st;
  if (dec) drawClimateChart($('chClimate'), V.decisions, V.meta, t);
}
function showNotes(n) {
  setHtml('notesBody', n.map((x) => `<div>${esc(x)}</div>`).join(''));
  setText('notesCount', n.length ? `${n.length} note${n.length > 1 ? 's' : ''} on approximations ${$('notes').classList.contains('collapsed') ? '▸' : '▾'}` : '');
  $('notes').style.display = n.length && UI.notes ? 'block' : 'none';
}
const _v = new THREE.Vector3();
function placeDvLabel(text) {
  const el = $('dvLabel');
  if (!text) { el.style.display = 'none'; return; }
  _v.copy(recoil.position).add(new THREE.Vector3().copy(recoil.userData.dir).multiplyScalar(1.5)).project(camera);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (_v.z > 1) { el.style.display = 'none'; return; }
  el.style.display = 'block'; el.style.left = `${(_v.x * 0.5 + 0.5) * w + 6}px`; el.style.top = `${(-_v.y * 0.5 + 0.5) * h - 8}px`; el.textContent = text;
}

// ------------------------------------------------------------------ live-training replay helpers
const TRAIN_LOOP_S = 4;             // at the end: wait this long for a newer episode, then replay again
const TRAIN_WANT_S = 0.8;           // at the end: ask the server for the newest finished episode after this pause
const TRAIN_REPLAY_S = 6;           // "keep up" pace: one training episode is replayed in about this many seconds
/** Replay speed [simulated years per wall-clock second]. "keep up" (default): the whole episode in ~6 s, so the screen
 *  keeps pace with training (it finishes an episode every ~1 s). "slow detail": the speed slider. Training itself
 *  runs in its own process and is never slowed down by the replay. */
function trainReplaySpeed() {
  const R = V.rows;
  if (($('trainPace') || {}).value === 'slider' || !R.length) return speedY;
  return Math.max(speedY, (R[R.length - 1][0] - R[0][0]) / TRAIN_REPLAY_S);
}
/** Earth on a training replay at time t: Hermite between the two recorded states around t (never across gaps). */
function trainEarthAt(t) {
  const R = V.rows, i = Math.max(0, bsearch(R, t));
  const a = R[i], P = (q) => [q[1], q[2], q[3]], W = (q) => [q[4] * KMS2AUY, q[5] * KMS2AUY, q[6] * KMS2AUY];
  if (i >= R.length - 1 || a.length < 7) return { p: P(a), v: a.length >= 7 ? W(a) : [0, 0, 0], i };
  const b = R[i + 1], h = b[0] - a[0];
  if (!(h > 0) || h > 1.0) return { p: P(a), v: W(a), i };                    // gap: hold, do not interpolate
  const o = polarHermite(P(a), W(a), P(b), W(b), h, Math.max(0, Math.min(1, (t - a[0]) / h)));
  return { p: o.p, v: o.v, i };
}
/** The decision record of a training replay that covers time t (asteroid mass left etc.). */
function trainDecAt(t) {
  const D = V.decisions; let k = -1;
  for (let j = 0; j < D.length; j++) { if (D[j].t_years - (D[j].years || 0) <= t) k = j; else break; }
  return k >= 0 ? D[k] : null;
}

// ------------------------------------------------------------------ frame loop
let lastFrame = performance.now();
let MARS = null;                    // current Mars state for HUD / POV: {p, v, d, el, shield}
let frameErrors = 0, FRAME_MS = 0;
// measured frames per second (time between frames, including the graphics card's drawing) + automatic resolution drop
let FPS = 60, fpsLowSince = 0, fpsWarned = false;
function trackFps(now, gapMs) {
  if (gapMs > 0 && gapMs < 5000) FPS = 0.9 * FPS + 0.1 * (1000 / gapMs);
  if (FPS < 20 && !document.hidden) {
    if (!fpsLowSince) fpsLowSince = now;
    if (now - fpsLowSince > 4000 && PIXEL_RATIO > 0.75) {   // 4 s of low frame rate: draw with fewer pixels
      PIXEL_RATIO = PIXEL_RATIO > 1 ? 1 : 0.75; renderer.setPixelRatio(PIXEL_RATIO); fpsLowSince = now;
      toast(`low frame rate (${FPS.toFixed(0)} fps): drawing resolution reduced to ${PIXEL_RATIO}x to keep the viewer responsive`);
    }
    if (!fpsWarned && now - fpsLowSince > 3000 && GL_GPU.software) {
      fpsWarned = true;
      toast(`the browser draws WITHOUT the graphics card (software renderer: ${GL_GPU.name}). Turn on "Use graphics acceleration when available" in the browser settings and restart the browser.`);
    }
  } else fpsLowSince = 0;
}
function frame() {
  // the next frame is requested FIRST, so an error in one frame can never stop the animation
  requestAnimationFrame(frame);
  const t0 = performance.now();
  try { drawFrame(); FRAME_MS = 0.9 * FRAME_MS + 0.1 * (performance.now() - t0); } catch (e) {
    if (frameErrors++ < 5) { console.error(e); toast(`display error (the animation continues): ${e.message}`); }
  }
}
function drawFrame() {
  // dt is capped at 0.25 s (not 0.1 s) so the chosen playback speed is kept down to 4 frames per second
  const now = performance.now(); trackFps(now, now - lastFrame); const dt = Math.min(0.25, (now - lastFrame) / 1000); lastFrame = now;
  resize();
  trailLine.visible = V.mode !== 'training'; trailDots.visible = V.mode === 'training';
  let st = null, el = null, i = 0, POVS = null;
  if (V.mode === 'training') { MARS = null; CURPOW = powerAt(null); }
  if (V.mode === 'training' && V.rows.length) {
    // LIVE TRAINING: replay the newest finished training episode. Rows = [t, x, y, z (AU), vx, vy, vz (km/s)],
    // recorded by the physics every 3/8 year; Earth moves between them with the same Hermite interpolation as a
    // mission playback. The view keeps moving even while PPO updates its network (no new episodes for a while).
    const R = V.rows, t0 = R[0][0], tEnd = R[R.length - 1][0];
    if (V.trailUpTo !== R.length) {
      V.trailN = 0; R.forEach((q, k) => { if (V.trailN < MAXTRAIL) setTrailVertex(V.trailN++, [q[1], q[2], q[3]], 10000 * k / R.length); });
      trailGeom.attributes.position.needsUpdate = true; trailGeom.attributes.color.needsUpdate = true; V.trailUpTo = R.length;
      V.disp = t0; V.trHold = 0;
    }
    const rate = trainReplaySpeed();
    if (V.playing) V.disp = Math.min(V.disp + rate * dt, tEnd);
    if (V.disp >= tEnd - 1e-9) V.trHold = (V.trHold || 0) + dt; else V.trHold = 0;
    if (V.trHold > TRAIN_LOOP_S) { V.disp = t0; V.trHold = 0; }            // no newer episode yet: replay this one again
    const ts = trainEarthAt(V.disp);
    trailGeom.setDrawRange(0, Math.min(V.trailN, ts.i + 1));                 // the path grows behind Earth
    earth.position.set(ts.p[0] * S, ts.p[1] * S, ts.p[2] * S);
    const f = V.final || {};
    placeMoon(earth.position, f.moon_rel_km || null); moonOrbGeom.setDrawRange(0, 0);
    const vm = Math.hypot(...ts.v) || 1, back = ts.v.map((x) => -x / vm);
    const dk = trainDecAt(V.disp), m0 = f.m_ast0 || (V.init ? V.init.m_ast0 : 0);
    const left = dk ? Math.max(0, dk.m_ast_frac) : 1;
    placeAsteroid(earth.position, back, left, V.disp);
    POVS = { t: V.disp, p: ts.p, v: ts.v, v_kms: vm / KMS2AUY, moon: f.moon_rel_km || null, moonSt: null, mAst: m0 * left, mAst0: m0,
      mSys: M_EARTH + M_MOON + m0 * left, ehat: back, events: [], recentPush: false, playing: V.playing, speedY: rate };
    recoil.visible = false; placeDvLabel(''); hideMars(); hidePlanets();
  } else if (V.mode === 'playback' && V.rows.length && V.meta) {
    const tEnd = V.rows[V.rows.length - 1][R_T];
    if (V.disp < V.rows[0][R_T]) V.disp = V.rows[0][R_T];
    if (V.playing) V.disp = Math.min(V.disp + speedY * dt, tEnd);
    V.waiting = V.playing && !V.done && V.disp >= tEnd - 1e-12;
    const t = V.disp;
    st = earthAt(t); i = st.i;
    // latest ejection at or before t
    for (let k = V.ejScan + 1; k <= i; k++) if (V.rows[k][R_DM] > 0) V.lastEj = k;
    V.ejScan = Math.max(V.ejScan, i);
    const p = st.p; earth.position.set(p[0] * S, p[1] * S, p[2] * S);
    el = elements(p, st.v);
    updateTrail(i, p);
    if (OV.ahead) updateAhead(i); else aheadGeom.setDrawRange(0, 0);
    if (OV.coast) updateCoast(p, st.v, el); else coastGeom.setDrawRange(0, 0);
    const ms = marsAt(t), r0 = V.rows[0];
    MARS = ms ? { ...ms, ...placeMars(ms, p, r0.length > R_SHIELD && r0[R_T] === 0 ? { p: rowM(r0), v: rowMV(r0) } : null), shield: V.rows[i][R_SHIELD] } : null;
    if (!ms) hideMars();
    placePlanets(PLAN.statesFromRows(V.rows, bsearch(V.rows, t), t), p);
    // asteroid: size ~ cbrt(remaining mass)
    const q = V.rows[i], frac = Math.max(0, q[R_MAST] / V.meta.m_ast0);
    CURPOW = powerAt(q);
    if (!(CURPOW.inW > 0) && V.lastEj >= 0 && t - V.rows[V.lastEj][R_T] <= Math.max(1.5 * eventSpacing(), 0.6 * speedY)) CURPOW = powerAt(V.rows[V.lastEj]);
    placeAsteroid(earth.position, astOffsetDir(i), frac, t);
    // Moon (simulated in both models): position from its own recorded stream + its current orbit
    placeMoon(earth.position, moonAt(t));
    drawMoonOrbit(earth.position, moonStateAt(t), q[R_MAST]);
    trimMoon(t);
    // recoil arrow: direction of v_post - v_pre of the latest ejection, shown while ejections are recent
    const lj = V.lastEj >= 0 ? V.rows[V.lastEj] : null;
    const recent = lj && (t - lj[R_T]) <= Math.max(1.5 * eventSpacing(), PERSIST_WALL_S * speedY);
    if (recent && lj[R_DV] > 0) {
      const d = [lj[R_VPOST] - lj[R_VPRE], lj[R_VPOST + 1] - lj[R_VPRE + 1], lj[R_VPOST + 2] - lj[R_VPRE + 2]], m = Math.hypot(...d) || 1;
      const dir = new THREE.Vector3(d[0] / m, d[1] / m, d[2] / m);
      recoil.position.copy(earth.position); recoil.setDirection(dir); recoil.userData.dir = dir; recoil.visible = true;
      placeDvLabel(`recoil ${num(lj[R_DV], 3)} m/s per packet · total Δv ${V.cumDv[i] < 1000 ? num(V.cumDv[i], 1) : int(V.cumDv[i])} m/s`);
    } else { recoil.visible = false; placeDvLabel(''); }
    if (OV.packets) updatePackets(t); else { pkGeom.setDrawRange(0, 0); headGeom.setDrawRange(0, 0); }
    processEvents(t, i);
    {   // POV: new ejection events since the last frame (direction, fate, speed) for the throw animation
      const events = [];
      for (let k = Math.max(V.povScan + 1, i - 400); k <= i; k++) { const r = V.rows[k]; if (r[R_DM] > 0) events.push({ ehat: [r[R_EHAT], r[R_EHAT + 1], r[R_EHAT + 2]], fate: r[R_FATE], u: r[R_U] }); }
      V.povScan = Math.max(V.povScan, i);
      OVTHROW.launching = !!(recent && lj[R_DV] > 0) && V.playing && !V.waiting; OVTHROW.dir = astOffsetDir(i);
      POVS = { t, p: st.p, v: st.v, v_kms: el.speed, moon: moonAt(t), moonSt: moonStateAt(t), mAst: q[R_MAST], mAst0: V.meta.m_ast0,
        mSys: M_EARTH + M_MOON + q[R_MAST], ehat: astOffsetDir(i), ehatRtn: astRtn(i), events: events.slice(-8), recentPush: !!(recent && lj[R_DV] > 0), playing: V.playing && !V.waiting, speedY,
        mars: ms ? ms.p.map((x, c) => x - st.p[c]) : null, marsSafe: marsSafeAU(), marsCaution: marsCautAU(), power: CURPOW, solar: SOL(),
        climate: climNow(t) };
    }
  } else {
    // nothing running: show the simulated START state (year 0) of the mission, if known
    const I = V.init;
    CURPOW = powerAt(null);
    if (I) earth.position.set(I.earth[0] * S, I.earth[1] * S, I.earth[2] * S); else earth.position.set(S, 0, 0);
    placeMoon(earth.position, I ? I.moon_rel_km : null);
    drawMoonOrbit(earth.position, I ? { r: I.moon_rel_km, v: I.moon_v_kms } : null, I ? I.m_ast0 : 0);
    placeAsteroid(earth.position, I ? I.ast_dir : [-1, 0, 0], 1, 0);
    recoil.visible = false; placeDvLabel(''); OVTHROW.launching = false;
    if (I && I.mars) { const m0 = { p: I.mars, v: I.mars_v_kms.map((x) => x * KMS2AUY) }; MARS = { ...m0, ...placeMars(m0, I.earth, m0), shield: 0 }; } else { hideMars(); MARS = null; }
    PLANETS = null; placePlanets(PLAN.statesFromInit(I), I ? I.earth : null);
    if (I && I.earth_v_kms) {
      const vk = I.earth_v_kms;
      POVS = { t: 0, p: I.earth, v: vk.map((x) => x * KMS2AUY), v_kms: Math.hypot(...vk), moon: I.moon_rel_km, moonSt: { r: I.moon_rel_km, v: I.moon_v_kms },
        mAst: I.m_ast0, mAst0: I.m_ast0, mSys: M_EARTH + M_MOON + I.m_ast0, ehat: I.ast_dir, events: [], recentPush: false, playing: false, speedY,
        mars: I.mars ? I.mars.map((x, c) => x - I.earth[c]) : null, marsSafe: marsSafeAU(), marsCaution: marsCautAU(), power: CURPOW, solar: SOL() };
    }
  }
  if (V.mode !== 'playback' && V.mode !== 'idle') moonOrbGeom.setDrawRange(0, 0);
  updateOvSolar(dt, earth.position);
  {
    const tt = V.mode === 'playback' ? V.disp : 0, cn = climNow(tt);
    OVX_STATE = OVX.update({ t: tt, speedY, playing: V.mode === 'playback' && V.playing && !V.waiting, sol: SOL(),
      belt: V.meta && V.meta.belt, dec: V.mode === 'playback' ? decFor(tt) : null, hide: closeUp(),
      layers: { collectors: OV.collectors, tracked: OV.tracked, mirrors: OV.mirrors, arms: OV.arms },
      mirRel: cn ? cn.mirRel : 0, mirrors: cn && cn.cl.mirrors ? { on: cn.cl.mirrors.on, capRel: cn.dec ? cn.dec.mirror_cap_rel : cn.cl.mirrors.cap_rel } : null,
      launching: OVTHROW.launching, throwDir: OVTHROW.dir,
      ast: asteroid.visible ? { dist: asteroid.position.distanceTo(earth.position), r: asteroid.scale.x } : null });
  }
  renderLedger();
  placeLabel('sun', sun.position, 'Sun', 12, -10);
  placeLabel('earth', earth.position, 'Earth', 8, 8, asteroid.visible ? asteroid.position : sun.position);
  resolveLabels();
  if ($('follow').checked) {        // keep the camera's offset to Earth constant
    const d = new THREE.Vector3().subVectors(earth.position, controls.target);
    controls.target.add(d); camera.position.add(d);
  }
  const usePov = povOn() && POVS, useCol = colOn() && SOL() && SOL().on, useWx = wxOn();
  if (useWx) {
    wx.update(weatherState(), dt, now / 1000);
    for (const el2 of Object.values(LBL)) el2.style.display = 'none';
    $('dvLabel').style.display = 'none'; pov.hideLabels(); col.hideLabels();
  }
  if (useCol) {
    col.update({ p: [earth.position.x / S, earth.position.y / S, earth.position.z / S], power: CURPOW, solar: SOL() }, dt, now / 1000);
    for (const el2 of Object.values(LBL)) el2.style.display = 'none';
    $('dvLabel').style.display = 'none'; pov.hideLabels();
  }
  if (usePov) {
    pov.update(POVS, dt, now / 1000);
    for (const el2 of Object.values(LBL)) el2.style.display = 'none';
    $('dvLabel').style.display = 'none';
  }
  renderHud(V.disp, st, el, i); renderNotes(st); renderPower(i);
  updateLegend();
  if (useWx) wx.render();
  else if (useCol) { wx.hideLabels(); col.render(); }
  else if (usePov) { col.hideLabels(); wx.hideLabels(); pov.render(); }
  else { pov.hideLabels(); col.hideLabels(); wx.hideLabels(); controls.update(); renderer.render(scene, camera); }
}
requestAnimationFrame(frame);

// ------------------------------------------------------------------ API
async function api(path, body) {
  const opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const j = await r.json();
  if (!r.ok || j.ok === false) throw new Error(j.error || r.statusText);
  return j;
}
// viewer messages are kept (newest first, last 8) and shown above the server log, which is redrawn every poll
const TOASTS = [];
function toast(msg) {
  TOASTS.unshift(`>> ${new Date().toTimeString().slice(0, 8)} ${msg}`); if (TOASTS.length > 8) TOASTS.length = 8;
  const l = $('log'); l.textContent = `${TOASTS[0]}\n` + l.textContent;
}

function ingest(v) {
  if (v.meta) { V.meta = v.meta; setTargetRing(v.meta.target_au); }
  if (v.trail) V.trailSrv = v.trail;
  if (v.dstart === V.decisions.length) for (const d of v.decisions) {
    const k = V.decisions.length; if (V.beltPend && V.beltPend[k]) { d.belt = V.beltPend[k]; delete V.beltPend[k]; }
    V.decisions.push(d);
  }
  // collector-belt results are computed in their own thread behind the physics and arrive separately
  if (v.belts && v.bstart === (V.bsince || 0)) {
    for (const [k, b] of v.belts) { if (V.decisions[k]) V.decisions[k].belt = b; else (V.beltPend = V.beltPend || {})[k] = b; }
    V.bsince = (V.bsince || 0) + v.belts.length;
  }
  if (v.rows && v.rows.length && V.mode !== 'training') {
    if (v.rstart !== V.since && V.rows.length) toast(`data window skipped ${v.rstart - V.since} rows (display was far behind)`);
    if (!V.rows.length && v.rstart > 0) {      // joined mid-run: start counters from the decision records
      const t0 = v.rows[0][R_T]; let pk = 0, dv = 0;
      for (const d of V.decisions) if (d.t_years <= t0) { pk += d.n_packets || 0; dv = d.dv_cum_ms || dv; }
      V.cumPk0 = pk; V.cumDv0 = dv;
    }
    for (const q of v.rows) {
      const k = V.rows.length;
      V.cumPk.push((k ? V.cumPk[k - 1] : (V.cumPk0 || 0)) + (q[R_DM] > 0 ? 1 : 0));
      V.cumDv.push((k ? V.cumDv[k - 1] : (V.cumDv0 || 0)) + (q[R_DV] || 0));
      V.rows.push(q);
    }
    V.since = v.rstart + v.rows.length;
  } else if (v.rows && v.rows.length && V.mode === 'training') {      // a training replay arrives in one piece
    if (v.rstart === 0) V.rows = v.rows; else for (const q of v.rows) V.rows.push(q);
    V.since = v.rstart + v.rows.length;
  }
  if (v.moon && v.moon.length) { for (const m of v.moon) V.moon.push(m); V.msince = v.mstart + v.moon.length; }
  if (v.pnext !== undefined && v.pnext > V.psince) { for (const p of v.packets) V.packets.push(p); V.psince = v.pnext; V.pstrideRx = v.pstride || 1; }
}

let lastEpisode = -1, lastCompare = '', resultsDirty = true;
async function pollState() {
  try {
    const pstride = speedY > 2 ? Math.ceil(speedY / 2) : 1;
    const q = `run=${V.run}&since=${V.since}&msince=${V.msince}&psince=${V.psince}&dsince=${V.decisions.length}&bsince=${V.bsince || 0}&pstride=${pstride}` +
      (V.mode === 'playback' ? `&disp=${V.disp.toFixed(4)}&ahead=${Math.min(1500, 4 * speedY).toFixed(2)}` : '') +
      ((V.mode === 'training' && ((V.trHold || 0) > TRAIN_WANT_S || V.trJump)) || V.mode === 'idle' ? '&trnext=1' : '');
    const st = await api('/api/state?' + q);
    $('conn').textContent = `connected · ${FPS.toFixed(0)} fps${GL_GPU.software ? ' · NO graphics card (software drawing)' : ''}`; $('conn').className = FPS < 15 ? 'pill warn' : 'pill ok';
    $('conn').title = `Frames per second drawn by the browser (graphics: ${GL_GPU.name}${GL_GPU.software ? ' = SOFTWARE rendering, the graphics card is not used by the browser' : ''}). JavaScript time per frame: ${FRAME_MS.toFixed(1)} ms. Below ~15 fps the view stutters; the resolution is lowered automatically.`;
    const v = st.viewer;
    if (v.run_id !== V.run || v.reset) { resetViewer(); V.run = v.run_id; V.trJump = false; }
    if (st.init) V.init = st.init;
    V.mode = v.mode; V.label = v.label; V.done = v.done; V.final = v.final; V.fidelity = v.fidelity; V.policy = v.policy; V.produced = v.produced_years;
    ingest(v);
    renderTrain(st);
    $('log').textContent = (TOASTS.length ? TOASTS.join('\n') + '\n' : '') + st.log.slice().reverse().join('\n');
    if (Math.floor(st.train.episode / 10) !== Math.floor(lastEpisode / 10) || st.compare_state !== lastCompare) resultsDirty = true;
    lastEpisode = st.train.episode; lastCompare = st.compare_state;
  } catch (e) {
    $('conn').textContent = 'disconnected'; $('conn').className = 'pill off';
  }
  setTimeout(pollState, V.mode === 'playback' || V.mode === 'training' ? 250 : 700);
}
function renderTrain(st) {
  const t = st.train;
  $('trainStatus').textContent = `${t.state} · episode ${t.episode}/${t.target} · ${t.timesteps} decisions · ${(t.wall_s || 0).toFixed(0)} s wall` +
    (t.eps_per_s ? ` · ${t.eps_per_s.toFixed(1)} episodes/s` : '') + (t.device ? ` · network on ${t.device === 'cuda' ? 'GPU' : 'CPU'}${t.gpu_mem_mb ? ` (${t.gpu_mem_mb} MB GPU memory)` : ''}` : '') +
    (st.gpu && st.gpu.fix ? `\nGPU NOT USED: ${st.gpu.fix}` : '') + (t.error ? `\nERROR: ${t.error}` : '') +
    (t.state === 'idle' && st.best ? '\n(showing saved results of a previous run)' : '');
  TRAIN_RUNNING = t.state === 'running'; TRAIN_EP = t.episode;
  if (typeof t.live === 'boolean' && Date.now() - LIVE_SET_AT > 3000) { TRAIN_LIVE = t.live; $('trainLive').checked = t.live; }
  if (t.device_request && Date.now() - DEV_SET_AT > 3000) $('trainDevice').value = t.device_request;
  markShow();
  const bn = t.bench || {};
  $('trainSpeed').textContent = (t.s_per_ep ? `speed: ${t.s_per_ep < 10 ? t.s_per_ep.toFixed(2) : int(t.s_per_ep)} s per episode` +
      (Number.isFinite(t.phys_frac) ? ` (${pct(t.phys_frac, 0)} of it physics)` : '') : '') +
    (bn.cpu && bn.cuda ? ` · measured network time per PPO iteration: CPU ${bn.cpu.toFixed(2)} s, GPU ${bn.cuda.toFixed(2)} s` : '');
  $('trainBar').style.width = `${Math.min(100, 100 * t.episode / Math.max(1, t.target))}%`;
  $('btnTrain').disabled = t.state === 'running' || st.checks_ok === false;
  $('btnStop').disabled = t.state !== 'running';
  $('btnCompare').disabled = st.compare_state === 'running';
  $('btnCompare').textContent = st.compare_state === 'running' ? 'Comparing…' : 'Compare policies';
  const b = st.best;
  $('bestBox').innerHTML = b ? `<b>best_model.zip</b> = episode ${b.episode}: deterministic test success ${pct(b.eval.success_rate, 0)}, a = ${num(b.eval.a_au)} AU, ` +
    `e = ${num(b.eval.e)}, energy ${pct(b.eval.energy_frac)}, ${int(b.eval.years)} yr. <span class="dim">Best of the tested checkpoints – not a global optimum.</span>` : 'best model: N/A (not trained yet)';
  $('feasPill').textContent = st.feasible ? 'FEASIBLE (in this model)' : 'INFEASIBLE';
  $('feasPill').className = 'pill ' + (st.feasible ? 'ok' : 'bad'); $('feasPill').title = st.verdict + '\n(see Guide → Feasibility)';
  $('checksPill').textContent = st.checks_ok === null ? 'checks …' : st.checks_ok ? 'physics checks PASS' : 'physics checks FAIL';
  $('checksPill').className = 'pill ' + (st.checks_ok ? 'ok' : st.checks_ok === false ? 'bad' : '');
  const L = t.losses || {};
  $('losses').textContent = Object.keys(L).length ? Object.entries(L).map(([k, x]) => `${k} = ${(+x).toPrecision(3)}`).join(' · ') + `  (update ${t.ppo_iter})` : 'no PPO update yet';
}
async function pollResults() {
  if (resultsDirty) { resultsDirty = false; try { renderResults(await api('/api/results')); } catch (e) { resultsDirty = true; } }
  setTimeout(pollResults, 1500);
}

// ------------------------------------------------------------------ charts (plain canvas)
function chart(cv, series, opt = {}) {
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  if (!W) return;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const L = 46, R = 8, T = 30, B = 18;
  const xs = [], ys = [];
  for (const s of series) for (const p of s.data) if (p[1] !== null && Number.isFinite(p[1])) { xs.push(p[0]); ys.push(p[1]); }
  if (opt.hline !== undefined) ys.push(opt.hline);
  g.fillStyle = '#9fb6d6'; g.font = '11px system-ui'; g.fillText(opt.title || '', L, 11);
  if (!xs.length) { g.fillStyle = '#7d8aa0'; g.fillText('N/A (no data yet)', L + 10, H / 2); return; }
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = opt.ymin ?? Math.min(...ys), y1 = opt.ymax ?? Math.max(...ys);
  if (x1 === x0) x1 = x0 + 1; if (y1 === y0) { y1 += 1; y0 -= 1; }
  const X = (x) => L + (x - x0) / (x1 - x0) * (W - L - R), Y = (y) => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  g.strokeStyle = '#1f2a3b'; g.beginPath(); g.moveTo(L, T); g.lineTo(L, H - B); g.lineTo(W - R, H - B); g.stroke();
  g.fillStyle = '#7d8aa0'; g.font = '10px system-ui';
  const fmt = (v) => Math.abs(v) >= 1e4 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(1) : (+v.toFixed(3)).toString();
  g.fillText(fmt(y1), 2, T + 4); g.fillText(fmt(y0), 2, H - B); g.fillText(fmt(x0), L, H - 4);
  g.fillText(fmt(x1) + (opt.xlabel ? ' ' + opt.xlabel : ''), W - R - 80, H - 4);
  if (opt.hline !== undefined) { g.strokeStyle = '#3ddc84'; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(L, Y(opt.hline)); g.lineTo(W - R, Y(opt.hline)); g.stroke(); g.setLineDash([]); }
  let lx = L;
  for (const s of series) {
    g.strokeStyle = g.fillStyle = s.color; g.globalAlpha = s.alpha ?? 1;
    if (s.dash) g.setLineDash(s.dash);
    if (s.dots) { for (const p of s.data) if (p[1] !== null) g.fillRect(X(p[0]) - 1, Y(p[1]) - 1, s.size || 2, s.size || 2); }
    else if (s.bars) { for (const p of s.data) if (p[1] !== null) g.fillRect(X(p[0]) - 1, Y(Math.max(p[1], y0)), 2.5, Y(y0) - Y(Math.max(p[1], y0))); }
    else { g.beginPath(); let on = false; for (const p of s.data) { if (p[1] === null) { on = false; continue; } on ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1])); on = true; } g.stroke(); }
    g.setLineDash([]); g.globalAlpha = 1;
    if (s.label) { g.fillText(s.label, lx, 24); lx += g.measureText(s.label).width + 12; }
  }
}
const movavg = (a, n) => a.map((_, i) => { const w = a.slice(Math.max(0, i - n + 1), i + 1).filter((v) => v !== null); return w.length ? w.reduce((x, y) => x + y, 0) / w.length : null; });

// ------------------------------------------------------------------ results rendering
const TH = (label, title, cls = '') => `<th class="${cls}" title="${esc(title)}">${label}</th>`;
let lastR = null;
function renderResults(r) {
  lastR = r;
  DOCS = { outputs: Object.fromEntries(r.output_docs.map((o) => [o.key, o])), reason_docs: r.reason_docs, enablers: r.enablers || [], problems: r.problems || [] };
  setTargetRing(r.config.target_au);
  buildParams(r);
  $('speedInfo').title = docText(r.speed_doc);
  // training curves
  const cfg = r.config, h = r.history, ep = h.map((x) => x.ep), rew = h.map((x) => x.reward);
  chart($('chReward'), [
    { data: h.map((x, i) => [ep[i], rew[i]]), color: '#4fa3ff', dots: true, alpha: 0.35, label: 'episode return (exploring)' },
    { data: movavg(rew, 50).map((v, i) => [ep[i], v]), color: '#ffffff', label: 'avg of 50' },
    { data: r.evals.map((e) => [e.episode, e.reward]), color: '#ffcc33', dots: true, size: 4, label: 'deterministic test' },
  ], { title: 'Training score: return per episode [points]', xlabel: 'episode' });
  chart($('chA'), [
    { data: h.map((x, i) => [ep[i], x.a]), color: '#7cc4ff', dots: true, alpha: 0.35, label: 'final a (exploring)' },
    { data: r.evals.map((e) => [e.episode, e.a_au]), color: '#ffcc33', label: 'deterministic test' },
  ], { title: 'Physics: final orbit size a [AU] (green = target)', hline: cfg.target_au, xlabel: 'episode' });
  chart($('chRes'), [
    { data: movavg(h.map((x) => x.mass), 50).map((v, i) => [ep[i], v]), color: '#ff7a45', label: 'asteroid used' },
    { data: movavg(h.map((x) => x.energy), 50).map((v, i) => [ep[i], v]), color: '#c084fc', label: 'energy used' },
    { data: movavg(h.map((x) => (x.success ? 1 : 0)), 50).map((v, i) => [ep[i], v]), color: '#3ddc84', label: 'success rate' },
  ], { title: 'Physics: resources used and success (fraction, avg of 50)', ymin: 0, ymax: 1, xlabel: 'episode' });
  const oc = r.training_outcomes || {}, tot = Object.values(oc).reduce((a, b) => a + b, 0);
  $('outcomes').innerHTML = tot ? '<table>' + Object.entries(oc).map(([k, n]) => `<tr><td title="${esc(r.reason_docs[k] || '')}">${esc(k)}</td><td>${n}</td><td>${pct(n / tot)}</td><td class="l dim">${esc(r.reason_docs[k] || '')}</td></tr>`).join('') + '</table>' : '<span class="dim">N/A – no training episodes in this session yet.</span>';
  renderComparison(r);
  renderGuide(r);
}

function renderComparison(r) {
  const c = r.comparison, cfg = r.config;
  if (!c) { $('verdict').className = 'verdict'; $('verdict').textContent = 'No comparison yet (runs automatically after training, or press “Compare policies”).'; $('cmpTable').innerHTML = ''; $('bestPanel').innerHTML = ''; chart($('chTraj'), [], { title: 'a(t)' }); return; }
  const rows = c.rows, ppo = rows.find((x) => x.policy === 'ppo');
  const vd = $('verdict');
  if (!ppo) { vd.className = 'verdict'; vd.textContent = 'No trained model – baselines only.'; }
  else if (ppo.mission_success) {
    vd.className = 'verdict ok';
    vd.textContent = `Target reached by the best PPO model in the high-fidelity N-body run, and the orbit stayed stable for ${cfg.validate_years} years without thrust: a = ${num(ppo.hifi.a_au)} AU, e = ${num(ppo.hifi.e)} after ${int(ppo.hifi.t_years)} simulated years.`;
  } else {
    vd.className = 'verdict bad';
    vd.textContent = `Target NOT reached by the best PPO model in the high-fidelity run (${ppo.hifi.reason}): a = ${num(ppo.hifi.a_au)} AU (short by ${num(ppo.hifi.shortfall_au)} AU), e = ${num(ppo.hifi.e)}.`;
  }
  let t = '<table><tr>' + TH('policy', 'Decision rule that controls the ejections.') + TH('fast<br>success', 'Success rate on the fast model, 3 starting seeds.') +
    TH('hi-fi<br>outcome', 'How the high-fidelity N-body run ended.') + TH('a [AU]', 'Final semi-major axis (orbit size), high-fidelity.') + TH('e', 'Final eccentricity, high-fidelity.') +
    TH('years', 'Simulated years until the mission ended.') + TH('asteroid<br>used', 'Fraction of the asteroid thrown away.') + TH('energy<br>[J]', 'Propulsion energy drawn.') +
    TH('sunlight', 'Insolation at the final orbit relative to today (estimate).') + TH('stable', 'Passed the ballistic stability check (high-fidelity).') + '</tr>';
  for (const x of rows) {
    t += `<tr><td>${esc(x.label)}</td><td>${pct(x.fast.success_rate, 0)}</td><td>${esc(x.hifi.reason)}</td><td>${num(x.hifi.a_au)}</td><td>${num(x.hifi.e)}</td><td>${int(x.hifi.t_years)}</td>` +
      `<td>${pct(x.hifi.mass_used_frac)}</td><td>${sci(x.hifi.energy_used_J)}</td><td>${num(x.hifi.insolation_rel, 3)}×</td><td class="${x.mission_success ? 'pass' : 'fail'}">${x.mission_success ? 'yes' : 'no'}</td></tr>`;
  }
  if (rows.some((x) => x.hifi.solar_on)) {
    t += '</table><table style="margin-top:6px"><tr>' + TH('policy', '') + TH('energy from<br>collectors [J]', 'Electric energy delivered by the microwave beams.') + TH('from<br>reserve [J]', 'Energy taken from the starting reserve.') +
      TH('mean beam<br>power [W]', 'Microwave power absorbed by the 8 dishes, averaged over the mission.') + TH('heat at arms<br>(× Earth sunlight)', 'Rectenna + launcher losses, mean; estimate, not simulated.') + '</tr>';
    for (const x of rows) if (x.hifi.solar_on) t += `<tr><td>${esc(x.label)}</td><td>${sci(x.hifi.energy_from_collectors_J)}</td><td>${sci(x.hifi.energy_from_reserve_J)}</td><td>${sci(x.hifi.beam_mean_W)}</td><td>${sci(x.hifi.arm_heat_vs_earth_sunlight, 1)}×</td></tr>`;
  }
  $('cmpTable').innerHTML = t + '</table><div class="small dim">Physical results only (reward is a training score and is listed separately in the best-model section). High-fidelity = N-body Sun, Earth, Moon, fragments; seed 2001.</div>';
  const col = { none: '#888', heuristic: '#ffcc33', ppo: '#4fa3ff' }, ser = [];
  for (const [k, tr] of Object.entries(c.traces)) {
    ser.push({ data: [[0, 1]].concat(tr.fast.map((d) => [d.t, d.a])), color: col[k], label: k });
    ser.push({ data: [[0, 1]].concat(tr.hifi.map((d) => [d.t, d.a])), color: col[k], dash: [3, 3], alpha: 0.7 });
  }
  chart($('chTraj'), ser, { title: 'Orbit size a(t) [AU] — solid fast model, dashed high-fidelity', hline: cfg.target_au, xlabel: 'yr' });
  renderBest(c, r);
}

function renderBest(c, r) {
  const bp = c.best_panel, cfg = r.config;
  if (!bp) { $('bestPanel').innerHTML = '<div class="note">This comparison was produced by an older version and has no best-model details. Press “Compare policies” to regenerate it.</div>'; return; }
  let h = '<h3>Best model</h3>';
  const f = bp.file, sel = bp.selection;
  h += `<div class="card small"><b>File:</b> ${esc(f.name)} ${f.exists ? '' : '<span class="fail">(missing)</span>'}<br><span class="dim">${esc(f.path)}</span><br>` +
    (f.exists ? `${f.size_kb} kB · saved ${esc(f.modified)}<br>` : '') +
    (sel ? `<b>Chosen at</b> training episode ${sel.episode} (${int(sel.timesteps)} decisions).<br><b>Selection rule:</b> ${esc(sel.selection)}.<br>` +
      `<b>Test at selection:</b> success ${pct(sel.eval.success_rate, 0)}, a = ${num(sel.eval.a_au)} AU, e = ${num(sel.eval.e)}, energy ${pct(sel.eval.energy_frac)}, ${int(sel.eval.years)} yr.<br>` +
      `<b>Network &amp; settings:</b> <span class="mono">${esc(Object.entries(sel.hyperparameters || {}).map(([k, v]) => `${k}=${Array.isArray(v) ? v.join('x') : v}`).join(', '))}</span>` : 'Selection info: N/A') + '</div>';
  h += bp.disclaimer.map((d) => `<div class="note">${esc(d)}</div>`).join('');
  // evals
  if (r.evals && r.evals.length) {
    h += '<h4>Deterministic tests during training <i class="info" title="Every few episodes the current policy is tested without random exploration on seeds 1001-1002. The best test result becomes best_model.zip.">ⓘ</i></h4><div class="scroll"><table><tr>' +
      TH('episode', 'Training episode at which the test was run.') + TH('success', 'Fraction of test episodes that reached the target.') + TH('a [AU]', 'Mean final orbit size.') +
      TH('e', 'Mean final eccentricity.') + TH('energy', 'Fraction of energy budget used.') + TH('years', 'Mean simulated years to finish.') + TH('score', 'Mean training reward (score, not physics).') + '</tr>';
    for (const e of r.evals) {
      const best = sel && e.episode === sel.episode;
      h += `<tr style="${best ? 'background:#1e3a5f' : ''}"><td>${e.episode}${best ? ' (best)' : ''}</td><td>${pct(e.success_rate, 0)}</td><td>${num(e.a_au)}</td><td>${num(e.e)}</td><td>${pct(e.energy_frac)}</td><td>${int(e.years)}</td><td>${num(e.reward, 1)}</td></tr>`;
    }
    h += '</table></div>';
  }
  if (!bp.available) { $('bestPanel').innerHTML = h + '<div class="note">No PPO model in this comparison.</div>'; return; }
  // comparison
  h += '<h4>Compared with the baselines (high-fidelity run)</h4><table><tr>' + TH('policy', 'Decision rule.') + TH('success', 'Reached the target and passed the stability check.') +
    TH('a [AU]', 'Final orbit size.') + TH('e', 'Final eccentricity.') + TH('years', 'Simulated years.') + TH('asteroid', 'Fraction of asteroid used.') + TH('energy [J]', 'Energy drawn.') +
    TH('score', 'Total training reward of that run (a score, NOT a physical quantity).') + '</tr>';
  for (const x of bp.comparison) h += `<tr><td>${esc(x.policy)}</td><td class="${x.success ? 'pass' : 'fail'}">${x.success ? 'yes' : 'no'}</td><td>${num(x.a)}</td><td>${num(x.e)}</td><td>${int(x.years)}</td><td>${pct(x.mass)}</td><td>${sci(x.energy_J)}</td><td class="dim">${num(x.reward, 1)}</td></tr>`;
  h += '</table>';
  // criteria
  h += '<h4>Final orbit vs. target criteria</h4><table><tr>' + TH('criterion', 'What is tested.') + TH('value', 'Value reached in the high-fidelity run.') + TH('required', 'Requirement.') + TH('', 'Pass or fail.') + '</tr>';
  for (const k of bp.criteria) {
    const v = typeof k.value === 'boolean' ? (k.value ? 'yes' : 'no') : (typeof k.value === 'number' ? (Math.abs(k.value) < 0.01 && k.value !== 0 ? sci(k.value) : num(k.value, k.unit === 'yr' ? 0 : 4)) : 'N/A');
    h += `<tr><td>${esc(k.name)}</td><td>${v} ${esc(k.unit)}</td><td>${esc(k.required)}</td><td class="${k.passed ? 'pass' : 'fail'}">${k.passed ? 'PASS' : 'FAIL'}</td></tr>`;
  }
  h += '</table>';
  // actions
  const cls = bp.action_classes || {};
  h += `<h4>What the model actually did, decision by decision <i class="info" title="Recorded from the high-fidelity run of best_model.zip. 'Helped' = the orbit grew and e stayed within tolerance or decreased; 'mixed' = grew but e rose above tolerance; 'hurt' = orbit shrank or overshot; 'coast' = no ejection.">ⓘ</i></h4>`;
  h += `<div class="small">${Object.entries(cls).map(([k, n]) => `<span class="c-${k}">${k}: ${n}</span>`).join(' · ')} (of ${bp.actions.length} decisions of ${cfg.decision_interval_years} years)</div>`;
  h += '<canvas id="chAct" class="chart"></canvas>';
  h += '<div class="scroll"><table><tr>' + TH('years', 'Decision interval (simulated years).') + TH('throttle', 'Fraction of the maximum packet mass chosen.') + TH('packets', 'Number of packets in this interval.') +
    TH('packet [kg]', 'Mass of each packet.') + TH('speed [km/s]', 'Launch speed of the packets.') + TH('thrust dir.', 'Direction of the push on Earth (packets fly the opposite way). RTN = radial, along-track, normal cosines.') +
    TH('Δv [m/s]', 'Speed change given to Earth-Moon in this interval.') + TH('Δa [AU]', 'Change of orbit size in this interval.') + TH('e after', 'Eccentricity at the end of the interval.') +
    TH('effect', 'Classification from the recorded change of a and e.') + TH('why (from recorded data)', 'Explanation built only from the recorded numbers.', 'l') + '</tr>';
  for (const a of bp.actions) {
    h += `<tr><td>${int(a.t - a.dt)}–${int(a.t)}</td><td>${num(a.thr, 2)}</td><td>${int(a.npk)}</td><td>${a.dm > 0 ? sci(a.dm) : '—'}</td><td>${a.dm > 0 ? int(a.u / 1000) : '—'}</td>` +
      `<td title="RTN ${a.d_rtn.map((x) => x.toFixed(2)).join(', ')}">${a.dm > 0 ? dirWords(a.d_rtn) : '—'}</td><td>${num(a.dv, 1)}</td><td>${a.da >= 0 ? '+' : ''}${num(a.da)}</td><td>${num(a.e)}</td>` +
      `<td class="c-${a.cls}">${a.cls}</td><td class="l small">${esc(a.why)}</td></tr>`;
  }
  h += '</table></div>';
  if (bp.top_helpful && bp.top_helpful.length) {
    h += '<h4>Decisions that helped most</h4><ul class="small">' + bp.top_helpful.map((a) => `<li>Years ${int(a.t - a.dt)}–${int(a.t)}: Δa ${a.da >= 0 ? '+' : ''}${num(a.da)} AU (${pct(a.share)} of all orbit growth) — ${esc(a.why)}; ${int(a.npk)} packets at ${int(a.u / 1000)} km/s, ${sci(a.da_per_1e19kg)} AU per 1e19 kg.</li>`).join('') + '</ul>';
  }
  h += '<h4>Why it behaves this way</h4><ul class="small">' + (c.explanation.summary || []).map((s) => `<li>${esc(s)}</li>`).join('') + '</ul>';
  // failures
  h += '<h4>Failed attempts, exhausted resources, constraint violations</h4>';
  const oc = bp.training_outcomes || {}, tot = Object.values(oc).reduce((a, b) => a + b, 0);
  h += tot ? `<div class="small">Training episodes (with random exploration): ${Object.entries(oc).map(([k, n]) => `<span title="${esc(r.reason_docs[k] || '')}">${esc(k)} ${n} (${pct(n / tot, 0)})</span>`).join(' · ')}</div>` : '<div class="small dim">Training outcome counts: N/A (training history not available in this session).</div>';
  h += bp.failed_evals && bp.failed_evals.length ? `<div class="small">Deterministic tests that did not fully succeed: ${bp.failed_evals.length} (episodes ${bp.failed_evals.slice(0, 12).map((e) => e.episode).join(', ')}${bp.failed_evals.length > 12 ? ', …' : ''}).</div>` : '<div class="small">All deterministic tests of this run succeeded (or none recorded).</div>';
  h += '<table>' + bp.constraints.map((k) => `<tr><td>${esc(k.name)}</td><td>${typeof k.value === 'number' ? (k.unit === 'fraction' ? pct(k.value) : int(k.value)) : esc(k.value)}</td><td class="l dim">${esc(k.note)}</td></tr>`).join('') + '</table>';
  $('bestPanel').innerHTML = h;
  const A = bp.actions;
  chart($('chAct'), [
    { data: A.map((a) => [a.t, a.thr]), color: '#ffcc33', label: 'throttle (0-1)' },
    { data: A.map((a) => [a.t, a.dm > 0 ? a.u / 1e7 : null]), color: '#c084fc', label: 'speed / 10,000 km/s' },
    { data: A.map((a) => [a.t, a.dm > 0 ? a.d_rtn[1] : null]), color: '#7fe08a', label: 'prograde cosine' },
    { data: A.map((a) => [a.t, Math.max(0, a.da) * 10]), color: '#4fa3ff', bars: true, label: 'Δa×10 [AU]' },
  ], { title: 'Actions over time (high-fidelity run)', xlabel: 'yr', ymin: -0.2, ymax: 1.1 });
}

// ------------------------------------------------------------------ guide
function docText(d) {
  if (!d) return '';
  const rng = d.min === null || d.min === undefined ? '' : `\nAllowed range: ${d.min} .. ${d.max} ${d.unit}`;
  return `${d.label} [${d.unit}] — ${d.category}\n${d.meaning}\nWhy it matters: ${d.why}\nIncrease: ${d.increase}\nDecrease: ${d.decrease}${rng}${d.constraint ? '\nConstraint: ' + d.constraint : ''}`;
}
function fmtVal(v) { return typeof v === 'number' && (Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0)) ? v.toExponential(4).replace('e+', 'e') : String(v); }
let guideBuilt = '';
function renderGuide(r) {
  const key = JSON.stringify([r.config, r.checks && r.checks.length, r.feasibility.verdict]);
  if (key === guideBuilt) return; guideBuilt = key;
  const g = [];
  const det = (sum, body, tags = '') => `<details><summary>${sum}${tags}</summary><div class="gd">${body}</div></details>`;
  g.push('<div class="note">How to use: 1) check the feasibility pill (green = the target is physically reachable in this model); 2) train; 3) compare policies; 4) watch the best model at a slow speed to see single packets, or fast to see the whole spiral. Hover any ⓘ, table header or telemetry line for an explanation.</div>');
  g.push('<h4>Feasibility in plain English</h4><ul class="small">' + (r.feasibility.plain || []).map((x) => `<li>${esc(x)}</li>`).join('') + '</ul>');
  g.push('<h4>Configurable parameters</h4>');
  const cats = [...new Set(r.param_docs.map((d) => d.category))];
  for (const c of cats) {
    g.push(`<div class="small dim" style="margin-top:6px">${esc(c)}</div>`);
    for (const d of r.param_docs.filter((x) => x.category === c)) {
      const val = r.config[d.key], def = r.defaults[d.key];
      g.push(det(`${esc(d.label)} <span class="u">= ${esc(fmtVal(val))} ${esc(d.unit)}${val !== def ? ` (default ${esc(fmtVal(def))})` : ''}</span>`,
        `<b>Meaning:</b> ${esc(d.meaning)}<br><b>Why it matters:</b> ${esc(d.why)}<br><b>Increase:</b> ${esc(d.increase)}<br><b>Decrease:</b> ${esc(d.decrease)}<br>` +
        (d.min !== null ? `<b>Allowed:</b> ${d.min} .. ${d.max} ${esc(d.unit)}<br>` : '') + (d.constraint ? `<b>Constraint:</b> ${esc(d.constraint)}<br>` : '') + `<b>Config key:</b> <span class="mono">${d.key}</span>`,
        `<span class="tag">${esc(d.category)}</span>`));
    }
  }
  const sd = r.speed_doc;
  g.push(det(`${esc(sd.label)} <span class="u">= ${fmtSpeed(speedY)}/s</span>`, `<b>Meaning:</b> ${esc(sd.meaning)}<br><b>Why:</b> ${esc(sd.why)}<br><b>Increase:</b> ${esc(sd.increase)}<br><b>Decrease:</b> ${esc(sd.decrease)}<br><b>Range:</b> ${esc(sd.constraint)}`, `<span class="tag">${esc(sd.category)}</span>`));
  g.push('<h4>Physical constants (fixed)</h4><table>' + r.constants.map((k) => `<tr><td title="${esc(k.meaning)}">${esc(k.name)}</td><td>${sci(k.value, 6)}</td><td class="l">${esc(k.unit)}</td></tr>`).join('') + '</table>');
  g.push('<h4>Outputs and results</h4>');
  for (const o of r.output_docs) g.push(det(`${esc(o.label)} <span class="u">[${esc(o.unit)}]</span>`, `<b>Measures:</b> ${esc(o.measures)}<br><b>How calculated:</b> ${esc(o.how)}<br><b>Type:</b> ${esc(o.kind)}<br><b>Desirable:</b> ${esc(o.better)}<br><b>Meaning for success:</b> ${esc(o.success)}`, `<span class="tag">${esc(o.kind)}</span>`));
  g.push('<h4>How an episode can end</h4><table>' + Object.entries(r.reason_docs).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="l">${esc(v)}</td></tr>`).join('') + '</table>');
  g.push('<h4>Physics checks (run at every start)</h4>');
  for (const k of r.checks || []) {
    const id = (k.name.match(/^(\d+)/) || [])[1];
    g.push(det(`<span class="${k.passed ? 'pass' : 'fail'}">${k.passed ? 'PASS' : 'FAIL'}</span> ${esc(k.name)}`, `<b>What it tests:</b> ${esc(r.check_docs[id] || '')}<br><b>Measured:</b> <span class="dim">${esc(k.detail)}</span>`));
  }
  g.push('<h4>Assumptions, approximations and limitations</h4>');
  for (const [a, b] of r.assumptions) g.push(det(`${esc(a)}`, esc(b)));
  g.push('<h4>Glossary</h4>');
  for (const [a, b] of r.glossary) g.push(det(esc(a), esc(b)));
  g.push(`<div class="small dim" style="margin-top:8px">Results folder: ${esc(r.out_dir)}</div>`);
  $('guide').innerHTML = g.join('');
  filterGuide();
}
function filterGuide() {
  const q = $('guideSearch').value.trim().toLowerCase();
  for (const d of document.querySelectorAll('#guide details')) {
    const hit = !q || d.textContent.toLowerCase().includes(q);
    d.style.display = hit ? '' : 'none'; if (q && hit) d.open = true;
  }
}
$('guideSearch').addEventListener('input', filterGuide);

// ------------------------------------------------------------------ parameters panel
let paramsKey = '';
function buildParams(r, force = false) {
  const key = JSON.stringify(r.config);
  if (key === paramsKey && !force) return; paramsKey = key;
  const sel = $('preset');
  sel.innerHTML = r.presets.map((p) => `<option ${p === r.config.preset ? 'selected' : ''}>${p}</option>`).join('') +
    (r.presets.includes(r.config.preset) ? '' : `<option selected>${r.config.preset}</option>`);
  let h = '', cat = '';
  const order = [...new Set(r.param_docs.map((x) => x.category))];
  const docs = r.param_docs.slice().sort((a, b) => order.indexOf(a.category) - order.indexOf(b.category));
  for (const d of docs) {
    if (d.category !== cat) { cat = d.category; h += `<div class="cat">${esc(cat)}</div>`; }
    const v = r.config[d.key];
    const inp = typeof v === 'boolean' ? `<select data-k="${d.key}" data-bool="1"><option ${v ? 'selected' : ''}>true</option><option ${!v ? 'selected' : ''}>false</option></select>`
      : `<input data-k="${d.key}" data-orig="${esc(fmtVal(v))}" value="${esc(fmtVal(v))}" title="${esc(`allowed ${d.min} .. ${d.max} ${d.unit}; default ${fmtVal(r.defaults[d.key])}`)}">`;
    h += `<div class="row"><span>${esc(d.label)} <span class="dim">[${esc(d.unit)}]</span></span><i class="info" title="${esc(docText(d) + `\nDefault: ${fmtVal(r.defaults[d.key])}`)}">ⓘ</i>${inp}</div>`;
  }
  $('params').innerHTML = h;
  for (const inp of document.querySelectorAll('#params input')) {
    inp.addEventListener('input', () => {
      const d = r.param_docs.find((x) => x.key === inp.dataset.k), x = Number(inp.value);
      inp.classList.toggle('bad', !(Number.isFinite(x) && x >= d.min && x <= d.max));
      inp.classList.toggle('changed', inp.value !== inp.dataset.orig);
    });
  }
}
async function applyParams(body) {
  try {
    const res = await api('/api/config', body);
    $('paramMsg').className = 'small ok'; $('paramMsg').textContent = `Applied (preset ${res.config.preset}). Feasibility: ${res.feasibility.verdict}`;
    resultsDirty = true; paramsKey = '';
  } catch (e) { $('paramMsg').className = 'small err'; $('paramMsg').textContent = 'Rejected: ' + e.message; }
}
$('btnApply').onclick = () => {
  const body = {};
  const preset = $('preset').value;
  if (lastR && preset !== lastR.config.preset) body.preset = preset;     // preset change: preset values win
  else for (const el of document.querySelectorAll('#params [data-k]')) if (el.dataset.bool || el.value !== el.dataset.orig) body[el.dataset.k] = el.value;
  applyParams(body);
};
$('btnDefaults').onclick = () => applyParams({ reset_defaults: true });

// ------------------------------------------------------------------ buttons
for (const b of document.querySelectorAll('.tab')) b.onclick = () => {
  for (const x of document.querySelectorAll('.tab')) x.classList.toggle('active', x === b);
  for (const p of document.querySelectorAll('.tabpane')) p.classList.toggle('active', p.id === b.dataset.tab);
  resultsDirty = true;
};
$('btnTrain').onclick = () => api('/api/train', { action: 'start', device: $('trainDevice').value })
  .then(() => toast(`training started (device: ${$('trainDevice').selectedOptions[0].textContent})`)).catch((e) => toast(e.message));
let TRAIN_EP = 0, TRAIN_LIVE = true, LIVE_SET_AT = 0, DEV_SET_AT = 0, TRAIN_RUNNING = false;
function setLive(on) {
  TRAIN_LIVE = on; LIVE_SET_AT = Date.now(); $('trainLive').checked = on; markShow();
  return api('/api/train', { action: 'live', on }).catch((e) => toast(e.message));
}
function markShow() {
  const cur = TRAIN_LIVE && (V.mode === 'training' || V.mode === 'idle') ? 'live' : (V.mode === 'playback' ? V.policy : '');
  for (const b of document.querySelectorAll('#showSeg button')) b.classList.toggle('on', b.dataset.show === cur);
}
$('trainLive').addEventListener('change', () => { setLive($('trainLive').checked); toast($('trainLive').checked ? 'live training view on' : 'live training view off – training continues in the background'); });
$('trainDevice').addEventListener('change', () => { DEV_SET_AT = Date.now(); api('/api/train', { action: 'device', device: $('trainDevice').value }).catch(() => {}); toast('device is used from the next training start'); });
for (const b of document.querySelectorAll('#showSeg button')) b.onclick = () => {
  if (b.dataset.show === 'live') {
    if (!TRAIN_RUNNING) toast('training is not running – press Start training to watch it learn');
    setLive(true).then(() => { V.playing = true; });
    return;
  }
  $('playPolicy').value = b.dataset.show; startRun();
};
$('btnStop').onclick = () => api('/api/train', { action: 'stop' }).then(() => toast('stopping – saving model')).catch((e) => toast(e.message));
$('btnCompare').onclick = () => api('/api/compare', {}).then(() => toast('comparison started')).catch((e) => toast(e.message));
function startRun() {
  const pre = TRAIN_LIVE ? setLive(false) : Promise.resolve();   // the screen is taken by the mission; training continues
  return pre.then(() => api('/api/play', { policy: $('playPolicy').value, fidelity: $('playFid').value })
    .then(() => { V.playing = true; toast(`mission started: ${$('playPolicy').value}, ${$('playFid').value}`); })).catch((e) => toast(e.message));
}
$('btnPlay').onclick = () => {
  if (V.mode === 'training' && !V.playing) { V.playing = true; return; }   // resume the training replay
  const same = V.mode === 'playback' && V.policy === $('playPolicy').value && V.fidelity === $('playFid').value;
  const finished = V.done && V.rows.length && V.disp >= V.rows[V.rows.length - 1][R_T] - 1e-9;
  if (same && !finished) { V.playing = true; return; }
  startRun();
};
$('btnPause').onclick = () => { V.playing = false; };
$('follow').addEventListener('change', () => {
  if ($('follow').checked) { controls.target.copy(earth.position); camera.position.copy(earth.position).add(new THREE.Vector3(0, -4.5, 3.6)); }
  else { controls.target.set(HOME_X, 0, 0); camera.position.set(HOME_X, -40, 34); }
});
$('btnReset').onclick = () => {
  if (V.mode === 'training') { V.trJump = true; V.playing = true; return; }  // training view: newest finished episode
  if (V.mode === 'playback' && V.policy) { $('playPolicy').value = V.policy; $('playFid').value = V.fidelity; }
  startRun();
};
setInterval(() => { const live = V.mode === 'playback' || V.mode === 'training';
  $('btnPlay').classList.toggle('on', live && V.playing); $('btnPause').classList.toggle('on', live && !V.playing); }, 200);

// ------------------------------------------------------------------ Earth POV controls
const LEGEND_MAIN = $('legend').innerHTML;
const POV_LEG = {
  arms: '<span style="color:#ff8a7a" title="Illustration only – no hardware is simulated.">⊙ ring + 8 arms (illustration)</span>',
  moon: '<span style="color:#dddddd">● Moon (true size &amp; distance)</span>', moonorbit: '<span style="color:#bbbbbb">— Moon orbit</span>',
  field: '<span style="color:#9fd8ff">— field lines (pale cyan strong → violet weak)</span>',
  magnetopause: '<span style="color:#5ee0ff">◠ magnetopause</span> <span style="color:#ffa040">◠ bow shock</span>',
  belts: '<span style="color:#ff9a6a">◯ radiation belts (schematic)</span>',
  wind: '<span style="color:#ffd27a">· solar wind</span> <span style="color:#ff8040">· slowed/heated wind</span>',
  gravwell: '<span style="color:#c8b8ff">▦ gravity well (log height)</span>', hill: '<span style="color:#9be37a">○ Hill sphere ◆ L1/L2</span>',
  gravvec: '<span style="color:#ffcc33">→ Sun pull</span> <span style="color:#dddddd">→ Moon pull</span>',
  motion: '<span style="color:#7cc4ff">→ orbital velocity</span> <span style="color:#7fe08a">→ push (recoil)</span>',
  mars: '<span style="color:#ff6a4d">● Mars direction (true distance in label)</span> <span style="color:#ff6a4d">→ Mars pull</span>',
  mirrors: '<span style="color:#dfe9ff" title="True orbit distance (20 Earth radii); mirrors drawn enlarged.">▫ space mirrors (true 20 R<sub>E</sub> orbit) — thin light lines: sunlight in, reflected to Earth</span>',
  beams: '<span style="color:#c8a8ff" title="Beam brightness = recorded beam power; microwaves are invisible (colour illustrative).">microwave beams → arm rectennas (true width)</span>',
  collectors: '<span style="color:#ffd27a">· solar collector belt (true directions)</span>',
  packets: '<span style="color:#ff9a4d">— escaping</span> <span style="color:#ffe14d">— Sun-bound</span> <span style="color:#ff5a5a">— fell back</span>',
};
let legendKey = '';
function updateLegend() {
  if (wxOn()) {
    if (legendKey === 'wx') return; legendKey = 'wx';
    $('legend').innerHTML = '<span style="color:#ffcc33">● Sun (direction; 3× its true size)</span><span style="color:#dfe9ff">▫ space mirrors (drawn at 2.6 R<sub>E</sub>, real 20 R<sub>E</sub>)</span><span style="color:#fffaf0">— thin light lines: sunlight arriving on each mirror → reflected down to Earth (very faint short lines: mirrors not needed now, light turned past Earth)</span><span style="color:#ffffff">clouds (illustration, ∝ water vapour)</span><span style="color:#9fd0ff">— wind streaks (three-cell circulation, illustration)</span><span style="color:#7ed957">■ growing zone</span><span style="color:#e8f0ff">■ ice / snow</span><span style="color:#ffcf7a">· city lights (night side)</span>';
    return;
  }
  if (colOn()) {
    if (legendKey === 'col') return; legendKey = 'col';
    $('legend').innerHTML = '<span style="color:#9cc0ff">▮ solar cells (face the Sun)</span><span style="color:#ffd27a">▮ gold-foil bus</span><span style="color:#d8c0ff">◎ microwave transmitter → Earth</span><span style="color:#c8a8ff">— power beam (bright = launch, dim = stand-by)</span><span style="color:#ffcc33">● Sun</span><span style="color:#7cc4ff">● Earth</span>';
    return;
  }
  const key = povOn() ? 'pov:' + POV_LAYERS.filter((l) => pov.layers[l.key]).map((l) => l.key).join(',') : 'main';
  if (key === legendKey) return; legendKey = key;
  $('legend').innerHTML = key === 'main' ? LEGEND_MAIN : '<span style="color:#ffcc33">● Sun (true angular size)</span><span style="color:#4fa3ff">● Earth</span>' +
    POV_LAYERS.filter((l) => pov.layers[l.key] && POV_LEG[l.key]).map((l) => POV_LEG[l.key]).join('');
}
function savePovLayers() { if (beamOn()) return; try { localStorage.setItem('oe_pov_layers', JSON.stringify(pov.layers)); } catch (e) { /* ignore */ } }
function syncPovBoxes() { for (const l of POV_LAYERS) { const c = $('lay_' + l.key); if (c) c.checked = !!pov.layers[l.key]; } }
{
  try { const s = JSON.parse(localStorage.getItem('oe_pov_layers') || 'null'); if (s) for (const l of POV_LAYERS) if (typeof s[l.key] === 'boolean') pov.layers[l.key] = s[l.key]; } catch (e) { /* ignore */ }
  $('povLayers').innerHTML = POV_LAYERS.map((l) => `<label title="${esc(l.tip)}"><input type="checkbox" id="lay_${l.key}" /> ${esc(l.label)} <i class="info">ⓘ</i></label>`).join('');
  for (const l of POV_LAYERS) $('lay_' + l.key).addEventListener('change', (e) => { pov.setLayer(l.key, e.target.checked); savePovLayers(); });
  syncPovBoxes();
  $('povNone').onclick = () => { for (const l of POV_LAYERS) pov.setLayer(l.key, false); syncPovBoxes(); savePovLayers(); };
  $('povDef').onclick = () => { for (const l of POV_LAYERS) pov.setLayer(l.key, POV_DEFAULTS[l.key]); syncPovBoxes(); savePovLayers(); pov.zoomTo(9); };
  for (const b of document.querySelectorAll('[data-look]')) b.onclick = () => pov.look(b.dataset.look);
  $('povLock').addEventListener('change', () => { pov.P.lockSun = $('povLock').checked; });
  $('viewMode').addEventListener('change', () => {
    const on = povOn();
    pov.setBeamView(beamOn()); syncPovBoxes(); legendKey = '';
    $('chipBeam').classList.toggle('on', beamOn());
    const co = colOn(), w = wxOn();
    $('chipCol').classList.toggle('on', co); col.setActive(co); $('colBox').hidden = !co;
    $('chipWx').classList.toggle('on', w); wx.setActive(w); $('wxBox').hidden = !w;
    $('povBox').hidden = !on; $('ovBox').hidden = on || co || w; controls.enabled = !on && !co && !w; pov.setActive(on); $('follow').disabled = on || co || w;
    $('chipView').classList.toggle('on', on && !beamOn()); $('chipView').textContent = on || co || w ? '◉ Overview' : '◉ POV';
    UI.view = $('viewMode').value; saveUI();
  });
}

// ------------------------------------------------------------------ uncluttered UI: drawers, chips, keys, quick start
const UI = { power: true, drawer: null, hudMore: false, events: false, legend: false, notes: true, hud: true, notesOpen: false, view: 'overview', speed: 560, qs: false, ov: {} };
function saveUI() { try { UI.speed = +$('speed').value; UI.ov = OV; localStorage.setItem('oe_ui', JSON.stringify(UI)); } catch (e) { /* ignore */ } }
try { Object.assign(UI, JSON.parse(localStorage.getItem('oe_ui') || '{}')); Object.assign(OV, UI.ov || {}); } catch (e) { /* ignore */ }
function openDrawer(id) {
  if (UI.drawer === id || !id) id = null;
  UI.drawer = id;
  for (const p of document.querySelectorAll('.dpane')) p.classList.toggle('on', p.id === id);
  for (const b of document.querySelectorAll('#rail [data-drawer]')) b.classList.toggle('on', b.dataset.drawer === id);
  const dr = $('drawer'); dr.hidden = !id;
  if (id) {
    const pane = $(id); dr.classList.toggle('wide', pane.classList.contains('wide')); $('drawerTitle').textContent = pane.dataset.title;
    resultsDirty = true;                         // charts can only be drawn while visible
  }
  document.body.classList.toggle('drawer-open', !!id);
  document.body.style.setProperty('--dwOpen', id ? dr.getBoundingClientRect().width + 'px' : '0px');
  saveUI();
}
for (const b of document.querySelectorAll('#rail [data-drawer]')) b.onclick = () => openDrawer(b.dataset.drawer);
$('drawerClose').onclick = () => openDrawer(null);
function applyUI() {
  $('hudWrap').style.display = UI.hud ? '' : 'none'; $('showHud').checked = UI.hud;
  hudMore = UI.hudMore; $('hudWrap').classList.toggle('more', hudMore); $('hudMore').textContent = hudMore ? 'less ▴' : 'more ▾';
  $('events').classList.toggle('collapsed', !UI.events); $('events').style.display = UI.showEvents === false ? 'none' : ''; $('showEvents').checked = UI.showEvents !== false;
  $('legend').hidden = !UI.legend; $('showLegend').checked = UI.legend; $('chipLegend').classList.toggle('on', UI.legend);
  $('notes').classList.toggle('collapsed', !UI.notesOpen); $('showNotes').checked = UI.notes;
  for (const [id, k] of [['ovMarsOrbit', 'marsOrbit'], ['ovMarsSafe', 'marsSafe'], ['ovCoast', 'coast'], ['ovAhead', 'ahead'], ['ovLabels', 'labels'], ['ovPackets', 'packets'], ['ovCollectors', 'collectors'], ['ovBeams', 'beams'], ['ovTracked', 'tracked'], ['ovMirrors', 'mirrors'], ['ovArms', 'arms'], ['ovPlanets', 'planets'], ['ovPlanetOrbits', 'planetOrbits']]) $(id).checked = OV[k] !== false;
  $('showPower').checked = UI.power !== false; $('power').classList.toggle('collapsed', !!UI.powerCollapsed);
}
$('hudMore').onclick = () => { UI.hudMore = !UI.hudMore; applyUI(); saveUI(); };
$('hudHide').onclick = () => { UI.hud = false; applyUI(); saveUI(); };
$('showHud').onchange = (e) => { UI.hud = e.target.checked; applyUI(); saveUI(); };
$('evHead').onclick = () => { UI.events = !UI.events; applyUI(); saveUI(); };
$('showEvents').onchange = (e) => { UI.showEvents = e.target.checked; applyUI(); saveUI(); };
$('notesHead').onclick = () => { UI.notesOpen = !UI.notesOpen; applyUI(); saveUI(); };
$('showNotes').onchange = (e) => { UI.notes = e.target.checked; applyUI(); saveUI(); };
const toggleLegend = () => { UI.legend = !UI.legend; applyUI(); saveUI(); };
$('chipLegend').onclick = toggleLegend; $('showLegend').onchange = toggleLegend;
for (const [id, k] of [['ovMarsOrbit', 'marsOrbit'], ['ovMarsSafe', 'marsSafe'], ['ovCoast', 'coast'], ['ovAhead', 'ahead'], ['ovLabels', 'labels'], ['ovPackets', 'packets'], ['ovCollectors', 'collectors'], ['ovBeams', 'beams'], ['ovTracked', 'tracked'], ['ovMirrors', 'mirrors'], ['ovArms', 'arms'], ['ovPlanets', 'planets'], ['ovPlanetOrbits', 'planetOrbits']])
  $(id).onchange = (e) => { OV[k] = e.target.checked; if (k === 'marsOrbit') marsOrb0.userData.key = null; saveUI(); };
function toggleView() { $('viewMode').value = closeUp() ? 'overview' : 'pov'; $('viewMode').dispatchEvent(new Event('change')); }
function toggleWx() { $('viewMode').value = wxOn() ? 'overview' : 'weather'; $('viewMode').dispatchEvent(new Event('change')); }
function toggleBeam() { $('viewMode').value = beamOn() ? 'pov' : 'beam'; $('viewMode').dispatchEvent(new Event('change')); }
function toggleCol() { $('viewMode').value = colOn() ? 'overview' : 'collector'; $('viewMode').dispatchEvent(new Event('change')); }
$('chipView').onclick = toggleView; $('chipBeam').onclick = toggleBeam; $('chipCol').onclick = toggleCol; $('chipWx').onclick = toggleWx;
$('wxReset').onclick = () => wx.lookDefault();
$('wxLabels').onchange = (e) => { wx.P.labels = e.target.checked; };
$('ovZoomSystem').onclick = () => {        // the whole modelled solar system: Sun to Jupiter (5.2 AU)
  $('follow').checked = false; controls.target.set(0, 0, 0); camera.position.set(0, -92 * S / 10, 80 * S / 10); controls.update();
};
$('ovZoomEarth').onclick = () => {          // jump close to Earth so the ring, the arms and the mirrors can be seen
  controls.target.copy(earth.position); camera.position.copy(earth.position).add(new THREE.Vector3(2.3, -3.3, 1.9));
  $('follow').checked = true; controls.update();
};
$('colReset').onclick = () => col.lookDefault();
$('colLabels').onchange = (e) => { col.P.labels = e.target.checked; };
$('showPower').onchange = (e) => { UI.power = e.target.checked; saveUI(); };
$('powerHead').onclick = () => { UI.powerCollapsed = !UI.powerCollapsed; applyUI(); saveUI(); };
$('btnWatch').onclick = () => startRun();
function showQuick(on) { $('quick').hidden = !on; if (!on) { UI.qs = true; saveUI(); } }
$('btnHelp').onclick = () => showQuick($('quick').hidden);
$('quickClose').onclick = $('quickOk').onclick = () => showQuick(false);
$('quickHeur').onclick = () => { $('playPolicy').value = 'heuristic'; showQuick(false); startRun(); };
$('speed').addEventListener('change', saveUI);
window.addEventListener('keydown', (e) => {
  const tag = (e.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'select' || tag === 'textarea' || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === 'Escape') { if (!$('quick').hidden) showQuick(false); else openDrawer(null); }
  else if (e.key === ' ') { e.preventDefault(); if ((V.mode === 'playback' || V.mode === 'training') && V.playing) V.playing = false; else $('btnPlay').click(); }
  else if (e.key === 'r' || e.key === 'R') $('btnReset').click();
  else if (e.key === 'v' || e.key === 'V') toggleView();
  else if (e.key === 'b' || e.key === 'B') toggleBeam();
  else if (e.key === 'c' || e.key === 'C') toggleCol();
  else if (e.key === 'w' || e.key === 'W') toggleWx();
  else if (e.key === 'l' || e.key === 'L') toggleLegend();
  else if (e.key === '?') showQuick($('quick').hidden);
});
$('speed').value = UI.speed; updSpeed();
applyUI();
if (UI.view === 'pov' || UI.view === 'beam' || UI.view === 'collector' || UI.view === 'weather') { $('viewMode').value = UI.view; $('viewMode').dispatchEvent(new Event('change')); }
if (UI.drawer && $(UI.drawer)) { const d = UI.drawer; UI.drawer = null; openDrawer(d); }
if (!UI.qs) showQuick(true);

window.__oe = { glGpu: GL_GPU, fps: () => FPS, V, pov, asteroid, earth, wx, UI, OV, openDrawer, camera, controls, OVX, renderer, get ovx() { return OVX_STATE; }, get speed() { return speedY; }, get frameMs() { return FRAME_MS; }, earthPos: () => [earth.position.x, earth.position.y].map((x) => +x.toFixed(2)) };
pollState(); pollResults();
window.addEventListener('resize', () => { resultsDirty = true; if (UI.drawer) document.body.style.setProperty('--dwOpen', $('drawer').getBoundingClientRect().width + 'px'); });
