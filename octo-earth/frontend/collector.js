// OCTO-EARTH – solar collector satellites (ILLUSTRATION).
// In the physics model each collector is simply an area of solar cells (the belt intercepts the configured
// fraction of the Sun's output; each of the 5 million collectors is ~65,000 km across as a thin film).
// This module draws them as satellites so their parts can be seen: two solar-panel wings that face the Sun,
// a gold-foil bus with radiators and station-keeping thrusters, and a phased-array microwave transmitter that
// faces Earth. Shapes and proportions are illustrative and NOT to scale. What comes from the simulation:
// Earth's position (so the directions to the Sun and to Earth), the belt geometry, and the beam power
// recorded for the current launch event (beam brightness).
import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';

const D2R = Math.PI / 180;
const V3 = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const UPY = V3(0, 1, 0), UPZ = V3(0, 0, 1);

function rnd(i) { let h = Math.imul(i ^ 0x9e3779b9, 0x85ebca6b); h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35); h ^= h >>> 16; return (h >>> 0) / 4294967296; }
function canvasTex(w, h, draw, srgb = true) {
  const c = document.createElement('canvas'); c.width = w; c.height = h; draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c); if (srgb) t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; return t;
}
// ---------------------------------------------------------------- textures
function pvTexture() {          // solar cells: dark blue cells, silver grid and bus bars
  return canvasTex(512, 512, (g, W, H) => {
    g.fillStyle = '#8f98a8'; g.fillRect(0, 0, W, H);
    const n = 8, cw = W / n, chh = H / n;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const x = i * cw + 3, y = j * chh + 3, w = cw - 6, h = chh - 6;
      const gr = g.createLinearGradient(x, y, x + w, y + h);
      const v = rnd(i * 31 + j) * 0.12;
      gr.addColorStop(0, `rgb(${34 + 30 * v},${70 + 50 * v},${170 + 50 * v})`); gr.addColorStop(1, `rgb(${16},${40 + 30 * v},${118 + 40 * v})`);
      g.fillStyle = gr; g.fillRect(x, y, w, h);
      g.strokeStyle = 'rgba(190,200,215,0.35)'; g.lineWidth = 1;
      for (let k = 1; k < 3; k++) { g.beginPath(); g.moveTo(x + k * w / 3, y); g.lineTo(x + k * w / 3, y + h); g.stroke(); }
      g.strokeStyle = 'rgba(160,175,200,0.12)';
      for (let k = 1; k < 12; k++) { g.beginPath(); g.moveTo(x, y + k * h / 12); g.lineTo(x + w, y + k * h / 12); g.stroke(); }
    }
  });
}
function backTexture() {
  return canvasTex(256, 256, (g, W, H) => {
    g.fillStyle = '#c9cdd3'; g.fillRect(0, 0, W, H);
    g.strokeStyle = '#9aa0a8'; g.lineWidth = 3;
    for (let k = 0; k <= 4; k++) { g.beginPath(); g.moveTo(k * W / 4, 0); g.lineTo(k * W / 4, H); g.stroke(); g.beginPath(); g.moveTo(0, k * H / 4); g.lineTo(W, k * H / 4); g.stroke(); }
    g.strokeStyle = 'rgba(120,125,135,0.5)'; g.lineWidth = 1.5; g.beginPath(); g.moveTo(0, 0); g.lineTo(W, H); g.moveTo(W, 0); g.lineTo(0, H); g.stroke();
  });
}
function goldTexture() {         // crinkled multi-layer insulation
  return canvasTex(256, 256, (g, W, H) => {
    g.fillStyle = '#c8952f'; g.fillRect(0, 0, W, H);
    for (let i = 0; i < 420; i++) {
      const x = rnd(i) * W, y = rnd(i + 999) * H, l = 8 + 30 * rnd(i + 77), a = rnd(i + 5) * Math.PI;
      const b = rnd(i + 3);
      g.strokeStyle = b > 0.5 ? `rgba(255,226,150,${0.25 + 0.3 * rnd(i + 11)})` : `rgba(110,70,10,${0.2 + 0.3 * rnd(i + 13)})`;
      g.lineWidth = 1 + 2 * rnd(i + 17); g.beginPath(); g.moveTo(x, y); g.lineTo(x + l * Math.cos(a), y + l * Math.sin(a)); g.stroke();
    }
  });
}
function arrayTexture(emissive) {  // phased-array transmitter: hexagonal elements
  return canvasTex(512, 512, (g, W, H) => {
    g.fillStyle = emissive ? '#000' : '#2a1c10'; g.fillRect(0, 0, W, H);
    const r = 13, dx = r * Math.sqrt(3), dy = r * 1.5;
    for (let row = 0, y = 0; y < H + r; row++, y += dy) for (let x = (row % 2) * dx / 2; x < W + r; x += dx) {
      const d = Math.hypot(x - W / 2, y - H / 2) / (W / 2); if (d > 1.0) continue;
      g.beginPath(); for (let k = 0; k < 6; k++) { const a = (k * 60 + 30) * D2R; g.lineTo(x + (r - 2) * Math.cos(a), y + (r - 2) * Math.sin(a)); } g.closePath();
      if (emissive) { const v = 0.55 + 0.45 * Math.cos(d * 9); g.fillStyle = `rgba(${150 * v},${100 * v},${255 * v},1)`; }
      else { const v = rnd(Math.round(x * 7 + y * 13)) * 0.25; g.fillStyle = `rgb(${176 + 40 * v},${122 + 30 * v},${58 + 20 * v})`; }
      g.fill();
    }
  });
}
function radiatorTexture() {
  return canvasTex(128, 256, (g, W, H) => {
    g.fillStyle = '#eef1f4'; g.fillRect(0, 0, W, H);
    g.fillStyle = '#b8c0c8'; for (let k = 0; k < 10; k++) g.fillRect(k * W / 10 + W / 20 - 1, 0, 2, H);
  });
}
export function glowTex(inner = 'rgba(255,255,255,1)', outer = 'rgba(255,255,255,0)') {
  return canvasTex(128, 128, (g) => {
    const gr = g.createRadialGradient(64, 64, 0, 64, 64, 64); gr.addColorStop(0, inner); gr.addColorStop(0.25, inner.replace(/[\d.]+\)$/, '0.55)')); gr.addColorStop(1, outer);
    g.fillStyle = gr; g.fillRect(0, 0, 128, 128);
  });
}
function rayTex() {               // soft star-burst around the Sun
  return canvasTex(256, 256, (g) => {
    g.translate(128, 128);
    for (let k = 0; k < 12; k++) {
      g.rotate(Math.PI / 6); const gr = g.createLinearGradient(0, 0, 128, 0); gr.addColorStop(0, 'rgba(255,240,200,0.55)'); gr.addColorStop(1, 'rgba(255,200,120,0)');
      g.fillStyle = gr; g.beginPath(); g.moveTo(0, -2 - (k % 2) * 2); g.lineTo(128, 0); g.lineTo(0, 2 + (k % 2) * 2); g.fill();
    }
  });
}

// ---------------------------------------------------------------- simplified satellite (one geometry, vertex colours)
// local frame: wings along +-Y, panel fronts face +Z (toward the Sun); transmitter disc faces +X.
export function satelliteGeometry(neutral = false) {
  // neutral = true: no colours (silver-grey cells, light-grey bus) - used in the solar-system overview
  const parts = [];
  const add = (geo, col, tf) => { const g = geo.toNonIndexed(); if (tf) g.applyMatrix4(tf); const n = g.attributes.position.count, c = new Float32Array(n * 3); for (let i = 0; i < n; i++) c.set(col, 3 * i); g.setAttribute('color', new THREE.BufferAttribute(c, 3)); parts.push(g); };
  const M = (x, y, z, rx = 0, ry = 0, rz = 0) => new THREE.Matrix4().compose(V3(x, y, z), new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz)), V3(1, 1, 1));
  for (const sgn of [1, -1]) {
    add(new THREE.PlaneGeometry(1.0, 2.6), neutral ? [0.42, 0.43, 0.45] : [0.16, 0.32, 0.78], M(0, sgn * 1.75, 0.012));                 // cells (front, sunward)
    add(new THREE.PlaneGeometry(1.0, 2.6), [0.78, 0.8, 0.84], M(0, sgn * 1.75, -0.012, 0, Math.PI, 0)); // white back
  }
  add(new THREE.BoxGeometry(0.06, 3.6, 0.06), [0.6, 0.62, 0.66]);                                         // boom
  add(new THREE.BoxGeometry(0.42, 0.42, 0.42), neutral ? [0.72, 0.72, 0.72] : [0.85, 0.62, 0.2]);                                        // gold bus
  add(new THREE.CylinderGeometry(0.42, 0.42, 0.05, 20), neutral ? [0.8, 0.8, 0.8] : [0.85, 0.6, 0.35], M(0.36, 0, 0, 0, 0, Math.PI / 2)); // transmitter disc
  const n = parts.reduce((s, g) => s + g.attributes.position.count, 0);
  const pos = new Float32Array(n * 3), nor = new Float32Array(n * 3), col = new Float32Array(n * 3); let o = 0;
  for (const g of parts) { pos.set(g.attributes.position.array, o * 3); nor.set(g.attributes.normal.array, o * 3); col.set(g.attributes.color.array, o * 3); o += g.attributes.position.count; }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.BufferAttribute(pos, 3)); out.setAttribute('normal', new THREE.BufferAttribute(nor, 3)); out.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return out;
}
// orientation of a satellite: panels (+Z) toward the Sun, wings (+Y) toward ecliptic north, transmitter (+X) roughly toward Earth
const _bx = V3(), _by = V3(), _bz = V3(), _bm = new THREE.Matrix4();
export function satQuat(toSun, north, out) {
  _bz.copy(toSun).normalize(); _by.copy(north).addScaledVector(_bz, -north.dot(_bz)); if (_by.lengthSq() < 1e-8) _by.set(0, 0, 1).addScaledVector(_bz, -_bz.z); _by.normalize();
  _bx.crossVectors(_by, _bz); _bm.makeBasis(_bx, _by, _bz); return out.setFromRotationMatrix(_bm);
}

// ---------------------------------------------------------------- close-up view of one collector
export function createCollectorView(renderer, canvas, container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x02030a);
  scene.fog = new THREE.FogExp2(0x02030a, 0.0024);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 20000); camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true; controls.minDistance = 5; controls.maxDistance = 900; controls.enabled = false;
  const P = { active: false, beamI: 0, init: false, sunDir: V3(-1, 0, 0), earthDir: V3(1, 0, 0), earthDistAU: 1, labels: true, collLon: 0 };

  scene.add(new THREE.AmbientLight(0x8090b0, 0.35));
  const sunLight = new THREE.DirectionalLight(0xfff4e0, 3.2); scene.add(sunLight); scene.add(sunLight.target);
  const fill = new THREE.DirectionalLight(0x6a7cff, 0.25); scene.add(fill);       // faint violet fill from the beams of the neighbours
  { // stars
    const n = 2500, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { const z = rnd(i) * 2 - 1, ph = rnd(i + 5000) * 2 * Math.PI, r = Math.sqrt(1 - z * z); pos.set([r * Math.cos(ph) * 9000, r * Math.sin(ph) * 9000, z * 9000], 3 * i); }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const st = new THREE.Points(g, new THREE.PointsMaterial({ color: 0x9aa4bb, size: 1.4, sizeAttenuation: false, fog: false, transparent: true, opacity: 0.7 }));
    st.frustumCulled = false; scene.add(st); P.stars = st;
  }
  // Sun (1.07 deg wide from 0.5 AU) and Earth (a blue point)
  const sunG = new THREE.Group(); scene.add(sunG);
  const sunCore = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16), new THREE.MeshBasicMaterial({ color: 0xfff4d0, fog: false }));
  const sunGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex('rgba(255,225,150,1)', 'rgba(255,150,40,0)'), blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  const sunRays = new THREE.Sprite(new THREE.SpriteMaterial({ map: rayTex(), blending: THREE.AdditiveBlending, depthWrite: false, fog: false, transparent: true, opacity: 0.8 }));
  sunG.add(sunCore); sunG.add(sunGlow); sunG.add(sunRays);
  const earthDot = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex('rgba(120,180,255,1)', 'rgba(60,120,255,0)'), blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  scene.add(earthDot);

  // ---- the satellite
  const sat = new THREE.Group(); scene.add(sat);
  const pvTex = pvTexture(), backTex = backTexture(), goldTex = goldTexture();
  goldTex.wrapS = goldTex.wrapT = THREE.RepeatWrapping; goldTex.repeat.set(2, 1);
  const pvMat = new THREE.MeshStandardMaterial({ map: pvTex, metalness: 0.05, roughness: 0.38, color: 0xffffff, emissive: 0x3050c0, emissiveIntensity: 0.6, emissiveMap: pvTex });
  const backMat = new THREE.MeshStandardMaterial({ map: backTex, metalness: 0.1, roughness: 0.7 });
  const frameMat = new THREE.MeshStandardMaterial({ color: 0x9aa2ad, metalness: 0.8, roughness: 0.35 });
  const goldMat = new THREE.MeshStandardMaterial({ map: goldTex, color: 0xffe0a0, metalness: 0.45, roughness: 0.38, emissive: 0x6a4010, emissiveIntensity: 0.55, emissiveMap: goldTex });
  const darkMat = new THREE.MeshStandardMaterial({ color: 0x2c3038, metalness: 0.7, roughness: 0.45 });
  const whiteMat = new THREE.MeshStandardMaterial({ map: radiatorTexture(), color: 0xffffff, metalness: 0.05, roughness: 0.55 });
  // bus: hexagonal prism, gold foil
  const bus = new THREE.Mesh(new THREE.CylinderGeometry(0.95, 0.95, 1.7, 6), goldMat); bus.rotation.x = Math.PI / 2; sat.add(bus);
  const capT = new THREE.Mesh(new THREE.CylinderGeometry(0.97, 0.97, 0.08, 6), darkMat); capT.position.z = 0.88; capT.rotation.x = Math.PI / 2; sat.add(capT);
  const capB = capT.clone(); capB.position.z = -0.88; sat.add(capB);
  // radiators (heat rejection) on the two sides away from the panels
  for (const sx of [1, -1]) { const r = new THREE.Mesh(new THREE.BoxGeometry(0.04, 1.2, 1.5), whiteMat); r.position.set(sx * 1.02, 0, 0); sat.add(r); }
  // solar wings: boom + yoke + 2 x 6 panels per side (fronts face +Z = the Sun)
  const PW = 1.45, PH = 1.15, GAP = 0.06, NJ = 6;
  const panelGeo = new THREE.BoxGeometry(PW, PH, 0.035);
  for (const sy of [1, -1]) {
    const boom = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 1.3, 10), frameMat); boom.position.set(0, sy * 1.45, 0); sat.add(boom);
    const yoke = new THREE.Mesh(new THREE.BoxGeometry(2 * PW + GAP + 0.1, 0.08, 0.08), frameMat); yoke.position.set(0, sy * 2.1, 0); sat.add(yoke);
    const spine = new THREE.Mesh(new THREE.BoxGeometry(0.06, NJ * (PH + GAP), 0.07), frameMat); spine.position.set(0, sy * (2.15 + NJ * (PH + GAP) / 2), -0.03); sat.add(spine);
    for (let j = 0; j < NJ; j++) for (const sx of [-1, 1]) {
      const pnl = new THREE.Mesh(panelGeo, [frameMat, frameMat, frameMat, frameMat, pvMat, backMat]);
      pnl.position.set(sx * (PW + GAP) / 2, sy * (2.15 + (j + 0.5) * (PH + GAP)), 0); sat.add(pnl);
    }
  }
  // station-keeping thrusters (4 nozzles) and their occasional puffs
  const nozGeo = new THREE.ConeGeometry(0.12, 0.3, 14, 1, true);
  const thr = [];
  for (let k = 0; k < 4; k++) {
    const a = (45 + 90 * k) * D2R, d = V3(Math.cos(a), Math.sin(a), -0.6).normalize();
    const nz = new THREE.Mesh(nozGeo, darkMat); nz.position.set(0.7 * Math.cos(a), 0.7 * Math.sin(a), -0.95); nz.quaternion.setFromUnitVectors(V3(0, -1, 0), d); sat.add(nz);
    const puff = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex('rgba(170,210,255,1)', 'rgba(120,170,255,0)'), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
    puff.position.copy(nz.position).addScaledVector(d, 0.35); sat.add(puff); thr.push({ puff, d, base: puff.position.clone() });
  }
  // small comm dish + star trackers
  const comm = new THREE.Mesh(new THREE.SphereGeometry(0.42, 24, 12, 0, Math.PI * 2, 0, 0.9), new THREE.MeshStandardMaterial({ color: 0xe8e8ea, metalness: 0.3, roughness: 0.4, side: THREE.DoubleSide }));
  comm.position.set(0.3, 0, 1.45); comm.rotation.x = Math.PI; sat.add(comm);
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.55, 8), frameMat); mast.position.set(0.3, 0, 1.15); mast.rotation.x = Math.PI / 2; sat.add(mast);
  for (const sx of [-1, 1]) { const st = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.18, 0.26), darkMat); st.position.set(sx * 0.5, 0.55, 1.0); sat.add(st); }
  // nav lights
  const navPos = new Float32Array(6), navCol = new Float32Array([1, 0.15, 0.1, 0.2, 1, 0.3]);
  const navGeo = new THREE.BufferGeometry(); navGeo.setAttribute('position', new THREE.BufferAttribute(navPos, 3)); navGeo.setAttribute('color', new THREE.BufferAttribute(navCol, 3));
  const nav = new THREE.Points(navGeo, new THREE.PointsMaterial({ size: 9, sizeAttenuation: false, vertexColors: true, map: glowTex(), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  sat.add(nav);
  // microwave transmitter: phased array on a gimbal boom, faces Earth (oriented in world space each frame)
  const tx = new THREE.Group(); scene.add(tx);
  const txFace = new THREE.Mesh(new THREE.CylinderGeometry(2.1, 2.1, 0.14, 64), [
    new THREE.MeshStandardMaterial({ color: 0x5a5f68, metalness: 0.8, roughness: 0.4 }),
    new THREE.MeshStandardMaterial({ map: arrayTexture(false), emissiveMap: arrayTexture(true), emissive: 0xffffff, emissiveIntensity: 0.2, metalness: 0.4, roughness: 0.4 }),
    new THREE.MeshStandardMaterial({ color: 0x3a3f48, metalness: 0.6, roughness: 0.5 })]);
  txFace.rotation.x = Math.PI / 2; tx.add(txFace);                  // disc axis = local +Z (toward Earth)
  const txRing = new THREE.Mesh(new THREE.TorusGeometry(2.15, 0.07, 10, 64), frameMat); tx.add(txRing);
  for (let k = 0; k < 4; k++) { const a = k * Math.PI / 2, st = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 2.4, 8), frameMat); st.position.set(1.0 * Math.cos(a), 1.0 * Math.sin(a), -0.8); st.lookAt(0, 0, -1.9); st.rotateX(Math.PI / 2); tx.add(st); }
  const txHub = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.32, 0.5, 16), darkMat); txHub.rotation.x = Math.PI / 2; txHub.position.z = -1.6; tx.add(txHub);
  const txBoom = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 1, 10), frameMat); scene.add(txBoom);
  // the beam: true direction to Earth; drawn as a glowing column with wavefronts moving toward Earth
  const beamG = new THREE.Group(); scene.add(beamG);
  const BL = 1200, NW = 40;
  const beamCore = new THREE.Mesh(new THREE.CylinderGeometry(1.8, 2.0, BL, 32, 1, true), new THREE.MeshBasicMaterial({ color: 0xc8a8ff, transparent: true, opacity: 0.25, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide, fog: false }));
  const beamHalo = new THREE.Mesh(new THREE.CylinderGeometry(3.2, 3.8, BL, 32, 1, true), new THREE.MeshBasicMaterial({ color: 0x7a4cff, transparent: true, opacity: 0.07, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide, fog: false }));
  beamG.add(beamCore); beamG.add(beamHalo);
  const waves = new THREE.InstancedMesh(new THREE.TorusGeometry(1, 0.05, 6, 48), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false }), NW);
  waves.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(NW * 3), 3); waves.frustumCulled = false; beamG.add(waves);
  const aperture = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex('rgba(210,180,255,1)', 'rgba(140,90,255,0)'), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  scene.add(aperture);
  // neighbours: thousands more collectors in the belt (simplified model, drawn far closer together than real)
  const NN = 1400, nbGeo = satelliteGeometry();
  const nb = new THREE.InstancedMesh(nbGeo, new THREE.MeshStandardMaterial({ vertexColors: true, metalness: 0.45, roughness: 0.4, emissive: 0x15102a, emissiveIntensity: 0.6 }), NN);
  nb.frustumCulled = false; scene.add(nb);
  const nbBeams = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xa080ff, transparent: true, opacity: 0.25, blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  nbBeams.frustumCulled = false; scene.add(nbBeams);

  // ---- labels
  const LBL = {}, LCOL = { panels: '#9cc0ff', tx: '#d8c0ff', bus: '#ffd27a', rad: '#e0e6ee', thr: '#aad2ff', sun: '#ffcc33', earth: '#7cc4ff', nb: '#c8b8e8', beam: '#c8a8ff' };
  for (const [k, c] of Object.entries(LCOL)) { const d = document.createElement('div'); d.className = 'blabel mono small povlbl'; d.style.color = c; d.style.display = 'none'; container.appendChild(d); LBL[k] = d; }
  const _lv = V3();
  function label(k, pos, text) {
    const el = LBL[k];
    if (!P.labels || !pos || !text) { el.style.display = 'none'; return; }
    _lv.copy(pos).project(camera);
    if (_lv.z > 1 || Math.abs(_lv.x) > 1 || Math.abs(_lv.y) > 1) { el.style.display = 'none'; return; }
    if (el.textContent !== text) el.textContent = text; el.style.display = 'block';
    const w = canvas.clientWidth, h = canvas.clientHeight, lw = el.offsetWidth, lh = el.offsetHeight;
    el.style.left = `${Math.max(2, Math.min(w - lw - 2, (_lv.x * 0.5 + 0.5) * w + 8))}px`; el.style.top = `${Math.max(2, Math.min(h - lh - 2, (-_lv.y * 0.5 + 0.5) * h - 6))}px`;
  }
  function resolveLabels() {
    const placed = [];
    for (const el of Object.values(LBL)) {
      if (el.style.display === 'none') continue;
      const x = parseFloat(el.style.left); let y = parseFloat(el.style.top); const w = el.offsetWidth, h = el.offsetHeight;
      for (let it = 0; it < 8; it++) { const hit = placed.find((b) => x < b.x + b.w && x + w > b.x && y < b.y + b.h && y + h > b.y); if (!hit) break; y = hit.y + hit.h + 1; }
      el.style.top = `${y}px`; placed.push({ x, y, w, h });
    }
  }
  function hideLabels() { for (const el of Object.values(LBL)) el.style.display = 'none'; }

  // ---- geometry of the chosen collector (heliocentric, AU): 50 deg ahead of Earth in the belt
  const _q = new THREE.Quaternion(), _m = new THREE.Matrix4(), _s = V3(), _a = V3(), _b = V3();
  let layoutKey = '';
  function layout(p, sol) {
    const lonE = Math.atan2(p[1], p[0]), lon = lonE + 50 * D2R, R = sol.orbit_au;
    const c = V3(R * Math.cos(lon), R * Math.sin(lon), 0);
    P.sunDir.copy(c).negate().normalize();
    _a.set(p[0] - c.x, p[1] - c.y, p[2] - c.z); P.earthDistAU = _a.length(); P.earthDir.copy(_a).normalize();
    satQuat(P.sunDir, UPZ, _q); sat.quaternion.copy(_q);
    sunG.position.copy(P.sunDir).multiplyScalar(6000); sunCore.scale.setScalar(6000 * Math.tan(0.5 * 1.0656 * D2R * (0.5 / R))); sunGlow.scale.setScalar(sunCore.scale.x * 9); sunRays.scale.setScalar(sunCore.scale.x * 16);
    sunLight.position.copy(P.sunDir).multiplyScalar(100); fill.position.copy(P.earthDir).multiplyScalar(-50).addScaledVector(UPZ, 30);
    earthDot.position.copy(P.earthDir).multiplyScalar(7000); earthDot.scale.setScalar(70);
    // transmitter: on a boom on the Earth-facing side of the bus, square to the beam
    const off = _b.copy(P.earthDir).addScaledVector(UPZ, -P.earthDir.dot(UPZ)).normalize();
    const txPos = V3().copy(off).multiplyScalar(3.4).addScaledVector(UPZ, -0.3);
    tx.position.copy(txPos); tx.quaternion.setFromUnitVectors(UPZ, P.earthDir);
    const bm = V3().copy(txPos).multiplyScalar(0.5), bl = txPos.length() - 1.1; txBoom.position.copy(off).multiplyScalar(1.1 + bl / 2).addScaledVector(UPZ, -0.15); txBoom.quaternion.setFromUnitVectors(UPY, off); txBoom.scale.set(1, bl, 1);
    beamG.position.copy(txPos).addScaledVector(P.earthDir, BL / 2 + 0.1); beamCore.quaternion.setFromUnitVectors(UPY, P.earthDir); beamHalo.quaternion.copy(beamCore.quaternion);
    aperture.position.copy(txPos).addScaledVector(P.earthDir, 0.15);
    P.txPos = txPos;
    // neighbours along the belt (tangent T, north N), each oriented to its own Sun / Earth directions
    const T = V3(-Math.sin(lon), Math.cos(lon), 0), sp = 46, nbP = [];
    let k = 0;
    for (let i = -30; i <= 30 && k < NN; i++) for (let j = -5; j <= 5 && k < NN; j++) {
      if (i === 0 && j === 0) continue;                                      // the close-up itself
      const pos = V3().copy(T).multiplyScalar(i * sp + (rnd(i * 41 + j) - 0.5) * 25).addScaledVector(UPZ, j * sp * 1.1 + (rnd(i * 7 + j * 3) - 0.5) * 25)
        .addScaledVector(P.sunDir, -110 + (rnd(i + j * 101) - 0.5) * 40 - 0.0003 * (i * sp) ** 2);   // behind the close-up; the belt curves around the Sun
      satQuat(P.sunDir, UPZ, _q); _s.setScalar(3.6 + 0.5 * rnd(i * 13 + j)); _m.compose(pos, _q, _s); nb.setMatrixAt(k, _m); nbP.push(pos); k++;
    }
    nb.count = k; nb.instanceMatrix.needsUpdate = true;
    const bp = new Float32Array(Math.ceil(k / 5) * 6);
    for (let n = 0; n < k; n += 5) { const q = nbP[n], o = n / 5; bp.set([q.x, q.y, q.z, q.x + P.earthDir.x * 4000, q.y + P.earthDir.y * 4000, q.z + P.earthDir.z * 4000], 6 * o); }
    nbBeams.geometry.setAttribute('position', new THREE.BufferAttribute(bp, 3)); nbBeams.geometry.computeBoundingSphere();
    P.nbLabelPos = nbP.reduce((b, q) => (q.length() < 190 && q.dot(UPZ) > 25 && (!b || q.length() < b.length()) ? q : b), null) || nbP[0];
  }
  function lookDefault() {
    // from the sunlit side (panel fronts visible), a little above, off to the side opposite Earth: the beam runs away toward Earth
    const w = V3().copy(P.earthDir).addScaledVector(P.sunDir, -P.earthDir.dot(P.sunDir)).normalize();
    camera.position.copy(P.sunDir).multiplyScalar(0.8).addScaledVector(w, 0.3).addScaledVector(UPZ, 0.4).normalize().multiplyScalar(40);
    controls.target.set(0, 0, 0); controls.update();
  }

  function update(S, dt, wall) {
    const sol = S.solar; if (!sol || !sol.on) return;
    // re-layout when Earth has moved by > 0.5 deg around the Sun or 0.01 AU in distance; the camera turns with the
    // satellite's frame so the view of the collector stays the same while the directions to Sun and Earth change
    const key = `${Math.round(Math.atan2(S.p[1], S.p[0]) / D2R * 2)}|${Math.hypot(...S.p).toFixed(2)}|${S.p[2].toFixed(2)}|${sol.orbit_au}`;
    if (key !== layoutKey) {
      const q0 = sat.quaternion.clone(); layoutKey = key; layout(S.p, sol);
      if (P.init) { const dq = sat.quaternion.clone().multiply(q0.invert()); camera.position.sub(controls.target).applyQuaternion(dq).add(controls.target.applyQuaternion(dq)); }
    }
    if (!P.init) { P.init = true; lookDefault(); }
    const pw = S.power || { on: false };
    const target = !pw.on ? 0.3 : pw.inW > 0 ? 1 : 0.35;
    P.beamI += (target - P.beamI) * Math.min(1, dt * 3); const I = P.beamI;
    P.power = pw; P.solar = sol;
    beamCore.material.opacity = 0.08 + 0.3 * I; beamHalo.material.opacity = 0.02 + 0.08 * I;
    aperture.scale.setScalar(4.5 + 3 * I); aperture.material.opacity = 0.35 + 0.65 * I;
    txFace.material[1].emissiveIntensity = 0.15 + 1.4 * I * (0.85 + 0.15 * Math.sin(wall * 6));
    nbBeams.material.opacity = 0.03 + 0.12 * I;
    // wavefronts leave the transmitter toward Earth
    const col = new THREE.Color(), base = new THREE.Color(0xd8c0ff);
    for (let j = 0; j < NW; j++) {
      const u = (j / NW + wall * 0.25) % 1, d = 0.4 + u * 120;
      _a.copy(P.txPos).addScaledVector(P.earthDir, d); _q.setFromUnitVectors(UPZ, P.earthDir); _s.setScalar(2.0 + d * 0.006); _m.compose(_a, _q, _s); waves.setMatrixAt(j, _m);
      col.copy(base).multiplyScalar(I * Math.min(1, d / 2) * (1 - u) ** 1.5); waves.setColorAt(j, col);
    }
    waves.instanceMatrix.needsUpdate = true; waves.instanceColor.needsUpdate = true;
    // thruster puffs (station keeping, illustrative) and blinking nav lights
    for (let k = 0; k < thr.length; k++) {
      const ph = ((wall + k * 1.7) % 7) / 7, on = ph < 0.06 ? Math.sin(ph / 0.06 * Math.PI) : 0;
      thr[k].puff.material.opacity = 0.9 * on; thr[k].puff.scale.setScalar(0.3 + 1.6 * on); thr[k].puff.position.copy(thr[k].base).addScaledVector(thr[k].d, 0.6 * on);
    }
    const bl = (Math.sin(wall * 4) > 0.6) ? 1 : 0.08;
    navPos.set([0, 2.15 + NJ * (PH + GAP), 0.05, 0, -2.15 - NJ * (PH + GAP), 0.05]); navGeo.attributes.position.needsUpdate = true; nav.material.opacity = bl;
    P.stars.rotation.z += dt * 0.002;
  }
  function placeLabels() {
    if (!P.labels) { hideLabels(); return; }
    const sol = P.solar; if (!sol) { hideLabels(); return; }
    const W = (v) => v.clone().applyQuaternion(sat.quaternion);
    label('panels', W(V3(PW + 0.2, 2.15 + 4.5 * (PH + GAP), 0.1)), `solar wing · faces the Sun · cells ${Math.round(sol.pv * 100)} %`);
    label('bus', W(V3(-0.9, -0.6, 0.9)), 'bus (gold insulation foil)');
    label('rad', W(V3(1.05, -0.3, -0.6)), 'radiator (sheds waste heat)');
    label('thr', W(V3(-0.8, 0.8, -1.2)), 'station-keeping thrusters');
    label('tx', V3().copy(P.txPos).addScaledVector(UPZ, 2.4), `microwave transmitter → Earth · ${sol.freq_ghz} GHz`);
    label('beam', V3().copy(P.txPos).addScaledVector(P.earthDir, 34).addScaledVector(UPZ, 3.5), P.power && P.power.inW > 0 ? 'beam to Earth\'s arms – power flowing' : 'beam to Earth\'s arms – stand-by (no launch right now)');
    label('earth', earthDot.position, `Earth + 8 arm rectennas · ${P.earthDistAU.toFixed(2)} AU away`);
    label('sun', sunG.position, `Sun · ${(1.0656 * 0.5 / sol.orbit_au).toFixed(2)}° wide from ${sol.orbit_au} AU`);
    label('nb', P.nbLabelPos, `more collectors · ${(sol.count / 1e6).toFixed(0)} million in the belt`);
    resolveLabels();
  }
  function render() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    camera.aspect = w / Math.max(1, h);
    const shift = Math.min(200, 0.14 * w); camera.setViewOffset(w, h, shift, 0, w, h);      // keep the model clear of the power panel
    camera.updateProjectionMatrix(); controls.update(); camera.updateMatrixWorld();
    placeLabels();
    renderer.render(scene, camera);
  }
  function setActive(on) { P.active = on; controls.enabled = on; if (!on) hideLabels(); }
  return { scene, camera, controls, P, update, render, setActive, hideLabels, lookDefault };
}
