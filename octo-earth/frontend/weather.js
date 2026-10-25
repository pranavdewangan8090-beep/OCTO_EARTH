// OCTO-EARTH – the "Weather & life" view.
//
// What this view shows, and how much of it is computed:
//   COMPUTED by the Python climate model (recorded for every decision of the mission being watched):
//     * Earth's global-mean surface temperature, the sunlight at Earth's distance and the extra light from the space
//       mirrors (enabler E3), the share of the mirrors' capacity the agent aims at Earth,
//     * the temperature Earth would settle at WITHOUT the mirrors,
//     * the life indicators derived from the temperature (ice line, liquid ocean, growing area). They use the same
//       simple zonal temperature profile as octoearth_rl.life_indicators: equator +30 K, poles -15 K vs the mean.
//   ILLUSTRATED (not simulated):
//     * the continents (procedural, not the real map), clouds, winds, storms and city lights. Their pattern is the
//       well-known three-cell circulation of today's Earth (trade winds, westerlies, polar easterlies, the tropical rain
//       belt, mid-latitude storm tracks). Cloud cover is scaled with the temperature (warmer air holds more water).
//     * the mirror ring is drawn at 2.6 Earth radii so it fits on screen; the real ring orbits at 20 Earth radii.
//
// The view is self-contained: app.js calls update(S, dt, wall) with the recorded climate state and render().
import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import { fbm } from './pov.js';
import { glowTex } from './collector.js';

const V3 = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const D2R = Math.PI / 180;
const SUN_DIR = V3(1, 0, 0);                  // in this view the Sun is always toward +x
const MIRROR_DRAW_R = 2.6;                    // drawn ring radius [Earth radii] (true: 20)
const N_MIRRORS = 30;                         // drawn mirrors (the real ring: ~1 million mirrors of ~40 km)
const N_WIND = 1300;                          // wind tracer particles
const DEFAULT_LIFE = { dT: 45, T_ice: 263, T_grow: [278, 313] };   // same numbers as LIFE_* in octoearth_rl.py

function hash(i, k) { let h = Math.imul(i + 1, 374761393) ^ Math.imul(k + 7, 668265263); h = Math.imul(h ^ (h >>> 13), 1274126177); return ((h ^ (h >>> 16)) >>> 0) / 4294967296; }
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const num = (v, d = 1) => (v !== null && v !== undefined && Number.isFinite(+v)) ? (+v).toFixed(d) : 'N/A';
const esc = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/**
 * Life indicators from the global-mean temperature T [K] – the JavaScript twin of octoearth_rl.life_indicators.
 * Local temperature at latitude lat: T(lat) = T + dT/3 - dT * sin(lat)^2 (its area average is exactly T).
 */
export function lifeFromT(T, L = DEFAULT_LIFE) {
  const dT = L.dT, Tice = L.T_ice, [g0, g1] = L.T_grow;
  const Tat = (x) => T + dT / 3 - dT * x * x;                       // x = sin(latitude)
  const xIce = clamp(Math.sqrt(Math.max(0, (T + dT / 3 - Tice) / dT)), 0, 1);
  let grow = 0; const n = 400;
  for (let k = 0; k < n; k++) { const x = (k + 0.5) / n, t = Tat(x); if (t >= g0 && t <= g1) grow += 1 / n; }
  const state = xIce <= 0 ? 'snowball (frozen to the equator)' : T > 313 ? 'hothouse' : xIce < 0.5 ? 'cold, ice to mid-latitudes'
    : T < 278 ? 'cool, large ice caps' : 'temperate, like today';
  return { T_equator_K: Tat(0), T_pole_K: Tat(1), ice_line_deg: Math.asin(xIce) / D2R, liquid_frac: xIce, grow_frac: grow, state, Tat };
}

// ---------------------------------------------------------------- textures (built on canvases, no downloads)
/** Surface colours for global-mean temperature T: oceans, sea ice, snow, tundra, forest, savanna and desert. */
function buildSurface(T, L, W = 512, H = 256) {
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const g = c.getContext('2d'), img = g.createImageData(W, H), d = img.data;
  const nl = document.createElement('canvas'); nl.width = W; nl.height = H;
  const gn = nl.getContext('2d'), imn = gn.createImageData(W, H), dn = imn.data;
  const life = lifeFromT(T, L), [g0, g1] = L.T_grow;
  for (let j = 0; j < H; j++) {
    const lat = (0.5 - (j + 0.5) / H) * Math.PI, cl = Math.cos(lat), sl = Math.sin(lat);
    const Tzone = life.Tat(sl);
    const desertBelt = Math.exp(-((((Math.abs(lat) / D2R) - 24) / 8) ** 2));          // dry sinking air of the Hadley cells
    for (let i = 0; i < W; i++) {
      const lon = (i + 0.5) / W * 2 * Math.PI, x = cl * Math.cos(lon), z = cl * Math.sin(lon);
      const n = fbm(x * 1.9 + 7, sl * 1.9 + 3, z * 1.9 + 11, 4);                       // same continents as the POV Earth
      const land = n > 0.53, height = land ? (n - 0.53) / 0.2 : 0;
      const Tloc = Tzone - 18 * height + 4 * (fbm(x * 5, sl * 5, z * 5, 2) - 0.5);    // mountains are colder
      let r, gg, b; const o = (j * W + i) * 4;
      if (Tloc < L.T_ice) {                                                             // snow / sea ice
        const k = clamp((L.T_ice - Tloc) / 25, 0, 1);
        r = 205 + 40 * k; gg = 220 + 30 * k; b = 235 + 18 * k; if (!land) { r -= 15; gg -= 8; }
      } else if (!land) {                                                               // liquid ocean, warmer = lighter
        const k = clamp((Tloc - 270) / 35, 0, 1), deep = clamp((0.53 - n) / 0.2, 0, 1);
        r = 12 + 18 * k; gg = 50 + 50 * k - 15 * deep; b = 110 + 45 * k - 20 * deep;
      } else {
        const dry = clamp(0.55 * fbm(x * 3 + 1, sl * 3, z * 3, 3) + 0.6 * desertBelt - 0.1, 0, 1);
        if (Tloc < g0) { r = 118; gg = 112; b = 92; }                                  // tundra: too cold to grow
        else if (Tloc > g1) { r = 214; gg = 176; b = 118; }                            // too hot: desert
        else {
          const wet = 1 - dry, k = clamp(1 - Math.abs(Tloc - 296) / 22, 0, 1);
          r = 40 * wet + 200 * dry; gg = (95 + 40 * k) * wet + 170 * dry; b = 40 * wet + 110 * dry;
        }
        r -= 25 * height; gg -= 25 * height; b -= 15 * height;
        // city lights: only where people can live (growing temperatures) – an illustration of a living planet
        if (Tloc >= g0 && Tloc <= g1 && hash(i, j) < 0.045 * (1 - 0.6 * dry)) {
          const q = 0.6 + 0.4 * hash(j, i); dn[o] = 255 * q; dn[o + 1] = 205 * q; dn[o + 2] = 120 * q; dn[o + 3] = 255;
        }
      }
      d[o] = clamp(r, 0, 255); d[o + 1] = clamp(gg, 0, 255); d[o + 2] = clamp(b, 0, 255); d[o + 3] = 255;
      if (!dn[o + 3]) dn[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0); gn.putImageData(imn, 0, 0);
  const day = new THREE.CanvasTexture(c); day.colorSpace = THREE.SRGBColorSpace;
  const night = new THREE.CanvasTexture(nl); night.colorSpace = THREE.SRGBColorSpace;
  return { day, night };
}
/** Cloud texture for one circulation band: 'tropics' (|lat| < 32) or 'midlat' (poleward of ~28 deg). */
function buildClouds(band, W = 512, H = 256) {
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const g = c.getContext('2d'), img = g.createImageData(W, H), d = img.data;
  for (let j = 0; j < H; j++) {
    const latD = (0.5 - (j + 0.5) / H) * 180, a = Math.abs(latD), cl = Math.cos(latD * D2R), sl = Math.sin(latD * D2R);
    let amount, mask;
    if (band === 'tropics') {
      mask = clamp((32 - a) / 6, 0, 1);
      amount = 0.95 * Math.exp(-(((latD - 5) / 7) ** 2)) + 0.3 * Math.exp(-(((a - 12) / 6) ** 2));   // rain belt (ITCZ)
    } else {
      mask = clamp((a - 26) / 6, 0, 1);
      amount = 0.9 * Math.exp(-(((a - 52) / 11) ** 2)) + 0.35 * clamp((a - 62) / 10, 0, 1);       // storm tracks + polar
    }
    for (let i = 0; i < W; i++) {
      const lon = (i + 0.5) / W * 2 * Math.PI, x = cl * Math.cos(lon), z = cl * Math.sin(lon);
      const n = fbm(x * 3.2 + 3, sl * 7 + 1, z * 3.2 + 5, 5);                   // streaky: stretched along latitude
      const al = clamp((n - 0.62 + 0.42 * amount) * 2.6, 0, 1) * mask;
      const o = (j * W + i) * 4; d[o] = d[o + 1] = d[o + 2] = 255; d[o + 3] = 255 * al;
    }
  }
  g.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
/** A spiral cloud texture for cyclones (transparent outside the spiral arms). */
function spiralTexture() {
  const S = 128, c = document.createElement('canvas'); c.width = c.height = S;
  const g = c.getContext('2d'), img = g.createImageData(S, S), d = img.data;
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const dx = (x - S / 2) / (S / 2), dy = (y - S / 2) / (S / 2), r = Math.hypot(dx, dy), th = Math.atan2(dy, dx);
    const arm = 0.5 + 0.5 * Math.cos(2 * (th - 3.2 * Math.log(r + 0.05)));
    const eye = clamp((r - 0.06) / 0.08, 0, 1), fall = clamp(1 - r, 0, 1);
    const al = clamp(arm * 1.4 - 0.25, 0, 1) * fall * eye + 0.35 * fall * fall * eye;
    const o = (y * S + x) * 4; d[o] = d[o + 1] = d[o + 2] = 255; d[o + 3] = 255 * clamp(al, 0, 1);
  }
  g.putImageData(img, 0, 0); const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

// ---------------------------------------------------------------- the view
export function createWeatherView(renderer, canvas, container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x01020a);
  const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 5000); camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true; controls.enablePan = false; controls.minDistance = 1.25; controls.maxDistance = 40; controls.enabled = false;
  const P = { active: false, labels: true, noMirrors: false, surfKey: '', T: 288, life: lifeFromT(288), S: null, panelWall: 0 };

  // light: the Sun (direction +x) and four lights standing in for the light reflected by the mirror ring
  scene.add(new THREE.AmbientLight(0x8090b0, 0.08));
  const sunLight = new THREE.DirectionalLight(0xfff6e8, 2.4); sunLight.position.copy(SUN_DIR).multiplyScalar(50); scene.add(sunLight);
  const mirrorLights = [V3(0, 1, 0), V3(0, -1, 0), V3(0, 0, 1), V3(0, 0, -1)].map((d) => {
    const l = new THREE.DirectionalLight(0xfff0d8, 0); l.position.copy(d).multiplyScalar(50); scene.add(l); return l;
  });
  { // stars
    const n = 1800, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { const z = hash(i, 1) * 2 - 1, ph = hash(i, 2) * 2 * Math.PI, r = Math.sqrt(1 - z * z); pos.set([r * Math.cos(ph) * 2000, r * Math.sin(ph) * 2000, z * 2000], 3 * i); }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0x8890a8, size: 1.2, sizeAttenuation: false, transparent: true, opacity: 0.6 })));
  }
  // the Sun: a disc with its true angular size at Earth's distance, plus glow
  const sunG = new THREE.Group(); scene.add(sunG); sunG.position.copy(SUN_DIR).multiplyScalar(1500);
  const sunCore = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 12), new THREE.MeshBasicMaterial({ color: 0xfff4d0 })); sunG.add(sunCore);
  const sunGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex('rgba(255,230,160,1)', 'rgba(255,160,40,0)'), blending: THREE.AdditiveBlending, depthWrite: false }));
  sunG.add(sunGlow);

  // Earth: tilted axis, spinning surface, night lights, two cloud bands, cyclones, wind tracers, air glow
  const tilt = new THREE.Group(); scene.add(tilt); tilt.rotation.y = 23.44 * D2R;   // axis tilted (season fixed: illustration)
  const spin = new THREE.Group(); tilt.add(spin);
  const surfMat = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0 });
  const surface = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 48), surfMat); surface.rotation.x = Math.PI / 2; spin.add(surface);
  const nightMat = new THREE.ShaderMaterial({
    uniforms: { map: { value: null }, sunDir: { value: SUN_DIR.clone() }, strength: { value: 1 } },
    vertexShader: 'varying vec2 vUv; varying vec3 vN; void main(){ vUv = uv; vN = normalize(mat3(modelMatrix) * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform sampler2D map; uniform vec3 sunDir; uniform float strength; varying vec2 vUv; varying vec3 vN; void main(){ float night = smoothstep(0.05, -0.25, dot(normalize(vN), sunDir)); vec3 c = texture2D(map, vUv).rgb; gl_FragColor = vec4(c * night * strength, 1.0); }',
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const nightShell = new THREE.Mesh(new THREE.SphereGeometry(1.002, 96, 48), nightMat); nightShell.rotation.x = Math.PI / 2; spin.add(nightShell);
  const cloudTropics = new THREE.Group(), cloudMid = new THREE.Group(); spin.add(cloudTropics); spin.add(cloudMid);
  const cloudMatT = new THREE.MeshStandardMaterial({ map: null, transparent: true, depthWrite: false, roughness: 1, opacity: 0.9 });
  const cloudMatM = cloudMatT.clone();
  const shellT = new THREE.Mesh(new THREE.SphereGeometry(1.014, 96, 48), cloudMatT); shellT.rotation.x = Math.PI / 2; cloudTropics.add(shellT);
  const shellM = new THREE.Mesh(new THREE.SphereGeometry(1.018, 96, 48), cloudMatM); shellM.rotation.x = Math.PI / 2; cloudMid.add(shellM);
  const spiralTex = spiralTexture(), cyclones = [];
  const addCyclone = (parent, latD, lonD, size, nh, tropical) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshStandardMaterial({ map: spiralTex, transparent: true, depthWrite: false, roughness: 1, side: THREE.DoubleSide }));
    const r = tropical ? 1.016 : 1.021, la = latD * D2R, lo = lonD * D2R;
    const p = V3(Math.cos(la) * Math.cos(lo), Math.cos(la) * Math.sin(lo), Math.sin(la));
    m.position.copy(p).multiplyScalar(r); m.lookAt(p.clone().multiplyScalar(2));
    const holder = new THREE.Group(); holder.add(m); parent.add(holder);
    cyclones.push({ m, nh, tropical, spinRate: (nh ? 1 : -1) * (tropical ? 1.6 : 0.8) });   // counter-clockwise in the north
  };
  [[48, 20], [55, 140], [43, 250], [-50, 70], [-56, 200], [-46, 310]].forEach(([la, lo], k) => addCyclone(cloudMid, la, lo, 0.42 + 0.08 * (k % 3), la > 0, false));
  [[16, 100], [-14, 280]].forEach(([la, lo]) => addCyclone(cloudTropics, la, lo, 0.22, la > 0, true));
  // wind tracers: short streaks that follow the three-cell circulation (relative to the spinning surface)
  const wind = { lat: new Float32Array(N_WIND), lon: new Float32Array(N_WIND), age: new Float32Array(N_WIND), life: new Float32Array(N_WIND) };
  const windPos = new Float32Array(N_WIND * 6), windCol = new Float32Array(N_WIND * 6), windGeo = new THREE.BufferGeometry();
  windGeo.setAttribute('position', new THREE.BufferAttribute(windPos, 3)); windGeo.setAttribute('color', new THREE.BufferAttribute(windCol, 3));
  const windLines = new THREE.LineSegments(windGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false }));
  windLines.frustumCulled = false; spin.add(windLines);
  const respawn = (k, seed) => { wind.lat[k] = Math.asin(2 * hash(k, seed) - 1); wind.lon[k] = 2 * Math.PI * hash(seed, k); wind.age[k] = 0; wind.life[k] = 2.5 + 4 * hash(k + 9, seed); };
  for (let k = 0; k < N_WIND; k++) { respawn(k, 1); wind.age[k] = wind.life[k] * hash(k, 77); }
  /** Surface wind of the three-cell circulation at latitude lat [rad]: east-west u and north-south v (visual units). */
  function windAt(lat, lon, t) {
    const a = Math.abs(lat);
    const u = -Math.sin(6 * a) * 0.11 + 0.025 * Math.sin(3 * lon + 0.4 * t + 5 * lat);   // trades (east) / westerlies / polar easterlies
    const v = -Math.sin(6 * lat) * 0.035 + 0.015 * Math.cos(2 * lon - 0.3 * t + 7 * lat); // toward the rain belt, poleward 30-60
    return [u, v];
  }
  const air = new THREE.Mesh(new THREE.SphereGeometry(1.045, 64, 32), new THREE.ShaderMaterial({
    uniforms: { strength: { value: 1 } },
    vertexShader: 'varying vec3 vN; varying vec3 vP; void main(){ vN = normalize(normalMatrix * normal); vec4 mv = modelViewMatrix * vec4(position, 1.0); vP = mv.xyz; gl_Position = projectionMatrix * mv; }',
    fragmentShader: 'uniform float strength; varying vec3 vN; varying vec3 vP; void main(){ float i = pow(1.0 - abs(dot(normalize(vN), normalize(-vP))), 2.5); gl_FragColor = vec4(vec3(0.35, 0.62, 1.0) * i * strength, 1.0); }',
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
  }));
  scene.add(air);

  // mirror ring (enabler E3): plane through Earth perpendicular to the sunlight (here the y-z plane)
  const mirrorGroup = new THREE.Group(); scene.add(mirrorGroup);
  const mirrorPanels = new THREE.InstancedMesh(new THREE.PlaneGeometry(0.17, 0.17),
    new THREE.MeshStandardMaterial({ color: 0xdfe9ff, metalness: 0.9, roughness: 0.15, emissive: 0x506080, emissiveIntensity: 0.6, side: THREE.DoubleSide }), N_MIRRORS);
  mirrorPanels.frustumCulled = false; mirrorGroup.add(mirrorPanels);
  const ringLine = new THREE.Mesh(new THREE.TorusGeometry(MIRROR_DRAW_R, 0.006, 6, 200), new THREE.MeshBasicMaterial({ color: 0x9fd8ff, transparent: true, opacity: 0.35 }));
  ringLine.rotation.y = Math.PI / 2; mirrorGroup.add(ringLine);
  const rayPos = new Float32Array(N_MIRRORS * 4 * 3), rayCol = new Float32Array(N_MIRRORS * 4 * 3), rayGeo = new THREE.BufferGeometry();
  rayGeo.setAttribute('position', new THREE.BufferAttribute(rayPos, 3)); rayGeo.setAttribute('color', new THREE.BufferAttribute(rayCol, 3));
  const rays = new THREE.LineSegments(rayGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.4, blending: THREE.AdditiveBlending, depthWrite: false }));   // thin, very light rays
  rays.frustumCulled = false; mirrorGroup.add(rays);
  const photonPos = new Float32Array(N_MIRRORS * 3 * 3), photonGeo = new THREE.BufferGeometry(); photonGeo.setAttribute('position', new THREE.BufferAttribute(photonPos, 3));
  const photons = new THREE.Points(photonGeo, new THREE.PointsMaterial({ color: 0xfffaf0, size: 2, sizeAttenuation: false, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false }));
  photons.frustumCulled = false; mirrorGroup.add(photons);
  const mirrors = [...Array(N_MIRRORS)].map((_, k) => ({ a: 2 * Math.PI * k / N_MIRRORS, p: V3(), out: V3(), hit: V3(), aimed: false }));

  // labels + life panel (DOM)
  const LCOL = { sun: '#ffcc33', mirrors: '#bfe6ff', itcz: '#e8f0ff', trades: '#9fd0ff', west: '#c6b8ff', ice: '#ffffff', grow: '#9be37a', polar: '#b8d8ff' };
  const LBL = {};
  for (const [k, col] of Object.entries(LCOL)) { const d = document.createElement('div'); d.className = 'blabel mono small povlbl'; d.style.color = col; d.style.display = 'none'; container.appendChild(d); LBL[k] = d; }
  const panel = document.createElement('div'); panel.id = 'wxPanel'; panel.className = 'small'; panel.hidden = true; container.appendChild(panel);
  panel.addEventListener('click', (e) => { const b = e.target.closest('[data-wx]'); if (!b) return; if (b.dataset.wx === 'toggle') { P.noMirrors = !P.noMirrors; P.panelWall = 0; } });

  const _lv = V3();
  function label(k, pos, text) {
    const el = LBL[k];
    if (!P.labels || !pos || !text) { el.style.display = 'none'; return; }
    _lv.copy(pos).project(camera);
    if (_lv.z > 1 || Math.abs(_lv.x) > 1 || Math.abs(_lv.y) > 1) { el.style.display = 'none'; return; }
    if (el.textContent !== text) el.textContent = text; el.style.display = 'block';
    const w = canvas.clientWidth, h = canvas.clientHeight;
    el.style.left = `${clamp((_lv.x * 0.5 + 0.5) * w + 8, 2, w - el.offsetWidth - 2)}px`; el.style.top = `${clamp((-_lv.y * 0.5 + 0.5) * h - 6, 2, h - el.offsetHeight - 2)}px`;
  }
  function hideLabels() { for (const el of Object.values(LBL)) el.style.display = 'none'; }
  /** A point at latitude latD on the meridian that faces the camera (shifted by dLon), in world coordinates. */
  const _axis = V3(), _c = V3(), _e = V3();
  function facingPoint(latD, dLonD, r = 1.03) {
    _axis.set(0, 0, 1).applyQuaternion(tilt.quaternion);
    _c.copy(camera.position).normalize(); _c.addScaledVector(_axis, -_c.dot(_axis)); if (_c.lengthSq() < 1e-6) _c.set(1, 0, 0); _c.normalize();
    _c.applyAxisAngle(_axis, dLonD * D2R); const la = latD * D2R;
    const p = V3().copy(_c).multiplyScalar(Math.cos(la)).addScaledVector(_axis, Math.sin(la)).multiplyScalar(r);
    _e.copy(camera.position).normalize(); return p.dot(_e) > 0.25 ? p : null;
  }

  function lookDefault() { camera.position.set(6.2, -6.0, 3.0); controls.target.set(0, 0, 0); controls.update(); }
  lookDefault();

  // ---------------------------------------------------------------- per-frame update
  let cloudsBuilt = false, wallSpin = 0;
  /**
   * S: { has, t, T, sun, mirRel, capRel, set, Tno, noTxt, rAU, md (mirror_design), life (constants), playing, speedY }
   * In "without mirrors" mode the view shows the temperature Earth would settle at with the same sunlight but no
   * mirror light (recorded by the Python climate model as T_no_mirror_K).
   */
  function update(S, dt, wall) {
    P.S = S;
    if (!cloudsBuilt) { cloudMatT.map = buildClouds('tropics'); cloudMatM.map = buildClouds('midlat'); cloudMatT.needsUpdate = cloudMatM.needsUpdate = true; cloudsBuilt = true; }
    const L = S.life || DEFAULT_LIFE, noMir = P.noMirrors && S.has;
    const T = noMir ? (Number.isFinite(S.Tno) ? S.Tno : S.T) : S.T, mirRel = noMir ? 0 : S.mirRel || 0, sun = Number.isFinite(S.sun) ? S.sun : 1;
    P.T = T; P.life = lifeFromT(T, L); P.mirRel = mirRel; P.sun = sun;
    // surface texture: rebuilt when the temperature changed by more than 0.75 K (it takes ~0.1 s)
    const key = `${Math.round(T / 0.75)}`;
    if (key !== P.surfKey) {
      P.surfKey = key; const tx = buildSurface(T, L);
      if (surfMat.map) surfMat.map.dispose(); if (nightMat.uniforms.map.value) nightMat.uniforms.map.value.dispose();
      surfMat.map = tx.day; surfMat.needsUpdate = true; nightMat.uniforms.map.value = tx.night;
    }
    // weather: water vapour rises ~7 % per kelvin (Clausius-Clapeyron) -> cloud cover scaled with temperature
    const vapour = clamp(Math.exp(0.07 * (T - 288)), 0.03, 1.6), cloudA = clamp(0.25 + 0.6 * Math.min(1, vapour), 0.12, 0.92);
    cloudMatT.opacity = cloudA * clamp(vapour, 0.15, 1); cloudMatM.opacity = cloudA;
    const lit = P.life.grow_frac > 0.02 && !noMir;
    nightMat.uniforms.strength.value = lit ? 1.1 : 0;
    // lights: sunlight ~ sqrt(relative flux) (so a dim Sun is visible but darker), mirror light from the ring
    sunLight.intensity = 2.6 * Math.sqrt(clamp(sun, 0, 2));
    const Im = 1.5 * Math.sqrt(clamp(mirRel, 0, 1.5));
    for (const l of mirrorLights) l.intensity = Im * 0.5;
    // the Sun's true angular size at this distance (0.533 deg at 1 AU), drawn 3x larger so it is visible
    const r = S.rAU || 1, ang = 0.533 * D2R / r * 3;
    sunCore.scale.setScalar(1500 * Math.tan(ang / 2)); sunGlow.scale.setScalar(1500 * Math.tan(ang / 2) * 9);
    // spin: one turn per 80 s of wall-clock time (illustration – a real day would be a blur at mission speed)
    wallSpin += dt;
    spin.rotation.z = 2 * Math.PI * wallSpin / 80;
    cloudTropics.rotation.z = -0.018 * wallSpin;        // easterly trade winds: clouds lag behind the surface
    cloudMid.rotation.z = 0.03 * wallSpin;              // westerlies: clouds run ahead
    for (const c of cyclones) { c.m.rotateZ(c.spinRate * dt); c.m.material.opacity = clamp((c.tropical ? vapour - 0.6 : vapour) * 1.2, 0, 0.95) * (c.tropical && T < 280 ? 0 : 1); }
    // wind tracers
    const tw = wallSpin; let o = 0;
    for (let k = 0; k < N_WIND; k++) {
      wind.age[k] += dt; if (wind.age[k] > wind.life[k] || Math.abs(wind.lat[k]) > 1.45) respawn(k, Math.floor(wall * 10) + k);
      const la = wind.lat[k], lo = wind.lon[k], [u, v] = windAt(la, lo, tw), cl = Math.max(0.2, Math.cos(la));
      wind.lon[k] = lo + u * dt / cl * 2.2; wind.lat[k] = la + v * dt * 2.2;
      const tail = 0.35, la0 = la - v * tail * 2.2, lo0 = lo - u * tail / cl * 2.2, R = 1.026;
      windPos[o] = R * Math.cos(la0) * Math.cos(lo0); windPos[o + 1] = R * Math.cos(la0) * Math.sin(lo0); windPos[o + 2] = R * Math.sin(la0);
      windPos[o + 3] = R * Math.cos(la) * Math.cos(lo); windPos[o + 4] = R * Math.cos(la) * Math.sin(lo); windPos[o + 5] = R * Math.sin(la);
      const f = Math.sin(Math.PI * clamp(wind.age[k] / wind.life[k], 0, 1)), sp = clamp(Math.hypot(u, v) / 0.11, 0, 1);
      const cr = (0.25 + 0.5 * sp) * f, cg = (0.5 + 0.35 * sp) * f, cb = 0.95 * f;
      windCol.set([cr * 0.15, cg * 0.15, cb * 0.15, cr, cg, cb], o); o += 6;
    }
    windGeo.attributes.position.needsUpdate = true; windGeo.attributes.color.needsUpdate = true;
    windLines.material.opacity = noMir ? 0.3 : 0.55;
    // mirror ring: the share aimed at Earth = mirror light / capacity (the agent's action a4)
    mirrorGroup.visible = !!(S.md && S.md.on) && !noMir;
    if (mirrorGroup.visible) {
      const cap = Math.max(1e-9, S.capRel || (S.md && S.md.cap_rel) || 1), aimed = clamp(mirRel / cap, 0, 1), nA = Math.round(aimed * N_MIRRORS);
      const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), sc = V3(1, 1, 1), nrm = V3(), toE = V3();
      let ro = 0, po = 0;
      for (let k = 0; k < N_MIRRORS; k++) {
        const M = mirrors[k], a = M.a + 0.02 * wallSpin;
        M.p.set(0, Math.cos(a) * MIRROR_DRAW_R, Math.sin(a) * MIRROR_DRAW_R);
        M.aimed = ((k * 7) % N_MIRRORS) < nA;
        toE.copy(M.p).negate().normalize();
        if (M.aimed) M.out.copy(toE); else M.out.copy(toE).multiplyScalar(-0.3).addScaledVector(SUN_DIR, -1).normalize();
        nrm.copy(SUN_DIR).add(M.out).normalize();
        m4.lookAt(V3(), nrm, V3(0, 0, 1).cross(nrm).lengthSq() > 1e-6 ? V3(0, 0, 1) : V3(0, 1, 0)); q.setFromRotationMatrix(m4);
        m4.compose(M.p, q, sc); mirrorPanels.setMatrixAt(k, m4);
        M.hit.copy(M.aimed ? toE : M.out).multiplyScalar(M.aimed ? 1.0 : 1.4);
        if (!M.aimed) M.hit.add(M.p);
        // incoming sunlight (from +x) and the reflected ray
        rayPos.set([M.p.x + 1.4, M.p.y, M.p.z, M.p.x, M.p.y, M.p.z, M.p.x, M.p.y, M.p.z, M.hit.x, M.hit.y, M.hit.z], ro * 3);
        // very light, nearly white colours: incoming sunlight fades in from the Sun side, aimed reflections are the
        // brightest, light turned past Earth is very faint
        const bi = 0.3 + 0.4 * Math.sqrt(clamp(sun, 0, 1)), br = M.aimed ? 0.95 : 0.12;
        rayCol.set([0, 0, 0, bi, bi, bi * 0.94, br, br, br * 0.94, br, br, br * 0.94], ro * 3); ro += 4;
        for (let j = 0; j < 3; j++) {                    // light packets running along the reflected ray
          const f = ((wall * 0.5 + j / 3 + k * 0.137) % 1);
          photonPos.set([M.p.x + (M.hit.x - M.p.x) * f, M.p.y + (M.hit.y - M.p.y) * f, M.p.z + (M.hit.z - M.p.z) * f], po * 3); po++;
        }
      }
      mirrorPanels.instanceMatrix.needsUpdate = true; rayGeo.attributes.position.needsUpdate = true; rayGeo.attributes.color.needsUpdate = true;
      photonGeo.attributes.position.needsUpdate = true;
      rays.material.opacity = 0.2 + 0.3 * Math.min(1, Math.sqrt(mirRel)); photons.material.opacity = 0.15 + 0.45 * Math.min(1, mirRel);
    }
    air.material.uniforms.strength.value = 0.6 + 0.5 * Math.min(1, vapour);
    if (wall - P.panelWall > 0.5) { P.panelWall = wall; renderPanel(); }
  }

  // ---------------------------------------------------------------- the life panel
  function bar(frac, col) { return `<span class="pb"><i style="width:${clamp(100 * frac, 0, 100).toFixed(1)}%;background:${col}"></i></span>`; }
  function renderPanel() {
    const S = P.S; if (!S) return;
    const lf = P.life, noMir = P.noMirrors && S.has, C = (K) => `${num(K - 273.15, 0)} °C`;
    let h = `<div class="wxh">Weather &amp; life${S.has ? ` · year ${Math.round(S.t).toLocaleString('en-US')} <span class="dim">(simulated)</span>` : ' <span class="dim">· no mission running</span>'}</div>`;
    if (!S.has) h += '<div class="psub">Start a mission (▶) to see Earth far from the Sun. Until then: today\'s Earth at 1 AU.</div>';
    h += `<div class="btns"><button data-wx="toggle" class="${noMir ? 'warnbtn' : ''}" title="Switch between the recorded climate WITH the mirrors and the temperature Earth would settle at with the same sunlight but WITHOUT the mirrors (also computed by the climate model).">${noMir ? 'showing: WITHOUT mirrors → show with' : 'showing: WITH mirrors → show without'}</button></div>`;
    const tot = P.sun + P.mirRel;
    h += `<div class="prow" title="Light reaching Earth in units of today's sunlight (1,361 W/m² at 1 AU). Yellow = the Sun at Earth's simulated distance; white = sunlight reflected by the mirror ring."><span class="pl">light on Earth</span><span class="pb"><i style="width:${clamp(100 * P.sun / 1.5, 0, 100)}%;background:#ffd84d;display:inline-block"></i><i style="width:${clamp(100 * P.mirRel / 1.5, 0, 100)}%;background:#f4f4ff;display:inline-block"></i></span><span class="pv">${num(tot, 2)}×</span></div>`;
    h += `<div class="psub">Sun ${num(P.sun, 2)}× (at ${num(S.rAU, 2)} AU) + mirrors ${num(P.mirRel, 2)}× today's sunlight</div>`;
    if (S.md && S.md.on && !noMir) {
      const cap = S.capRel || S.md.cap_rel, aim = clamp(P.mirRel / Math.max(1e-9, cap), 0, 1);
      h += `<div class="prow" title="The agent (action a4) decides how much of the mirrors' light goes to Earth; the rest is turned past Earth. Capacity falls as 1/distance² like sunlight."><span class="pl">mirrors aimed at Earth</span>${bar(aim, '#bfe6ff')}<span class="pv">${num(100 * aim, 0)} %</span></div>`;
      h += `<div class="psub">ring: ${(S.md.area_m2 / 1e12).toFixed(0)} million km² of 2-µm film (${(S.md.mass_kg / 1e12).toFixed(0)}e12 kg) at ${S.md.orbit_re} Earth radii, can give up to ${num(cap, 2)}× here</div>`;
    }
    const okT = Math.abs(P.T - 288) <= 15;
    h += `<div class="prow" title="Global-mean surface temperature (climate model). Equator and poles from the simple zonal profile used for the life indicators."><span class="pl">temperature</span>${bar((P.T - 150) / 170, okT ? '#3ddc84' : '#ff6a5a')}<span class="pv">${C(P.T)}</span></div>`;
    h += `<div class="psub">equator ${C(lf.T_equator_K)} · poles ${C(lf.T_pole_K)} (today 15 °C · 30 °C · −15 °C)</div>`;
    h += `<div class="prow" title="Latitude where the zonal temperature drops below 263 K (−10 °C): ice and snow all year poleward of it. Today ~70°."><span class="pl">ice line</span>${bar(lf.ice_line_deg / 90, '#e8f0ff')}<span class="pv">${num(lf.ice_line_deg, 0)}°</span></div>`;
    h += `<div class="prow" title="Share of Earth's surface warmer than 263 K: open (liquid) ocean and ice-free land."><span class="pl">ice-free surface</span>${bar(lf.liquid_frac, '#4fa3ff')}<span class="pv">${num(100 * lf.liquid_frac, 0)} %</span></div>`;
    h += `<div class="prow" title="Share of Earth's surface at 5-40 °C, where crops and forests grow. Today ~75 % in this simple profile."><span class="pl">growing zone</span>${bar(lf.grow_frac, '#7ed957')}<span class="pv">${num(100 * lf.grow_frac, 0)} %</span></div>`;
    h += `<div class="pone"><b>${esc(lf.state)}</b>${noMir ? ' – this is the Earth WITHOUT the mirrors (same sunlight).' : ''}</div>`;
    if (S.has && Number.isFinite(S.Tno) && !noMir) h += `<div class="psub warn">Without the mirrors Earth would settle at ${num(S.Tno, 0)} K (${C(S.Tno)}): ${esc(S.noTxt || lifeFromT(S.Tno, S.life || DEFAULT_LIFE).state)}. Press the button above to see it.</div>`;
    h += `<div class="psub">Weather drawn: rain belt at the equator, dry subtropics, trade winds, westerlies with storm tracks, polar easterlies – today's three-cell circulation, kept because the equator-to-pole contrast and the day-night cycle stay like today${noMir ? ' (on a frozen Earth the air is dry: few clouds)' : ''}.</div>`;
    h += '<div class="psub dim">Computed: temperatures, light, mirror aim, ice line, growing zone. Illustration: continents, clouds, winds, storms, city lights, mirror ring drawn at 2.6 R<sub>E</sub> (real 20 R<sub>E</sub>, ~1 million mirrors).</div>';
    if (panel.innerHTML !== h) panel.innerHTML = h;
  }

  function placeLabels() {
    if (!P.labels) { hideLabels(); return; }
    const S = P.S || {}, lf = P.life, noMir = P.noMirrors && S.has;
    label('sun', V3().copy(SUN_DIR).multiplyScalar(6).add(V3(0, 0, 0.4)), `Sun → ${num((P.sun || 0), 2)}× today's sunlight`);
    label('mirrors', mirrorGroup.visible ? V3(0, -0.4 * MIRROR_DRAW_R, 0.93 * MIRROR_DRAW_R) : null, `space mirrors E3 · ${num(100 * clamp((P.mirRel || 0) / Math.max(1e-9, S.capRel || 1), 0, 1), 0)} % aimed at Earth · drawn at 2.6 R_E (real 20)`);
    label('itcz', lf.liquid_frac > 0.2 ? facingPoint(4, -20) : null, 'tropical rain belt (clouds, thunderstorms)');
    label('trades', lf.liquid_frac > 0.2 ? facingPoint(-17, 25) : null, '← trade winds (easterlies)');
    label('west', facingPoint(47, 18), noMir ? 'westerlies (dry, frozen surface)' : 'westerlies + storms →');
    label('polar', lf.ice_line_deg < 88 ? facingPoint(-72, -10) : null, 'polar easterlies');
    label('ice', lf.ice_line_deg > 1 && lf.ice_line_deg < 89 ? facingPoint(lf.ice_line_deg, -35) : facingPoint(0, -40), lf.ice_line_deg > 1 ? `ice line ${num(lf.ice_line_deg, 0)}°` : 'ice to the equator – snowball');
    label('grow', lf.grow_frac > 0.05 ? facingPoint(28, -45) : null, `growing zone ${num(100 * lf.grow_frac, 0)} % · cities on the night side`);
    // keep labels from piling up: shift down when overlapping
    const placed = [];
    for (const el of Object.values(LBL)) {
      if (el.style.display === 'none') continue;
      let y = parseFloat(el.style.top), x = parseFloat(el.style.left); const w = el.offsetWidth, hh = el.offsetHeight;
      for (let n = 0; n < 8; n++) { const hit = placed.find((b) => x < b.x + b.w && x + w > b.x && y < b.y + b.h && y + hh > b.y); if (!hit) break; y = hit.y + hit.h + 2; }
      el.style.top = `${y}px`; placed.push({ x, y, w, h: hh });
    }
  }
  function render() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    camera.aspect = w / Math.max(1, h);
    const shift = Math.min(190, 0.13 * w); camera.setViewOffset(w, h, shift, 0, w, h);    // keep Earth clear of the life panel
    camera.updateProjectionMatrix(); controls.update(); camera.updateMatrixWorld(); scene.updateMatrixWorld();
    placeLabels();
    renderer.render(scene, camera);
  }
  function setActive(on) { P.active = on; controls.enabled = on; panel.hidden = !on; if (!on) hideLabels(); }
  return { scene, camera, controls, P, update, render, setActive, hideLabels, lookDefault, panel };
}
