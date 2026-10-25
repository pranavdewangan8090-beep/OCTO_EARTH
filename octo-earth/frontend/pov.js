// OCTO-EARTH – Earth point-of-view (POV) close-up.
// Coordinates: origin = Earth's centre, axes = the simulation's ecliptic axes, unit = Earth radius (R_E).
// Everything is drawn at TRUE scale (Earth, Moon, its distance, the asteroid size for its mass and an
// assumed density, magnetopause, Hill sphere, the Sun's angular size).
// What comes from the physics engine: Earth's position and velocity (-> Sun direction, orbital speed,
// distance), the Moon's relative position/velocity, remaining asteroid mass, and every ejection
// (time, direction, speed, fate). What is an ILLUSTRATION: the launcher ring and the eight arms
// (no hardware is simulated). What is a standard SIMPLIFIED MODEL evaluated at Earth's simulated Sun
// distance (see envmodel.js): solar wind, magnetopause, bow shock, field lines, radiation belts.
// SOLAR COLLECTORS / MICROWAVE BEAMS: the collector belt is drawn in its TRUE directions as seen from Earth
// (distances scaled like the Sun's); each beam has its TRUE width (the 1,000 km rectenna), and its brightness
// follows the beam power recorded by the physics engine for the current ejection event. Microwaves are
// invisible: their colour, the moving wavefronts (real wavelength 5 cm) and the pulses running down the arms
// are illustrations.
// None of these layers feed back into the orbit simulation.
import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import { satelliteGeometry, satQuat } from './collector.js';
import { spaceEnv, shueR, dipoleB, RE_KM, AU_KM, GM_SUN, GM_MOON } from './envmodel.js';

const D2R = Math.PI / 180;
const OBLIQ = 23.44 * D2R;
const SPIN_AXIS = new THREE.Vector3(0, Math.sin(OBLIQ), Math.cos(OBLIQ));   // Earth's north pole, ecliptic J2000 axes
const DIP_COLAT = 9.6 * D2R, DIP_LON = -72.7 * D2R;                         // geomagnetic pole (centred dipole)
const SUN_D = 40000;                    // Sun drawn at this distance [R_E] with its TRUE angular size
import { ARM_L, RING_R, GRAB_Z, armCurve, strokeLength } from './launcher.js';
const RING_TUBE = 0.055;
const MOON_R = 1737.4 / RE_KM;
const RHO_AST = 2000;                   // kg/m^3, assumed asteroid density (size display only)
const ROT_MAX_DAYS_PER_S = 2;           // above this the daily rotation is not animated
const SIDEREAL_DAY = 0.99726968;        // days
const NSEG = 12;                        // arm segments
const WIND_V = 30;                      // R_E per wall-clock second (visual speed, not real)

export const LAYERS = [
  { key: 'arms', label: 'Launcher ring + 8 arms', def: true, tip: 'ILLUSTRATION of the mass launcher: a ring around Earth with eight arms. For every ejection the physics records, ALL EIGHT arms close around the same single rock and throw it together in the recorded direction (one packet per throw, so no single arm carries the whole load). Each arm is 2 Earth radii long; the docked asteroid sits on the ring\'s axis, so it is the same distance from all 8 arm roots. No hardware, structure or arm dynamics are simulated.' },
  { key: 'moon', label: 'Moon', def: true, tip: 'The Moon at its simulated position, true size (0.27 R_E) and true distance (~60 R_E).' },
  { key: 'moonorbit', label: 'Moon orbit', def: false, tip: "The Moon's current orbit around Earth (osculating ellipse from the simulated Earth-Moon state)." },
  { key: 'field', label: 'Magnetic field lines', def: true, tip: "Earth's magnetic field as a centred dipole tilted 9.6° from the spin axis. Closed lines are cut where they would cross the magnetopause; the tail lines on the night side are schematic. Colour = field strength (pale cyan strong, violet weak)." },
  { key: 'magnetopause', label: 'Magnetopause + bow shock', def: false, tip: "Magnetopause (cyan): where the solar-wind pressure balances Earth's magnetic pressure (Shue 1998 model, IMF Bz = 0). Bow shock (orange): where the supersonic solar wind is slowed, ~1.3x farther out. Both move outward as Earth moves away from the Sun because the wind thins as 1/r^2." },
  { key: 'belts', label: 'Radiation belts', def: false, tip: 'Van Allen belts of trapped energetic particles: inner (~1.2-2.5 R_E, protons) and outer (~3-7 R_E, electrons). Schematic tori around the magnetic axis; real belts vary strongly with solar activity.' },
  { key: 'wind', label: 'Solar-wind plasma', def: true, tip: 'Protons and electrons streaming from the Sun at ~400 km/s. Particle count ∝ density (5 per cm^3 at 1 AU, falling as 1/r^2). Particles are animated at a slowed visual speed; behind the bow shock they are slowed, heated (orange) and deflected around the magnetosphere.' },
  { key: 'gravwell', label: 'Gravity well (curvature picture)', def: false, tip: "The effective gravitational potential around Earth in the frame that orbits the Sun with Earth (Earth + Moon + the Sun's tide, Hill approximation), drawn as a rubber sheet. Height is log-scaled; the rim saddles are the L1/L2 points. Earth sits at the bottom of its well. Turning this on zooms out." },
  { key: 'hill', label: 'Hill sphere, L1, L2', def: false, tip: "Hill sphere: region where Earth's gravity dominates over the Sun's tide (radius ∝ distance from the Sun, so it GROWS as Earth moves out). L1 and L2 are the balance points toward and away from the Sun. Turning this on zooms out." },
  { key: 'mars', label: 'Mars (direction, safety)', def: true, tip: "Mars as seen from Earth: a red marker in its TRUE direction (at its true distance up to 40,000 R_E, beyond that on a far sphere; the label gives the true distance). The planetary-safe distance (red) and the shield's caution distance (yellow) around Mars are drawn as TRUE-scale wire spheres when Mars is within 0.3 AU and you zoom out beyond 150 R_E (always when Mars is within 2x the caution distance). 1 AU = 23,481 R_E." },
  { key: 'gravvec', label: 'Gravity pulls (Sun, Moon, Mars)', def: false, tip: "Gravitational acceleration on Earth from the Sun, the Moon and Mars, in m/s^2 (arrow length is log-scaled; the Mars arrow has a minimum length so it stays visible). The docked asteroid's gravity acts inside the Earth+asteroid body (it raises Earth's pull on the Moon and on packets) and gives no net pull on Earth itself." },
  { key: 'motion', label: 'Motion + thrust arrows', def: true, tip: 'Orbital velocity of Earth around the Sun (simulated), and the direction of the latest push on Earth from a thrown packet (the recoil, opposite to the throw).' },
  { key: 'packets', label: 'Thrown packets', def: true, tip: 'Each recorded ejection as a streak in its true direction, coloured by its fate (orange: leaves the Solar System, yellow: stays bound to the Sun, red: fell back, grey: hit the Moon). At ~1% of light speed they really leave this view within seconds of simulated time, so the streaks move at a slowed visual speed and fade out; their DIRECTION and fate are the recorded ones.' },
  { key: 'beams', label: 'Microwave power beams', def: true, tip: 'Beamed power from the solar collectors to the 8 rectenna dishes at the arm tips. Beam width and dish size are TRUE scale (1,000 km = 0.157 Earth radii); brightness follows the beam power recorded by the physics engine (bright = the launcher is drawing power, faint = idle). Microwaves are invisible – the violet colour, wavefronts and the pulses running down the arms are illustrations.' },
  { key: 'collectors', label: 'Solar collector belt (sky)', def: false, tip: 'The collector belt at its configured distance from the Sun, drawn as dots in their TRUE directions as seen from Earth (distances scaled like the Sun\'s). Each dot stands for thousands of collectors; each collector is really ~65,000 km wide (5 times Earth\'s diameter).' },
  { key: 'mirrors', label: 'Space mirrors (climate)', def: true, tip: "The ring of thin space mirrors (enabler E3) that keeps Earth warm far from the Sun: they orbit Earth at 20 Earth radii (TRUE distance) in the plane through Earth perpendicular to the sunlight, tilted 45 degrees, and reflect extra sunlight onto Earth. The share of mirrors aimed at Earth and the brightness of their light follow the mirror light recorded by the climate model (the agent's action a4); the others turn their light past Earth. Each real mirror is ~40 km wide (drawn ~100x enlarged); 48 drawn for ~1 million." },
  { key: 'labels', label: 'Labels', def: true, tip: 'Text labels on bodies and boundaries.' },
];
export const DEFAULT_LAYERS = Object.fromEntries(LAYERS.map((l) => [l.key, l.def]));

// ---------------------------------------------------------------- helpers
const V3 = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const _q = new THREE.Quaternion(), _m = new THREE.Matrix4(), _s = new THREE.Vector3(), _a = new THREE.Vector3(), _b = new THREE.Vector3();
const UPY = V3(0, 1, 0), UPZ = V3(0, 0, 1);
function hash3(x, y, z) {
  let h = (Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(z, 1274126177)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177); h ^= h >>> 16; return (h >>> 0) / 4294967296;
}
function vnoise(x, y, z) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z), xf = x - xi, yf = y - yi, zf = z - zi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf), w = zf * zf * (3 - 2 * zf);
  const L = (a, b, t) => a + (b - a) * t;
  return L(L(L(hash3(xi, yi, zi), hash3(xi + 1, yi, zi), u), L(hash3(xi, yi + 1, zi), hash3(xi + 1, yi + 1, zi), u), v),
    L(L(hash3(xi, yi, zi + 1), hash3(xi + 1, yi, zi + 1), u), L(hash3(xi, yi + 1, zi + 1), hash3(xi + 1, yi + 1, zi + 1), u), v), w);
}
export function fbm(x, y, z, oct = 5) { let s = 0, a = 0.5, f = 1; for (let i = 0; i < oct; i++) { s += a * vnoise(x * f, y * f, z * f); f *= 2.03; a *= 0.5; } return s / (1 - Math.pow(0.5, oct)); }

let _earthTex = null;
export function earthTexture() { return _earthTex || (_earthTex = buildEarthTexture()); }   // shared by the overview
function buildEarthTexture() {      // procedural, not a map of the real continents
  const W = 1024, H = 512, c = document.createElement('canvas'); c.width = W; c.height = H;
  const g = c.getContext('2d'), img = g.createImageData(W, H), d = img.data;
  for (let j = 0; j < H; j++) {
    const lat = (0.5 - (j + 0.5) / H) * Math.PI, cl = Math.cos(lat), sl = Math.sin(lat);
    for (let i = 0; i < W; i++) {
      const lon = (i + 0.5) / W * 2 * Math.PI, x = cl * Math.cos(lon), z = cl * Math.sin(lon);
      const n = fbm(x * 1.9 + 7, sl * 1.9 + 3, z * 1.9 + 11);
      const ice = Math.abs(lat) > (68 + 8 * (fbm(x * 4, sl * 4, z * 4, 3) - 0.5) * 2) * D2R;
      let r, gg, b;
      if (ice) { r = 235; gg = 240; b = 245; }
      else if (n > 0.53) {
        const k = Math.min(1, (n - 0.53) / 0.2), dry = fbm(x * 3 + 1, sl * 3, z * 3, 3);
        r = 60 + 90 * k * dry + 40 * (1 - Math.abs(sl)) * dry; gg = 105 + 25 * (1 - k) - 10 * dry; b = 55 + 20 * k;
      } else { const k = Math.min(1, (0.53 - n) / 0.2); r = 18 + 10 * (1 - k); gg = 60 + 40 * (1 - k); b = 120 + 40 * (1 - k); }
      const o = (j * W + i) * 4; d[o] = r; d[o + 1] = gg; d[o + 2] = b; d[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
function glowTexture(inner, outer) {
  const c = document.createElement('canvas'); c.width = c.height = 128; const g = c.getContext('2d');
  const gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  gr.addColorStop(0, inner); gr.addColorStop(0.25, inner); gr.addColorStop(1, outer);
  g.fillStyle = gr; g.fillRect(0, 0, 128, 128); return new THREE.CanvasTexture(c);
}
function lineMat(color, opacity, extra = {}) { return new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false, ...extra }); }
function dynLines(maxVerts, mat, segments = true) {
  const pos = new Float32Array(maxVerts * 3), col = new Float32Array(maxVerts * 3), g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3)); g.setDrawRange(0, 0);
  const o = segments ? new THREE.LineSegments(g, mat) : new THREE.Line(g, mat); o.frustumCulled = false; o.userData = { pos, col, max: maxVerts }; return o;
}
function fieldColor(B, out, o) {        // log |B|: 1e-8.5 T violet -> 1e-6.5 cyan -> 3e-5 white
  const t = Math.max(0, Math.min(1, (Math.log10(B) + 8.5) / 4.0));
  let r, g, b;
  if (t < 0.5) { const k = t / 0.5; r = 0.45 - 0.1 * k; g = 0.25 + 0.6 * k; b = 0.95; }
  else { const k = (t - 0.5) / 0.5; r = 0.35 + 0.45 * k; g = 0.85 + 0.12 * k; b = 0.95 + 0.05 * k; }
  out[o] = r; out[o + 1] = g; out[o + 2] = b;
}

// ---------------------------------------------------------------- the POV object
export function createPov(renderer, canvas, container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x020308);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 6e4);
  camera.up.set(0, 0, 1);                       // ecliptic north is "up"
  camera.position.set(-6, 5, 3);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true; controls.enablePan = false; controls.minDistance = 1.3; controls.maxDistance = 3000; controls.zoomSpeed = 1.4;
  controls.enabled = false;
  const layers = { ...DEFAULT_LAYERS };
  const P = { layers, env: null, sunDir: V3(1, 0, 0), nrm: V3(0, 0, 1), vhat: V3(0, 1, 0), speedKms: NaN, rAU: 1, t: 0,
    astR: 0, astPos: V3(), mSys: 6.05e24, moonPos: null, aSun: NaN, aMoon: NaN, rotFrozen: false, lastThrowWall: -1e9,
    notes: [], gravVisible: false, lockSun: true, initView: false, throwDelay: 0 };

  // lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.07));
  const sunLight = new THREE.DirectionalLight(0xffffff, 2.6); scene.add(sunLight); scene.add(sunLight.target);

  // stars (fixed directions; orientation aid only)
  {
    const n = 1400, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const z = hash3(i, 1, 7) * 2 - 1, ph = hash3(i, 2, 9) * Math.PI * 2, rr = Math.sqrt(1 - z * z);
      pos[3 * i] = rr * Math.cos(ph) * 50000; pos[3 * i + 1] = rr * Math.sin(ph) * 50000; pos[3 * i + 2] = z * 50000;
    }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const st = new THREE.Points(g, new THREE.PointsMaterial({ color: 0x8890a0, size: 1.3, sizeAttenuation: false, transparent: true, opacity: 0.55, depthWrite: false }));
    st.renderOrder = -20; st.frustumCulled = false; scene.add(st);
  }
  // Sun: disc with true angular size + glow
  const sunCore = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16), new THREE.MeshBasicMaterial({ color: 0xfff2c0, depthTest: false, depthWrite: false }));
  sunCore.renderOrder = -10; scene.add(sunCore);
  const sunGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture('rgba(255,220,140,0.9)', 'rgba(255,160,40,0)'), depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending }));
  sunGlow.renderOrder = -9; scene.add(sunGlow);

  // Earth (rotates about its spin axis)
  const earthTilt = new THREE.Group(); earthTilt.quaternion.setFromUnitVectors(UPY, SPIN_AXIS); scene.add(earthTilt);
  const earthSpin = new THREE.Group(); earthTilt.add(earthSpin);
  const earthMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9, metalness: 0.0 });
  const earthMesh = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 48), earthMat); earthSpin.add(earthMesh);
  const atmo = new THREE.Mesh(new THREE.SphereGeometry(1.025, 64, 32), new THREE.MeshBasicMaterial({ color: 0x5aa0ff, transparent: true, opacity: 0.12, side: THREE.BackSide, depthWrite: false }));
  scene.add(atmo);
  const dipoleBody = V3(Math.sin(DIP_COLAT) * Math.cos(DIP_LON), Math.cos(DIP_COLAT), -Math.sin(DIP_COLAT) * Math.sin(DIP_LON));
  const mhat = V3(0, 0, 1);
  let texDone = false;
  // ring of space mirrors (enabler E3) at its TRUE orbit radius of 20 Earth radii; mirrors drawn enlarged
  const MIR_N = 48, MIR_ORBIT = 20;
  const mirrorRing = new THREE.Group(); scene.add(mirrorRing); mirrorRing.visible = false;
  const mirrorPanels = new THREE.InstancedMesh(new THREE.PlaneGeometry(0.9, 0.9),
    new THREE.MeshBasicMaterial({ color: 0xeef6ff, side: THREE.DoubleSide, transparent: true, opacity: 0.95 }), MIR_N);
  mirrorPanels.frustumCulled = false; mirrorRing.add(mirrorPanels);
  const mirrorOrbitLine = new THREE.Mesh(new THREE.TorusGeometry(MIR_ORBIT, 0.04, 6, 256),
    new THREE.MeshBasicMaterial({ color: 0x9fd8ff, transparent: true, opacity: 0.3, depthWrite: false })); mirrorRing.add(mirrorOrbitLine);
  const mirrorRays = dynLines(MIR_N * 4, lineMat(0xffffff, 0.4, { blending: THREE.AdditiveBlending, vertexColors: true }));   // incoming + reflected, thin and very light
  mirrorRing.add(mirrorRays);
  // four lights from the ring directions stand in for the reflected light (brightness ~ recorded mirror light)
  const mirrorLights = [...Array(4)].map(() => { const l = new THREE.DirectionalLight(0xfff2dc, 0); scene.add(l); scene.add(l.target); return l; });

  // ---------------------------------------------------------------- launcher ring + arms (illustration)
  const armGroup = new THREE.Group(); scene.add(armGroup);
  const metal = new THREE.MeshStandardMaterial({ color: 0x8b94a3, roughness: 0.35, metalness: 0.85 });
  const darkMetal = new THREE.MeshStandardMaterial({ color: 0x4a505a, roughness: 0.5, metalness: 0.8 });
  const jointMat = new THREE.MeshStandardMaterial({ color: 0x401010, emissive: 0xff2a1a, emissiveIntensity: 1.2, roughness: 0.4 });
  const ring = new THREE.Mesh(new THREE.TorusGeometry(RING_R, RING_TUBE, 14, 160), metal); armGroup.add(ring);
  const struts = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 10), darkMetal, 4); armGroup.add(struts);
  const hubs = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.11, 0.13, 0.16, 16), darkMetal, 8); armGroup.add(hubs);
  const segs = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 12), metal, 8 * NSEG); armGroup.add(segs);
  const joints = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 14, 10), jointMat, 8 * 5); armGroup.add(joints);
  const claws = new THREE.InstancedMesh(new THREE.ConeGeometry(0.045, 0.36, 8), darkMetal, 8 * 3); armGroup.add(claws);
  for (const o of [struts, hubs, segs, joints, claws]) o.frustumCulled = false;
  // the single packet that all eight arms throw together (drawn enlarged: a real 1e18 kg packet is ~100 km across = 0.015 R_E)
  const ROCK_R = 0.13, GROUP_T = 0.95;
  const heldRock = new THREE.Mesh(new THREE.IcosahedronGeometry(ROCK_R, 1), new THREE.MeshStandardMaterial({ color: 0xc08a58, emissive: 0x4a1c06, roughness: 0.9, flatShading: true }));
  { const pa = heldRock.geometry.attributes.position; for (let i = 0; i < pa.count; i++) { const f = 0.8 + 0.4 * hash3(Math.round(pa.getX(i) * 97), Math.round(pa.getY(i) * 89), Math.round(pa.getZ(i) * 83)); pa.setXYZ(i, pa.getX(i) * f, pa.getY(i) * f, pa.getZ(i) * f); } heldRock.geometry.computeVertexNormals(); }
  const group = { phase: 1, released: false, eh: V3(1, 0, 0), pending: [], G: V3(), F: V3(), rock: V3(), u: V3(), w: V3() };
  armGroup.add(heldRock);
  const arms = [...Array(8)].map((_, k) => ({ k, phase: 1, ehat: V3(1, 0, 0), dir: V3(), base: V3(), pts: [...Array(NSEG + 1)].map(() => V3()), tip: V3(), pending: [] }));

  // asteroid: rock with real size for its mass (density assumed)
  const astGeom = new THREE.IcosahedronGeometry(1, 3);
  { const p = astGeom.attributes.position; for (let i = 0; i < p.count; i++) { _a.fromBufferAttribute(p, i); const f = 0.78 + 0.4 * fbm(_a.x * 1.7 + 5, _a.y * 1.7, _a.z * 1.7, 4); _a.multiplyScalar(f); p.setXYZ(i, _a.x, _a.y, _a.z); } astGeom.computeVertexNormals(); }
  const asteroid = new THREE.Mesh(astGeom, new THREE.MeshStandardMaterial({ color: 0xa27b58, roughness: 1, flatShading: true }));
  scene.add(asteroid);

  // Moon
  const moon = new THREE.Mesh(new THREE.SphereGeometry(MOON_R, 48, 24), new THREE.MeshStandardMaterial({ color: 0xb8b8b8, roughness: 1 }));
  scene.add(moon);
  const moonOrbit = dynLines(129, lineMat(0xbbbbbb, 0.45), false); scene.add(moonOrbit);

  // magnetic field lines
  const field = dynLines(40000, lineMat(0xffffff, 0.42, { vertexColors: true })); scene.add(field);
  // magnetopause + bow shock (built along +Y, rotated to the Sun direction)
  const bounds = new THREE.Group(); scene.add(bounds);
  let mpObjs = [], boundKey = '';
  // radiation belts
  const belts = new THREE.Group(); scene.add(belts);
  const beltMat = (c, o) => new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: o, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide });
  const inner = new THREE.Mesh(new THREE.TorusGeometry(1.85, 0.55, 24, 96), beltMat(0xff5a3a, 0.16)); inner.scale.set(1, 1, 0.75); belts.add(inner);
  const outer = new THREE.Mesh(new THREE.TorusGeometry(4.8, 1.55, 24, 128), beltMat(0xffa040, 0.10)); outer.scale.set(1, 1, 0.7); belts.add(outer);
  // solar wind
  const NW = 2600, wx = new Float32Array(NW), wy = new Float32Array(NW), wz = new Float32Array(NW), wSheath = new Uint8Array(NW);
  for (let i = 0; i < NW; i++) { wx[i] = -100 + 170 * Math.random(); wy[i] = (Math.random() * 2 - 1) * 70; wz[i] = (Math.random() * 2 - 1) * 70; }
  const windPos = new Float32Array(NW * 3), windCol = new Float32Array(NW * 3), windGeom = new THREE.BufferGeometry();
  windGeom.setAttribute('position', new THREE.BufferAttribute(windPos, 3)); windGeom.setAttribute('color', new THREE.BufferAttribute(windCol, 3));
  const wind = new THREE.Points(windGeom, new THREE.PointsMaterial({ size: 3.0, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.95, depthWrite: false }));
  wind.frustumCulled = false; scene.add(wind);
  // gravity well grid
  const GW_N = 44, GW_S = 88;
  const well = dynLines(2 * GW_N * GW_S * 2, lineMat(0xffffff, 0.55, { vertexColors: true })); scene.add(well);
  const wellMarks = new THREE.Group(); scene.add(wellMarks);
  const wellDrop = dynLines(4, lineMat(0xdddddd, 0.5)); wellMarks.add(wellDrop);
  let wellFrame = 0;
  // Hill sphere
  const hillG = new THREE.Group(); scene.add(hillG);
  const hillCircle = dynLines(257, lineMat(0x9be37a, 0.7), false); hillG.add(hillCircle);
  const hillSphere = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16), new THREE.MeshBasicMaterial({ color: 0x9be37a, wireframe: true, transparent: true, opacity: 0.06, depthWrite: false }));
  hillG.add(hillSphere);
  const lpMat = new THREE.MeshBasicMaterial({ color: 0x9be37a });
  const L1 = new THREE.Mesh(new THREE.OctahedronGeometry(1, 0), lpMat), L2 = new THREE.Mesh(new THREE.OctahedronGeometry(1, 0), lpMat); hillG.add(L1); hillG.add(L2);
  // arrows
  const mkArrow = (c) => { const a = new THREE.ArrowHelper(V3(1, 0, 0), V3(), 1, c, 0.4, 0.22); a.frustumCulled = false; scene.add(a); return a; };
  const arSun = mkArrow(0xffcc33), arMoon = mkArrow(0xdddddd), arVel = mkArrow(0x7cc4ff), arPush = mkArrow(0x7fe08a), arMars = mkArrow(0xff6a4d);
  // Mars: marker in its true direction + true-scale safety spheres
  const marsG = new THREE.Group(); scene.add(marsG);
  const marsDotGeom = new THREE.BufferGeometry(); marsDotGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
  const marsDot = new THREE.Points(marsDotGeom, new THREE.PointsMaterial({ size: 12, sizeAttenuation: false, color: 0xff6a4d, transparent: true, depthWrite: false, map: glowTexture('rgba(255,255,255,1)', 'rgba(255,255,255,0)') }));
  marsDot.frustumCulled = false; marsG.add(marsDot);
  const marsSafeS = new THREE.Mesh(new THREE.SphereGeometry(1, 28, 14), new THREE.MeshBasicMaterial({ color: 0xff4d4d, wireframe: true, transparent: true, opacity: 0.10, depthWrite: false }));
  const marsCautS = new THREE.Mesh(new THREE.SphereGeometry(1, 28, 14), new THREE.MeshBasicMaterial({ color: 0xffcf4d, wireframe: true, transparent: true, opacity: 0.045, depthWrite: false }));
  marsG.add(marsSafeS); marsG.add(marsCautS);
  const GM_MARS = 4.282837e13;
  // packets
  const NPK = 48; const pk = [...Array(NPK)].map(() => ({ on: false }));
  const pkLines = dynLines(NPK * 2, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 1, depthWrite: false, blending: THREE.AdditiveBlending })); scene.add(pkLines);
  const pkHeadPos = new Float32Array(NPK * 3), pkHeadCol = new Float32Array(NPK * 3), pkHeadGeom = new THREE.BufferGeometry();
  pkHeadGeom.setAttribute('position', new THREE.BufferAttribute(pkHeadPos, 3)); pkHeadGeom.setAttribute('color', new THREE.BufferAttribute(pkHeadCol, 3));
  const pkHeads = new THREE.Points(pkHeadGeom, new THREE.PointsMaterial({ size: 7, sizeAttenuation: false, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, map: glowTexture('rgba(255,255,255,1)', 'rgba(255,255,255,0)') }));
  pkHeads.frustumCulled = false; scene.add(pkHeads);
  const PKCOL = { 1: [1.0, 0.6, 0.3], 2: [1.0, 0.88, 0.3], 3: [1.0, 0.35, 0.35], 4: [0.8, 0.8, 0.8] };

  // ---------------------------------------------------------------- solar collectors + microwave beams
  const NWAVE = 14, WAVE_SPAN = 14, NPULSE = 5, BEAM_LEN = 3000, NBELT_LON = 360, NBELT_LAT = 5;
  const add = THREE.AdditiveBlending;
  const beamG = new THREE.Group(); scene.add(beamG);
  const beamCyl = new THREE.CylinderGeometry(1, 1, 1, 16, 1, true);
  const beamGlowTex = glowTexture('rgba(255,255,255,1)', 'rgba(255,255,255,0)');
  const beams = arms.map((A) => {
    const core = new THREE.Mesh(beamCyl, new THREE.MeshBasicMaterial({ color: 0xc8a8ff, transparent: true, opacity: 0.5, depthWrite: false, blending: add, side: THREE.DoubleSide }));
    const halo = new THREE.Mesh(beamCyl, new THREE.MeshBasicMaterial({ color: 0x7a4cff, transparent: true, opacity: 0.12, depthWrite: false, blending: add, side: THREE.DoubleSide }));
    const line = dynLines(2, lineMat(0xb48cff, 0.5, { blending: add }));
    const dish = new THREE.Mesh(new THREE.CircleGeometry(1, 40), new THREE.MeshStandardMaterial({ color: 0x3a4660, emissive: 0x6a3cff, emissiveIntensity: 0, roughness: 0.4, metalness: 0.6, side: THREE.DoubleSide }));
    const grid = new THREE.Mesh(new THREE.RingGeometry(0.12, 1, 24, 4), new THREE.MeshBasicMaterial({ color: 0xd8c0ff, wireframe: true, transparent: true, opacity: 0.5, depthWrite: false, blending: add }));
    const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: beamGlowTex, color: 0xd0b0ff, blending: add, depthWrite: false, transparent: true, opacity: 0 }));
    for (const o of [core, halo, line, dish, grid, glow]) { o.frustumCulled = false; beamG.add(o); }
    return { A, core, halo, line, dish, grid, glow, src: V3(1, 0, 0), ok: false };
  });
  const waves = new THREE.InstancedMesh(new THREE.TorusGeometry(1, 0.07, 6, 36), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9, depthWrite: false, blending: add }), 8 * NWAVE);
  waves.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(8 * NWAVE * 3), 3); waves.frustumCulled = false; beamG.add(waves);
  const pulsePos = new Float32Array(8 * NPULSE * 3), pulseCol = new Float32Array(8 * NPULSE * 3), pulseGeom = new THREE.BufferGeometry();
  pulseGeom.setAttribute('position', new THREE.BufferAttribute(pulsePos, 3)); pulseGeom.setAttribute('color', new THREE.BufferAttribute(pulseCol, 3));
  const pulses = new THREE.Points(pulseGeom, new THREE.PointsMaterial({ size: 9, sizeAttenuation: false, vertexColors: true, transparent: true, depthWrite: false, blending: add, map: beamGlowTex }));
  pulses.frustumCulled = false; beamG.add(pulses);
  const ringGlow = new THREE.Mesh(new THREE.TorusGeometry(RING_R, RING_TUBE * 2.4, 10, 160), new THREE.MeshBasicMaterial({ color: 0x9f7bff, transparent: true, opacity: 0, depthWrite: false, blending: add }));
  beamG.add(ringGlow);
  // collector belt: fixed sample of the band (heliocentric, AU), re-projected into Earth's sky each frame
  const NB = NBELT_LON * NBELT_LAT, beltHelio = new Float32Array(NB * 3), beltPos = new Float32Array(NB * 3), beltGeom = new THREE.BufferGeometry();
  beltGeom.setAttribute('position', new THREE.BufferAttribute(beltPos, 3));
  const belt = new THREE.Points(beltGeom, new THREE.PointsMaterial({ size: 2.2, sizeAttenuation: false, color: 0xffd27a, transparent: true, opacity: 0.8, depthWrite: false, blending: add }));
  belt.frustumCulled = false; belt.renderOrder = -10; scene.add(belt);
  let beltKey = '';
  function buildBelt(sol) {
    const key = `${sol.orbit_au}|${sol.half_width_deg}`; if (key === beltKey) return; beltKey = key;
    const hw = sol.half_width_deg * D2R;
    for (let i = 0; i < NBELT_LON; i++) for (let j = 0; j < NBELT_LAT; j++) {
      const lon = (i + hash3(i, j, 3)) / NBELT_LON * 2 * Math.PI, lat = Math.asin(Math.sin(hw) * ((j + 0.5) / NBELT_LAT * 2 - 1)), o = 3 * (i * NBELT_LAT + j);
      beltHelio[o] = sol.orbit_au * Math.cos(lat) * Math.cos(lon); beltHelio[o + 1] = sol.orbit_au * Math.cos(lat) * Math.sin(lon); beltHelio[o + 2] = sol.orbit_au * Math.sin(lat);
    }
  }
  const beamI = { v: 0 };       // smoothed visual beam intensity 0..1
  // collectors drawn as small satellites (enlarged): panels face the Sun, wings point to ecliptic north
  const satGeo = satelliteGeometry(), NSKY = 720;
  const skySats = new THREE.InstancedMesh(satGeo, new THREE.MeshBasicMaterial({ vertexColors: true }), NSKY);
  skySats.frustumCulled = false; skySats.renderOrder = -9; scene.add(skySats);
  const srcSats = new THREE.InstancedMesh(satGeo, new THREE.MeshBasicMaterial({ vertexColors: true }), 8);
  srcSats.frustumCulled = false; scene.add(srcSats);
  const srcGlow = arms.map(() => { const g = new THREE.Sprite(new THREE.SpriteMaterial({ map: beamGlowTex, color: 0xc8a8ff, blending: add, depthWrite: false, transparent: true, opacity: 0 })); g.frustumCulled = false; scene.add(g); return g; });
  const _sq = new THREE.Quaternion(), _sv = V3(), _ss = V3(), _sm = new THREE.Matrix4(), NORTH = V3(0, 0, 1);

  // labels (HTML)
  const LBL = {};
  const LCOL = { sat: '#ffd27a', sun: '#ffcc33', earth: '#7cc4ff', moon: '#dddddd', ast: '#e0a46a', arms: '#ff8a7a', mp: '#5ee0ff', bs: '#ffa040', belts: '#ff9a6a',
    l1: '#9be37a', l2: '#9be37a', mars: '#ff8a6a', gmars: '#ff8a6a', hill: '#9be37a', gsun: '#ffcc33', gmoon: '#dddddd', vel: '#7cc4ff', push: '#7fe08a', wind: '#ffd27a', well: '#c8b8ff', field: '#9fd8ff', beam: '#c8a8ff', dish: '#d8c0ff', belt: '#ffd27a' };
  for (const [k, c] of Object.entries(LCOL)) {
    const d = document.createElement('div'); d.className = 'blabel mono small povlbl'; d.style.color = c; d.style.display = 'none'; container.appendChild(d); LBL[k] = d;
  }
  const _lv = V3();
  function label(k, pos, text) {
    const el = LBL[k];
    if (!text || !layers.labels || !pos) { el.style.display = 'none'; return; }
    _lv.copy(pos).project(camera);
    if (_lv.z > 1 || Math.abs(_lv.x) > 1.0 || Math.abs(_lv.y) > 1.0) { el.style.display = 'none'; return; }
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (el.textContent !== text) el.textContent = text;
    el.style.display = 'block';
    const lw = el.offsetWidth, lh = el.offsetHeight;
    const x = (_lv.x * 0.5 + 0.5) * w + 8, y = (-_lv.y * 0.5 + 0.5) * h - 6;
    el.style.left = `${Math.max(2, Math.min(w - lw - 2, x))}px`; el.style.top = `${Math.max(2, Math.min(h - lh - 2, y))}px`;
  }
  function resolveLabels() {
    const placed = [];
    for (const k of Object.keys(LBL)) {
      const el = LBL[k]; if (el.style.display === 'none') continue;
      const x = parseFloat(el.style.left); let y = parseFloat(el.style.top); const w = el.offsetWidth, h = el.offsetHeight;
      for (let it = 0; it < 8; it++) { const hit = placed.find((b) => x < b.x + b.w && x + w > b.x && y < b.y + b.h && y + h > b.y); if (!hit) break; y = hit.y + hit.h + 1; }
      el.style.top = `${y}px`; placed.push({ x, y, w, h });
    }
  }
  function hideLabels() { for (const el of Object.values(LBL)) el.style.display = 'none'; }

  // ---------------------------------------------------------------- builders
  function buildBounds(env) {
    const key = `${env.r0_re.toFixed(3)}|${env.alpha.toFixed(4)}`;
    if (key === boundKey) return; boundKey = key;
    for (const o of mpObjs) { bounds.remove(o); o.geometry.dispose(); }
    mpObjs = [];
    const mk = (r0, al, thMax, color, op) => {
      const pts = []; for (let i = 0; i <= 28; i++) { const th = thMax * i / 28, r = shueR(r0, al, th); pts.push(new THREE.Vector2(r * Math.sin(th) + 1e-4, r * Math.cos(th))); }
      const lathe = new THREE.LatheGeometry(pts, 48);
      const surf = new THREE.Mesh(lathe, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: op, side: THREE.DoubleSide, depthWrite: false }));
      const lp = [];                            // clean wireframe: 12 meridians + rings every 15 degrees
      for (let m = 0; m < 12; m++) { const ph = m / 12 * Math.PI * 2; for (let i = 0; i < pts.length - 1; i++) { const A = pts[i], B = pts[i + 1]; lp.push(A.x * Math.cos(ph), A.y, A.x * Math.sin(ph), B.x * Math.cos(ph), B.y, B.x * Math.sin(ph)); } }
      for (let th = 15 * D2R; th <= thMax + 1e-6; th += 15 * D2R) { const r = shueR(r0, al, th), rho = r * Math.sin(th), y = r * Math.cos(th); for (let j = 0; j < 64; j++) { const a0 = j / 64 * Math.PI * 2, a1 = (j + 1) / 64 * Math.PI * 2; lp.push(rho * Math.cos(a0), y, rho * Math.sin(a0), rho * Math.cos(a1), y, rho * Math.sin(a1)); } }
      const wg = new THREE.BufferGeometry(); wg.setAttribute('position', new THREE.Float32BufferAttribute(lp, 3));
      const wire = new THREE.LineSegments(wg, lineMat(color, 0.32));
      mpObjs.push(surf, wire); bounds.add(surf); bounds.add(wire);
    };
    mk(env.r0_re, env.alpha, 140 * D2R, 0x3fd0ff, 0.06);
    mk(env.rbs_re, env.alpha_bs, 120 * D2R, 0xff9a3a, 0.035);
  }
  function buildField(env) {
    const u = field.userData, pos = u.pos, col = u.col; let n = 0;
    const zm = mhat, xm = V3().crossVectors(zm, P.sunDir); if (xm.lengthSq() < 1e-6) xm.set(1, 0, 0); xm.normalize(); const ym = V3().crossVectors(zm, xm);
    const s = P.sunDir, r0 = env.r0_re, al = env.alpha;
    const inside = (p) => { const r = p.length(); const c = p.dot(s) / r; return r < 0.97 * shueR(r0, al, Math.acos(Math.max(-1, Math.min(1, c)))); };
    const push = (a, b, Ba, Bb) => { if (n + 2 > u.max) return; pos[3 * n] = a.x; pos[3 * n + 1] = a.y; pos[3 * n + 2] = a.z; fieldColor(Ba, col, 3 * n); n++; pos[3 * n] = b.x; pos[3 * n + 1] = b.y; pos[3 * n + 2] = b.z; fieldColor(Bb, col, 3 * n); n++; };
    const p = V3(), q = V3();
    for (const L of [2, 3, 4.5, 6.5, 9, 11.5]) {
      const lm = Math.acos(Math.sqrt(1 / L)), N = 64;
      for (let f = 0; f < 8; f++) {
        const ph = f / 8 * Math.PI * 2, cp = Math.cos(ph), sp = Math.sin(ph);
        let prevIn = false, prevB = 0;
        for (let j = 0; j <= N; j++) {
          const lam = -lm + 2 * lm * j / N, r = L * Math.cos(lam) ** 2;
          q.copy(xm).multiplyScalar(r * Math.cos(lam) * cp).addScaledVector(ym, r * Math.cos(lam) * sp).addScaledVector(zm, r * Math.sin(lam));
          const isIn = inside(q), B = dipoleB(r, lam);
          if (isIn && prevIn) push(p, q, prevB, B);
          p.copy(q); prevIn = isIn; prevB = B;
        }
      }
    }
    // open (tail) lines from the polar caps, night side only: schematic
    const tail = V3().copy(s).negate();
    for (const hemi of [1, -1]) {
      for (let f = 0; f < 8; f++) {
        const ph = f / 8 * Math.PI * 2, cp = Math.cos(ph), sp = Math.sin(ph);
        const horiz = V3().copy(xm).multiplyScalar(cp).addScaledVector(ym, sp);
        if (horiz.dot(tail) < -0.2) continue;
        const lat0 = hemi * (73 + 2 * (f % 3)) * D2R, Lc = 1 / Math.cos(lat0) ** 2;
        let prevB = 0, first = true, lam = lat0;
        for (let j = 0; j < 40; j++) {         // dipole part until 0.55 r0
          lam = lat0 - hemi * j * 0.02; const r = Lc * Math.cos(lam) ** 2;
          q.copy(horiz).multiplyScalar(r * Math.cos(lam)).addScaledVector(zm, r * Math.sin(lam));
          const B = dipoleB(r, lam);
          if (!first) push(p, q, prevB, B);
          p.copy(q); prevB = B; first = false;
          if (r > 0.55 * r0) break;
        }
        const x0 = p.dot(tail), lobe = V3().copy(p).addScaledVector(tail, -x0); const off = lobe.length(); lobe.normalize();
        for (let k = 1; k <= 26; k++) {        // stretch anti-sunward, slowly flaring (schematic)
          const d = k * 3.2, rad = off * (1 + 0.012 * d);
          q.copy(tail).multiplyScalar(x0 + d).addScaledVector(lobe, rad);
          const B = prevB * 0.93;
          push(p, q, prevB, B); p.copy(q); prevB = B;
        }
      }
    }
    field.geometry.attributes.position.needsUpdate = true; field.geometry.attributes.color.needsUpdate = true; field.geometry.setDrawRange(0, n);
  }
  function buildWell(env) {
    const u = well.userData, pos = u.pos, col = u.col; let n = 0;
    const rH = env.hill_re, R = 1.6 * rH, ex = V3().copy(P.sunDir).negate(), ey = V3().crossVectors(P.nrm, ex), ez = P.nrm;
    const muM = P.moonPos ? 7.3458e22 / (P.mSys - 7.3458e22) : 0;
    const mx = P.moonPos ? P.moonPos.dot(ex) : 0, my = P.moonPos ? P.moonPos.dot(ey) : 0;
    const phiL1 = -1.5 / rH, A = 0.11 * rH;
    const phi = (x, y) => { const r = Math.max(1, Math.hypot(x, y)); let f = -1 / r - 0.5 * x * x / (rH ** 3); if (muM) f -= muM / Math.max(MOON_R, Math.hypot(x - mx, y - my)); return f; };
    const H = (x, y) => { const d = phi(x, y) - phiL1; return A * Math.sign(d) * Math.log(1 + Math.abs(d) / Math.abs(phiL1)); };
    const h0 = H(1, 0), off = -h0 - 1.15;                     // Earth sits at the bottom of its well
    const put = (x, y) => {
      const h = H(x, y) + off;
      pos[3 * n] = ex.x * x + ey.x * y + ez.x * h; pos[3 * n + 1] = ex.y * x + ey.y * y + ez.y * h; pos[3 * n + 2] = ex.z * x + ey.z * y + ez.z * h;
      const t = Math.max(0, Math.min(1, (h - off) / (-h0)));  // 0 rim .. -1 bottom
      const k = Math.max(0, Math.min(1, -(h - off) / (-h0)));
      col[3 * n] = 0.45 + 0.35 * (1 - k) + 0 * t; col[3 * n + 1] = 0.4 + 0.45 * (1 - k); col[3 * n + 2] = 0.95; n++;
    };
    for (let dir = 0; dir < 2; dir++) for (let i = 0; i <= GW_N; i++) {
      const c = -R + 2 * R * i / GW_N;
      for (let j = 0; j < GW_S; j++) {
        const s0 = -R + 2 * R * j / GW_S, s1 = -R + 2 * R * (j + 1) / GW_S;
        if (n + 2 > u.max) break;
        if (dir === 0) { put(c, s0); put(c, s1); } else { put(s0, c); put(s1, c); }
      }
    }
    well.geometry.attributes.position.needsUpdate = true; well.geometry.attributes.color.needsUpdate = true; well.geometry.setDrawRange(0, n);
    // drop line from the Moon to its dimple
    const dp = wellDrop.userData.pos;
    if (P.moonPos) {
      const h = H(mx, my) + off;
      dp.set([P.moonPos.x, P.moonPos.y, P.moonPos.z, ex.x * mx + ey.x * my + ez.x * h, ex.y * mx + ey.y * my + ez.y * h, ex.z * mx + ey.z * my + ez.z * h]);
      wellDrop.geometry.attributes.position.needsUpdate = true; wellDrop.geometry.setDrawRange(0, 2);
    } else wellDrop.geometry.setDrawRange(0, 0);
    P.wellLabelPos = V3().copy(ey).multiplyScalar(R * 0.95).addScaledVector(ez, H(0, R * 0.95) + off);
  }
  function drawMoonOrbit(ms, mu) {
    const g = moonOrbit.geometry, pos = moonOrbit.userData.pos;
    if (!ms) { g.setDrawRange(0, 0); return; }
    const r = ms.r, v = ms.v, rm = Math.hypot(...r), v2 = v[0] ** 2 + v[1] ** 2 + v[2] ** 2;
    const h = [r[1] * v[2] - r[2] * v[1], r[2] * v[0] - r[0] * v[2], r[0] * v[1] - r[1] * v[0]], hm = Math.hypot(...h);
    const eps = v2 / 2 - mu / rm; if (eps >= 0) { g.setDrawRange(0, 0); return; }
    const a = -mu / (2 * eps);
    const ev = [(v[1] * h[2] - v[2] * h[1]) / mu - r[0] / rm, (v[2] * h[0] - v[0] * h[2]) / mu - r[1] / rm, (v[0] * h[1] - v[1] * h[0]) / mu - r[2] / rm];
    const e = Math.hypot(...ev), Pp = e > 1e-6 ? ev.map((x) => x / e) : r.map((x) => x / rm), hh = h.map((x) => x / hm);
    const Q = [hh[1] * Pp[2] - hh[2] * Pp[1], hh[2] * Pp[0] - hh[0] * Pp[2], hh[0] * Pp[1] - hh[1] * Pp[0]], b = a * Math.sqrt(Math.max(0, 1 - e * e));
    for (let j = 0; j <= 128; j++) {
      const E = j / 128 * 2 * Math.PI, x = a * (Math.cos(E) - e), y = b * Math.sin(E);
      for (let c = 0; c < 3; c++) pos[3 * j + c] = (x * Pp[c] + y * Q[c]) / RE_KM;
    }
    g.attributes.position.needsUpdate = true; g.setDrawRange(0, 129);
  }

  // ---------------------------------------------------------------- arms
  const e1 = V3(), e2 = V3();
  function bez(p0, p1, p2, p3, t, out) {
    const u = 1 - t; return out.set(0, 0, 0).addScaledVector(p0, u * u * u).addScaledVector(p1, 3 * u * u * t).addScaledVector(p2, 3 * u * t * t).addScaledVector(p3, t * t * t);
  }
  const _P1 = V3(), _P2 = V3(), _T = V3(), _dir = V3(), _rad = V3();
  function setInst(mesh, idx, pos, dir, sx, sy, sz) { _q.setFromUnitVectors(UPY, _dir.copy(dir).normalize()); _s.set(sx, sy, sz); _m.compose(pos, _q, _s); mesh.setMatrixAt(idx, _m); }
  function updateArms(wall, dt, astPos, astR, frozenMotion) {
    e1.copy(P.sunDir).negate(); e2.crossVectors(P.nrm, e1);
    ring.quaternion.setFromUnitVectors(UPZ, P.nrm);
    for (let s = 0; s < 4; s++) {          // struts from the surface to the ring
      const a = (s * 90 + 22.5) * D2R; _a.copy(e1).multiplyScalar(Math.cos(a)).addScaledVector(e2, Math.sin(a));
      _b.copy(_a).multiplyScalar((1.0 + RING_R) / 2); setInst(struts, s, _b, _a, 0.035, RING_R - 0.98, 0.035);
    }
    struts.instanceMatrix.needsUpdate = true;
    let held = false;
    // ---- joint throw: all 8 arms close around ONE rock and fling it together (one packet per throw)
    if (group.phase >= 1 && group.pending.length) { group.phase = 0; group.released = false; }
    const gOn = group.phase < 1;
    let gph = 1;
    if (gOn) {
      group.phase = Math.min(1, group.phase + dt / GROUP_T); gph = group.phase;
      const eh = group.eh;
      // grab point: on the ring axis, under the asteroid (the same distance from all 8 roots); release point: as far
      // along the throw direction as ALL 8 arms (length ARM_L = 2 R_E) can still hold the packet
      group.G.copy(P.nrm).multiplyScalar(GRAB_Z - ROCK_R * 0.6);
      if (!group.roots) group.roots = [...Array(8)].map(() => V3());
      for (let k = 0; k < 8; k++) { const a = k * 45 * D2R; group.roots[k].copy(e1).multiplyScalar(Math.cos(a)).addScaledVector(e2, Math.sin(a)).multiplyScalar(RING_R + 0.05); }
      group.stroke = strokeLength(group.G, eh, group.roots);
      group.F.copy(group.G).addScaledVector(eh, Math.max(0.05, group.stroke - ROCK_R));   // release point
      group.u.copy(Math.abs(eh.dot(P.nrm)) < 0.9 ? P.nrm : e1).cross(eh).normalize(); group.w.crossVectors(eh, group.u);
      if (gph < 0.45) group.rock.copy(group.G);
      else if (gph < 0.66) { const k = (gph - 0.45) / 0.21; group.rock.lerpVectors(group.G, group.F, k * k); }
      else group.rock.copy(group.F);
      held = gph > 0.38 && gph < 0.66;
      if (gph >= 0.66 && !group.released) { group.released = true; for (const e of group.pending.splice(0)) startPacket(group.F, e); }
    }
    for (const A of arms) {
      const ang = A.k * 45 * D2R; A.dir.copy(e1).multiplyScalar(Math.cos(ang)).addScaledVector(e2, Math.sin(ang));
      A.base.copy(A.dir).multiplyScalar(RING_R + 0.05);
      _b.copy(A.dir).multiplyScalar(RING_R + 0.02); setInst(hubs, A.k, _b, A.dir, 1, 1, 1);
      // rest pose: reaching outward with an idle sway (illustration)
      const sw = frozenMotion ? 0 : wall;
      _T.copy(A.base).addScaledVector(A.dir, 1.25 + 0.12 * Math.sin(0.9 * sw + A.k)).addScaledVector(P.nrm, (A.k % 2 ? -0.7 : 0.7) + 0.18 * Math.sin(0.7 * sw + 1.3 * A.k))
        .addScaledVector(_a.crossVectors(P.nrm, A.dir), 0.25 * Math.sin(0.5 * sw + 2.1 * A.k));
      A.rest = (A.rest || V3()).copy(_T);
      let open = 0.35;
      if (gOn) {
        // each claw takes its own place on a cage around the rock (8 directions around the throw axis)
        const th = A.k * 45 * D2R, cage = _a.copy(group.u).multiplyScalar(Math.cos(th)).addScaledVector(group.w, Math.sin(th)).multiplyScalar(ROCK_R * 1.9);
        const grabT = _P1.copy(group.G).add(cage).addScaledVector(group.eh, -ROCK_R * 0.4);
        const flingT = _P2.copy(group.F).addScaledVector(cage, 1.15).addScaledVector(group.eh, -ROCK_R * 0.4);
        const rest = _b.copy(_T), sm = (k) => k * k * (3 - 2 * k);
        if (gph < 0.36) { _T.lerpVectors(rest, grabT, sm(gph / 0.36)); open = 0.75; }
        else if (gph < 0.45) { _T.copy(grabT); open = 0.75 - 0.7 * (gph - 0.36) / 0.09; }
        else if (gph < 0.66) { const k = (gph - 0.45) / 0.21; _T.lerpVectors(grabT, flingT, k * k); open = 0.05; }
        else { const k = (gph - 0.66) / 0.34; _T.lerpVectors(flingT, rest, sm(k)); open = 1.0 - 0.65 * k; }
      }
      A.open = open;
      // every arm has the same length ARM_L (2 Earth radii): it bends away from Earth when the claw is closer
      const sg = A.k % 2 ? -1 : 1;
      armCurve(A.base, _T, _rad.copy(P.nrm).multiplyScalar(sg), A.pts, NSEG);
      for (let j = 0; j < NSEG; j++) {
        const a = A.pts[j], b = A.pts[j + 1]; _a.addVectors(a, b).multiplyScalar(0.5); _rad.subVectors(b, a);
        const r = 0.075 - 0.032 * j / NSEG; setInst(segs, A.k * NSEG + j, _a, _rad, r, _rad.length() * 1.04, r);
      }
      for (let j = 0; j < 5; j++) { const idx = Math.min(NSEG, Math.round((j + 1) * NSEG / 5.5)); _s.setScalar(0.078 - 0.006 * j); _m.compose(A.pts[idx], _q.identity(), _s); joints.setMatrixAt(A.k * 5 + j, _m); }
      _rad.subVectors(A.pts[NSEG], A.pts[NSEG - 1]).normalize(); A.tip.copy(A.pts[NSEG]);
      const side = _a.crossVectors(_rad, P.nrm); if (side.lengthSq() < 1e-6) side.set(1, 0, 0); side.normalize(); const side2 = _b.crossVectors(_rad, side);
      for (let c = 0; c < 3; c++) {
        const th = c * 2.094, out = V3().copy(side).multiplyScalar(Math.cos(th)).addScaledVector(side2, Math.sin(th));
        const pd = V3().copy(_rad).multiplyScalar(Math.cos(A.open)).addScaledVector(out, Math.sin(A.open));
        const pp = V3().copy(A.tip).addScaledVector(pd, 0.17);
        setInst(claws, A.k * 3 + c, pp, pd, 1, 1, 1);
      }
    }
    if (held) { heldRock.position.copy(group.rock); heldRock.rotation.x += dt * 2.5; heldRock.rotation.y += dt * 1.7; }
    heldRock.visible = held && layers.arms;
    for (const o of [hubs, segs, joints, claws]) o.instanceMatrix.needsUpdate = true;
  }
  /**
   * Space mirrors: ring in the plane through Earth perpendicular to the sunlight (a "terminator orbit", never in
   * Earth's shadow). A mirror's normal bisects the directions to the Sun and to its target. The share of mirrors aimed
   * at Earth = recorded mirror light / mirror capacity; the rest send their light past Earth.
   */
  const _mb1 = V3(), _mb2 = V3(), _mp = V3(), _mt = V3(), _mn = V3(), _mlook = new THREE.Matrix4(), _mup = V3();
  function updateMirrorRing(S, wall) {
    const c = S.climate, mirRel = c ? c.mirRel || 0 : 0, cap = c && c.cl && c.cl.mirrors ? c.cl.mirrors.cap_rel || 1 : 1;
    const capNow = c && c.dec ? c.dec.mirror_cap_rel || cap : cap;
    const on = layers.mirrors && !!(c && c.cl && c.cl.mirrors && c.cl.mirrors.on);
    mirrorRing.visible = on; P.mirrors = { on, mirRel, capNow, T: c ? c.T : NaN };
    for (const l of mirrorLights) l.intensity = 0;
    if (!on) return;
    _mb1.crossVectors(P.nrm, P.sunDir); if (_mb1.lengthSq() < 1e-6) _mb1.set(0, 1, 0); _mb1.normalize(); _mb2.crossVectors(P.sunDir, _mb1);
    mirrorOrbitLine.quaternion.setFromUnitVectors(UPZ, P.sunDir);
    const aimed = Math.max(0, Math.min(1, mirRel / Math.max(1e-9, capNow))), nAimed = Math.round(aimed * MIR_N);
    const ud = mirrorRays.userData; let nv = 0;
    for (let k = 0; k < MIR_N; k++) {
      const a = 2 * Math.PI * k / MIR_N + 0.01 * wall;
      _mp.copy(_mb1).multiplyScalar(Math.cos(a) * MIR_ORBIT).addScaledVector(_mb2, Math.sin(a) * MIR_ORBIT);
      const hits = ((k * 7) % MIR_N) < nAimed;
      _mt.copy(_mp).negate().normalize();                                     // toward Earth
      if (!hits) _mt.multiplyScalar(-0.25).addScaledVector(P.sunDir, -1).normalize();   // past Earth, away from the Sun
      _mn.copy(P.sunDir).add(_mt).normalize();
      _mup.copy(Math.abs(_mn.dot(P.nrm)) < 0.95 ? P.nrm : _mb1);
      _mlook.lookAt(V3(), _mn, _mup); _q.setFromRotationMatrix(_mlook); _s.setScalar(1); _m.compose(_mp, _q, _s); mirrorPanels.setMatrixAt(k, _m);
      // incoming sunlight onto the mirror (fades in from the Sun side) ...
      const LIN = 9;
      ud.pos.set([_mp.x + P.sunDir.x * LIN, _mp.y + P.sunDir.y * LIN, _mp.z + P.sunDir.z * LIN, _mp.x, _mp.y, _mp.z], 3 * nv);
      ud.col.set([0, 0, 0, 0.55, 0.55, 0.52], 3 * nv); nv += 2;
      // ... and its reflection: down to Earth's surface when aimed, faint and short when turned past Earth
      const L = hits ? MIR_ORBIT - 1.0 : 5;
      ud.pos.set([_mp.x, _mp.y, _mp.z, _mp.x + _mt.x * L, _mp.y + _mt.y * L, _mp.z + _mt.z * L], 3 * nv);
      const b = hits ? 0.95 : 0.12; ud.col.set([b, b, b * 0.94, b, b, b * 0.94], 3 * nv); nv += 2;
    }
    mirrorPanels.instanceMatrix.needsUpdate = true;
    mirrorRays.geometry.attributes.position.needsUpdate = true; mirrorRays.geometry.attributes.color.needsUpdate = true; mirrorRays.geometry.setDrawRange(0, nv);
    mirrorRays.material.opacity = 0.18 + 0.27 * Math.min(1, Math.sqrt(mirRel));
    const I = 1.6 * Math.sqrt(Math.min(1.5, mirRel));
    [_mb1, _mb2].forEach((b, j) => { for (const sg of [1, -1]) { const l = mirrorLights[2 * j + (sg > 0 ? 0 : 1)]; l.position.copy(b).multiplyScalar(30 * sg); l.target.position.set(0, 0, 0); l.intensity = I * 0.5; } });
  }
  function startPacket(from, e) {
    let s = pk.find((x) => !x.on); if (!s) s = pk.reduce((a, b) => (a.age > b.age ? a : b));
    Object.assign(s, { on: true, age: 0, from: V3().copy(from), dir: V3(...e.ehat).normalize(), fate: e.fate, u: e.u });
  }
  function updatePackets(dt) {
    const u = pkLines.userData; let n = 0, nh = 0;
    for (const s of pk) {
      if (!s.on) continue; s.age += dt; if (s.age > 1.4) { s.on = false; continue; }
      const fade = Math.max(0, 1 - s.age / 1.4), c = PKCOL[s.fate] || [1, 1, 1];
      let far;
      if (s.fate === 3) far = Math.min(6, 2 + s.age * 20);                 // fell back: short arc
      else if (s.fate === 4 && P.moonPos) far = Math.min(P.moonPos.length(), s.age * 600);
      else far = Math.min(600, s.age * 300);
      const near = Math.max(0, far - 60);
      const a = V3().copy(s.from).addScaledVector(s.dir, near), b = V3().copy(s.from).addScaledVector(s.dir, far);
      u.pos.set([a.x, a.y, a.z, b.x, b.y, b.z], 3 * n); u.col.set([c[0] * fade * 0.25, c[1] * fade * 0.25, c[2] * fade * 0.25, c[0] * fade, c[1] * fade, c[2] * fade], 3 * n); n += 2;
      pkHeadPos.set([b.x, b.y, b.z], 3 * nh); pkHeadCol.set([c[0] * fade, c[1] * fade, c[2] * fade], 3 * nh); nh++;
    }
    pkHeadGeom.attributes.position.needsUpdate = true; pkHeadGeom.attributes.color.needsUpdate = true; pkHeadGeom.setDrawRange(0, nh);
    pkLines.geometry.attributes.position.needsUpdate = true; pkLines.geometry.attributes.color.needsUpdate = true; pkLines.geometry.setDrawRange(0, n);
  }

  // ---------------------------------------------------------------- solar wind
  function updateWind(env, dt, animate) {
    const N = Math.round(Math.max(60, Math.min(NW, 1600 * env.n_cm3 / 5)));
    const r0 = env.r0_re, al = env.alpha, rb = env.rbs_re, ab = env.alpha_bs;
    const s = P.sunDir, ez = P.nrm, ey = _b.crossVectors(ez, s);
    for (let i = 0; i < N; i++) {
      if (animate) {
        let x = wx[i], y = wy[i], z = wz[i];
        const r = Math.hypot(x, y, z), th = Math.acos(Math.max(-1, Math.min(1, x / r)));
        const rbs = shueR(rb, ab, Math.min(th, 2.6)), rmp = shueR(r0, al, Math.min(th, 2.6));
        if (r > rbs || th > 2.6) { x -= WIND_V * dt; wSheath[i] = r > rbs ? 0 : wSheath[i]; }
        else {                                // magnetosheath: slowed, flowing along the boundary
          wSheath[i] = 1;
          const nx = x / r, ny = y / r, nz = z / r, d = -nx;     // flow (-1,0,0) minus normal part
          let tx = -1 - d * nx, ty = -d * ny, tz = -d * nz; const tm = Math.hypot(tx, ty, tz) || 1;
          if (tm < 1e-3) { tx = 0; ty = 1; tz = 0; }
          const sp = WIND_V * (0.3 + 0.6 * th / Math.PI);
          x += sp * dt * tx / tm; y += sp * dt * ty / tm; z += sp * dt * tz / tm;
          const r2 = Math.hypot(x, y, z), th2 = Math.acos(Math.max(-1, Math.min(1, x / r2))), lim = 1.04 * shueR(r0, al, Math.min(th2, 2.6));
          if (r2 < lim) { const k = lim / r2; x *= k; y *= k; z *= k; }
          void rmp;
        }
        if (x < -100 || Math.abs(y) > 70 || Math.abs(z) > 70) { x = 70; y = (Math.random() * 2 - 1) * 70; z = (Math.random() * 2 - 1) * 70; wSheath[i] = 0; }
        wx[i] = x; wy[i] = y; wz[i] = z;
      }
      windPos[3 * i] = s.x * wx[i] + ey.x * wy[i] + ez.x * wz[i]; windPos[3 * i + 1] = s.y * wx[i] + ey.y * wy[i] + ez.y * wz[i]; windPos[3 * i + 2] = s.z * wx[i] + ey.z * wy[i] + ez.z * wz[i];
      if (wSheath[i]) { windCol[3 * i] = 1.0; windCol[3 * i + 1] = 0.5; windCol[3 * i + 2] = 0.25; } else { windCol[3 * i] = 1.0; windCol[3 * i + 1] = 0.86; windCol[3 * i + 2] = 0.45; }
    }
    windGeom.attributes.position.needsUpdate = true; windGeom.attributes.color.needsUpdate = true; windGeom.setDrawRange(0, N);
  }

  // ---------------------------------------------------------------- camera helpers
  function updateBeams(S, p, r, dt, wall) {
    const pw = S.power, sol = S.solar, on = !!(pw && pw.on && sol && sol.on);
    P.power = on ? pw : null; P.solar = on ? sol : null;
    // collector belt in Earth's sky (true directions; distances scaled like the Sun's)
    belt.visible = on && layers.collectors;
    if (on) buildBelt(sol);
    const kSc = SUN_D / r;
    if (belt.visible) {
      for (let i = 0; i < NB; i++) {
        const o = 3 * i, x = beltHelio[o] - p.x, y = beltHelio[o + 1] - p.y, z = beltHelio[o + 2] - p.z, d = Math.hypot(x, y, z), k = Math.min(kSc, 55000 / d);
        beltPos[o] = x * k; beltPos[o + 1] = y * k; beltPos[o + 2] = z * k;
      }
      beltGeom.attributes.position.needsUpdate = true;
      for (let n = 0; n < NSKY; n++) {
        const i = Math.floor(n * NB / NSKY), o = 3 * i, d = Math.hypot(beltPos[o], beltPos[o + 1], beltPos[o + 2]);
        _sv.set(-beltHelio[o], -beltHelio[o + 1], -beltHelio[o + 2]); satQuat(_sv, NORTH, _sq);
        _ss.setScalar(d * 0.0013); _sv.set(beltPos[o], beltPos[o + 1], beltPos[o + 2]); _sm.compose(_sv, _sq, _ss); skySats.setMatrixAt(n, _sm);
      }
      skySats.instanceMatrix.needsUpdate = true;
    }
    skySats.visible = belt.visible;
    beamG.visible = on && layers.beams;
    srcSats.visible = beamG.visible;
    for (const g of srcGlow) g.visible = beamG.visible;
    // visual intensity: bright while the launcher draws beam power in the current event, faint (pilot) when idle
    const offered = on ? pw.avail / Math.max(1e-9, sol.rx) : 0, frac = on && offered > 0 ? Math.min(1, pw.inW / offered) : 0;
    // stand-by (no launch in this event): a pilot beam stays visible so the beams can always be seen
    const target = !on ? 0 : pw.inW > 0 ? 0.45 + 0.55 * Math.min(1, Math.sqrt(frac / 0.2)) : (P.beamView ? 0.4 : 0.25);
    P.beamStandby = on && !(pw.inW > 0);
    beamI.v += (target - beamI.v) * Math.min(1, dt * 4); P.beamFrac = frac; P.beamI = beamI.v;
    if (!beamG.visible) return;
    // source direction per arm: a real belt direction near the belt's edge on that arm's side, with a clear line of sight to the dish
    if (on && (!P.beamSelWall || wall - P.beamSelWall > 0.5 || P.beamSelR !== r)) {
      buildBelt(sol); P.beamSelWall = wall; P.beamSelR = r;
      const el = r > sol.orbit_au ? Math.asin(sol.orbit_au / r) : Math.PI / 2, e2s = V3().crossVectors(P.nrm, P.sunDir);
      for (const b of beams) {
        const A = b.A, perp = V3().copy(A.dir).addScaledVector(P.sunDir, -A.dir.dot(P.sunDir)).addScaledVector(P.nrm, -A.dir.dot(P.nrm));
        if (perp.length() < 0.3) perp.copy(e2s).multiplyScalar(A.k < 4 ? -1 : 1); perp.normalize();
        const want = V3().copy(P.sunDir).multiplyScalar(Math.cos(0.8 * el)).addScaledVector(perp, Math.sin(0.8 * el)).addScaledVector(P.nrm, A.k % 2 ? -0.12 : 0.12).normalize();
        let best = -2, bi = -1;
        for (let i = 0; i < NB; i += 3) {
          const o = 3 * i; _a.set(beltHelio[o] - p.x, beltHelio[o + 1] - p.y, beltHelio[o + 2] - p.z).normalize();
          const tdot = A.tip.dot(_a), perpD = Math.sqrt(Math.max(0, A.tip.lengthSq() - tdot * tdot));
          if (tdot < 0 && perpD < 1.08) continue;                // Earth would block this beam
          const sc = _a.dot(want); if (sc > best) { best = sc; bi = i; }
        }
        b.ok = bi >= 0;
        if (b.ok) {
          b.src.set(beltHelio[3 * bi] - p.x, beltHelio[3 * bi + 1] - p.y, beltHelio[3 * bi + 2] - p.z);
          const dd = b.src.length(); b.srcDist = dd * Math.min(kSc, 55000 / dd); b.src.normalize();
          b.srcSun = (b.srcSun || V3()).set(-beltHelio[3 * bi], -beltHelio[3 * bi + 1], -beltHelio[3 * bi + 2]);
        }
      }
    }
    const I = beamI.v, rb = (sol.dr_km / 2) / RE_KM, wcol = new THREE.Color(), base = new THREE.Color(0xd2b8ff);
    // keep the beam at least ~3 px wide on screen when zoomed out (the true 1,000 km width would vanish)
    const pxU = 2 * Math.tan(camera.fov * D2R / 2) / Math.max(1, canvas.clientHeight);
    const rbVis = Math.max(rb, camera.position.distanceTo(arms[0].tip) * pxU * 1.6); P.beamEnlarged = rbVis > rb * 1.05;
    const phase = wall * 0.55;                                   // wavefront animation (visual speed)
    for (const b of beams) {
      const vis = b.ok; b.core.visible = b.halo.visible = b.line.visible = b.glow.visible = vis;
      b.dish.visible = b.grid.visible = layers.arms || vis;
      const T = b.A.tip, s = b.src;
      _b.copy(T).addScaledVector(s, BEAM_LEN / 2);
      _q.setFromUnitVectors(UPY, s);
      b.core.position.copy(_b); b.core.quaternion.copy(_q); b.core.scale.set(rbVis, BEAM_LEN, rbVis); b.core.material.opacity = 0.65 * I;
      b.halo.position.copy(_b); b.halo.quaternion.copy(_q); b.halo.scale.set(rbVis * 1.9, BEAM_LEN, rbVis * 1.9); b.halo.material.opacity = 0.16 * I;
      const lp = b.line.userData.pos; lp[0] = T.x; lp[1] = T.y; lp[2] = T.z; lp[3] = T.x + s.x * 40000; lp[4] = T.y + s.y * 40000; lp[5] = T.z + s.z * 40000;
      b.line.geometry.attributes.position.needsUpdate = true; b.line.geometry.setDrawRange(0, 2); b.line.material.opacity = 0.15 + 0.5 * I;
      _q.setFromUnitVectors(UPZ, s);
      b.dish.position.copy(T).addScaledVector(s, 0.03); b.dish.quaternion.copy(_q); b.dish.scale.setScalar(rb); b.dish.material.emissiveIntensity = 2.5 * I;
      b.grid.position.copy(b.dish.position).addScaledVector(s, 0.002); b.grid.quaternion.copy(_q); b.grid.scale.setScalar(rb); b.grid.material.opacity = 0.25 + 0.6 * I;
      b.glow.position.copy(b.dish.position); b.glow.scale.setScalar(rb * (1.6 + 2.6 * I)); b.glow.material.opacity = Math.min(1, 1.1 * I);
      // the collector satellite that sends this beam (enlarged ~15x; real collector ~0.05 deg across)
      if (vis && b.srcSun) {
        _sv.copy(T).addScaledVector(s, b.srcDist); satQuat(b.srcSun, NORTH, _sq); _ss.setScalar(b.srcDist * 0.0042); _sm.compose(_sv, _sq, _ss);
        srcGlow[b.A.k].position.copy(_sv).addScaledVector(s, -b.srcDist * 0.004); srcGlow[b.A.k].scale.setScalar(b.srcDist * 0.02 * (0.6 + I)); srcGlow[b.A.k].material.opacity = 0.25 + 0.6 * I;
        b.satPos = (b.satPos || V3()).copy(_sv);
      } else { _ss.setScalar(0); _sm.compose(_sv.set(0, 0, 0), _sq.identity(), _ss); srcGlow[b.A.k].material.opacity = 0; }
      srcSats.setMatrixAt(b.A.k, _sm);
      for (let j = 0; j < NWAVE; j++) {
        const u = (j / NWAVE + phase) % 1, d = (1 - u) * WAVE_SPAN, idx = b.A.k * NWAVE + j;
        _a.copy(T).addScaledVector(s, d); _s.setScalar(vis ? rb * 1.35 : 0); _m.compose(_a, _q, _s); waves.setMatrixAt(idx, _m);
        const fade = Math.min(1, d / 1.2) * Math.min(1, (WAVE_SPAN - d) / 3) * I;
        wcol.copy(base).multiplyScalar(fade); waves.setColorAt(idx, wcol);
      }
      // energy pulses running down the arm from the dish to the ring (rate grows with the power drawn)
      for (let j = 0; j < NPULSE; j++) {
        const u = (j / NPULSE + wall * (0.35 + 1.2 * P.beamFrac)) % 1, f = (1 - u) * NSEG, i0 = Math.floor(f), i1 = Math.min(NSEG, i0 + 1);
        _a.lerpVectors(b.A.pts[i0], b.A.pts[i1], f - i0); const o = 3 * (b.A.k * NPULSE + j);
        pulsePos[o] = _a.x; pulsePos[o + 1] = _a.y; pulsePos[o + 2] = _a.z;
        const c = layers.arms && pw.inW > 0 ? I : 0; pulseCol[o] = 0.85 * c; pulseCol[o + 1] = 0.75 * c; pulseCol[o + 2] = 1.0 * c;
      }
    }
    srcSats.instanceMatrix.needsUpdate = true;
    waves.instanceMatrix.needsUpdate = true; waves.instanceColor.needsUpdate = true;
    pulseGeom.attributes.position.needsUpdate = true; pulseGeom.attributes.color.needsUpdate = true;
    ringGlow.visible = layers.arms; ringGlow.quaternion.copy(ring.quaternion); ringGlow.material.opacity = pw.inW > 0 ? 0.45 * I : 0;
  }
  function beamCam(dt) {
    // beam view: the camera rides with one arm tip
    if (P.beamView) {
      const A = arms[P.beamArm];
      if (!P.beamCamInit) {
        const anc = V3().copy(A.base).addScaledVector(A.dir, 1.1);
        P.beamCamInit = true; P.camAnchor.copy(anc);
        const s = beams[P.beamArm].src, off = V3().copy(A.dir).multiplyScalar(1.0).addScaledVector(P.nrm, 0.6).addScaledVector(s, -1.0).normalize().multiplyScalar(3.4);
        controls.target.copy(anc); camera.position.copy(anc).add(off); controls.update();
      } else {
        const old = V3().copy(P.camAnchor); P.camAnchor.lerp(_b.copy(A.base).addScaledVector(A.dir, 1.1), Math.min(1, dt * 3));
        camera.position.add(_a.subVectors(P.camAnchor, old)); controls.target.copy(P.camAnchor);
      }
    }
  }
  function setBeamView(on) {
    if (on === !!P.beamView) return;
    P.beamView = on;
    if (on) {
      P.savedLayers = { ...layers };
      Object.assign(layers, { arms: true, beams: true, collectors: true, labels: true, moon: true, packets: true, field: false, magnetopause: false, belts: false,
        wind: false, gravwell: false, hill: false, gravvec: false, motion: false, moonorbit: false });
      // ride on the side arm farthest from the current throw direction (so the throw animation does not swing the camera around)
      const cand = [2, 6].map((k) => [k, arms[k].dir.dot(P.ehNow || V3(1, 0, 0))]).sort((a, b) => a[1] - b[1]);
      P.beamArm = cand[0][0]; P.beamCamInit = false; P.camAnchor = V3();
      controls.minDistance = 0.12; P.lockSunSaved = P.lockSun;
    } else {
      if (P.savedLayers) Object.assign(layers, P.savedLayers);
      controls.minDistance = 1.3; controls.target.set(0, 0, 0); look('oblique', 9);
    }
  }
  function viewDir(kind) {
    const s = P.sunDir, n = P.nrm, side = V3().crossVectors(n, s);
    switch (kind) {
      case 'sun': return V3().copy(s).negate().multiplyScalar(1).addScaledVector(n, 0.28).addScaledVector(side, 0.12).normalize();       // behind Earth, looking at the Sun
      case 'side': return V3().copy(side).addScaledVector(n, 0.25).normalize();
      case 'motion': return V3().copy(P.vhat).negate().addScaledVector(n, 0.3).normalize();
      case 'top': return V3().copy(n).addScaledVector(side, 0.02).normalize();
      case 'moon': { if (!P.moonPos) return V3().copy(side); const m = V3().copy(P.moonPos).normalize(), lat = V3().crossVectors(n, m).normalize(); return m.negate().addScaledVector(lat, 0.5).addScaledVector(n, 0.3).normalize(); }
      default: return V3().copy(side).multiplyScalar(0.8).addScaledVector(s, -0.45).addScaledVector(n, 0.42).normalize();   // oblique: day and night side
    }
  }
  function look(kind, dist) {
    if (P.beamView) { P.beamCamInit = false; return; }      // beam view: 'look' buttons re-centre on the arm tip
    const d = dist ?? Math.max(controls.minDistance, camera.position.length());
    camera.position.copy(viewDir(kind)).multiplyScalar(kind === 'moon' ? Math.max(d, 16) : d);
    controls.target.set(0, 0, 0); controls.update();
    if (kind === 'moon' && P.moonPos) {      // choose the side that puts the Moon in the right half (the HUD covers the left)
      const best = [];
      for (const sg of [1, -1]) {
        const m = V3().copy(P.moonPos).normalize(), lat = V3().crossVectors(P.nrm, m).normalize();
        camera.position.copy(m.negate().addScaledVector(lat, 0.5 * sg).addScaledVector(P.nrm, 0.3).normalize().multiplyScalar(Math.max(d, 16)));
        camera.lookAt(0, 0, 0); camera.updateMatrixWorld();
        best.push([sg, V3().copy(P.moonPos).project(camera).x, camera.position.clone()]);
      }
      best.sort((a, b) => b[1] - a[1]); camera.position.copy(best[0][2]); controls.update();
    }
  }
  function zoomTo(dist) { camera.position.setLength(Math.min(controls.maxDistance, Math.max(controls.minDistance, dist))); controls.update(); }

  // ---------------------------------------------------------------- per-frame update
  const prevSun = V3(1, 0, 0); let havePrev = false, rotAngle = 0;
  function update(S, dt, wall) {
    if (!texDone) { earthMat.map = earthTexture(); earthMat.needsUpdate = true; texDone = true; }
    const p = V3(...S.p), v = V3(...S.v), r = p.length();
    P.rAU = r; P.t = S.t; P.mSys = S.mSys;
    P.sunDir.copy(p).negate().normalize();
    P.vhat.copy(v).normalize(); P.speedKms = S.v_kms;
    P.nrm.crossVectors(p, v); if (P.nrm.lengthSq() < 1e-12) P.nrm.set(0, 0, 1); P.nrm.normalize();
    const env = spaceEnv(r, S.mSys); P.env = env;
    // keep the view fixed relative to the Sun (Earth's orbital motion turns the Sun direction once per orbit)
    if (P.lockSun && havePrev) { _q.setFromUnitVectors(prevSun, P.sunDir); camera.position.applyQuaternion(_q); if (P.beamView) P.camAnchor.applyQuaternion(_q); }
    prevSun.copy(P.sunDir); havePrev = true;
    if (!P.initView) { P.initView = true; look('oblique', 9); }
    // Sun
    sunCore.position.copy(P.sunDir).multiplyScalar(SUN_D); sunCore.scale.setScalar(SUN_D * env.sun_ang_rad);
    sunGlow.position.copy(sunCore.position); sunGlow.scale.setScalar(SUN_D * env.sun_ang_rad * 9);
    sunLight.position.copy(P.sunDir).multiplyScalar(100); sunLight.intensity = 2.6 * Math.max(0.35, Math.min(1, env.insolation_wm2 / 1361) ** 0.35);
    // Earth rotation (sidereal), frozen at fast playback
    const daysPerS = S.speedY * 365.25;
    P.rotFrozen = S.playing && daysPerS > ROT_MAX_DAYS_PER_S;
    if (!P.rotFrozen) rotAngle = 2 * Math.PI * ((S.t * 365.25 / SIDEREAL_DAY) % 1);
    earthSpin.rotation.y = rotAngle;
    earthSpin.updateMatrixWorld(true);
    mhat.copy(dipoleBody).applyQuaternion(earthSpin.getWorldQuaternion(_q)).normalize();
    // Moon
    P.moonPos = S.moon ? V3(S.moon[0] / RE_KM, S.moon[1] / RE_KM, S.moon[2] / RE_KM) : null;
    moon.visible = layers.moon && !!P.moonPos; if (P.moonPos) moon.position.copy(P.moonPos);
    moonOrbit.visible = layers.moonorbit; if (layers.moonorbit) drawMoonOrbit(S.moonSt, 398600.4418 * (S.mSys - 7.3458e22) / 5.97216787e24 + 4902.8);
    // asteroid (true size for its mass with assumed density), docked on the ring's axis
    const frac = S.mAst0 > 0 ? Math.max(0, S.mAst / S.mAst0) : 0;
    const rTrue = Math.cbrt(3 * Math.max(0, S.mAst) / (4 * Math.PI * RHO_AST)) / (RE_KM * 1000);
    P.astRTrue = rTrue; P.astFrac = frac;
    const astR = Math.max(rTrue, frac > 0 ? 0.03 : 0); P.astR = astR; P.astEnlarged = frac > 0 && rTrue < 0.03;
    // throw direction: the latest one, kept fixed in Earth's orbital frame (radial, along-track, normal)
    const eh = S.ehatRtn ? V3().copy(P.sunDir).multiplyScalar(-S.ehatRtn[0]).addScaledVector(V3().crossVectors(P.nrm, P.sunDir).negate(), S.ehatRtn[1]).addScaledVector(P.nrm, S.ehatRtn[2]).normalize()
      : V3(...S.ehat).normalize();
    P.ehNow = eh;
    // docked on the ring's AXIS (update p36): its Earth-facing side rests on the grab point GRAB_Z, so it is the same
    // distance from all 8 arm roots; the clamps lower it as it shrinks (the grab point never moves)
    P.astPos.copy(P.nrm).multiplyScalar(GRAB_Z + astR * 0.9);
    asteroid.visible = frac > 0; asteroid.scale.setScalar(astR); asteroid.position.copy(P.astPos);
    if (!S.playing || !P.rotFrozen) asteroid.rotation.y += dt * 0.2;
    // throws: every recorded launch is ONE packet thrown by all eight arms together, in its recorded direction.
    // Launches recorded while a throw is under way are released with the next throw (fast playback).
    if (S.events && S.events.length) {
      for (const e of S.events.slice(-8)) {
        if (group.phase >= 1 && !group.pending.length) { group.phase = 0; group.released = false; }
        group.pending.push(e); if (group.pending.length > 6) group.pending.shift();
      }
      if (group.phase < 0.36) group.eh.copy(eh);            // aim is fixed once the claws have closed
      P.lastThrowWall = wall;
    }
    P.groupThrow = group.phase < 1; P.group = group; P.arms = arms;   // (exposed for tests)
    armGroup.visible = layers.arms;
    if (layers.arms) updateArms(wall, dt, P.astPos, astR, false);
    else {
      if (group.phase < 1 || group.pending.length) { group.phase = 1; for (const e of group.pending.splice(0)) startPacket(P.astPos, e); }
      if (layers.beams || P.beamView) updateArms(wall, dt, P.astPos, astR, false);     // hidden arms still carry the rectennas
    }
    updateMirrorRing(S, wall);
    pkLines.visible = pkHeads.visible = layers.packets; updatePackets(dt);
    updateBeams(S, p, r, dt, wall); beamCam(dt);
    // environment layers
    field.visible = layers.field; if (layers.field) buildField(env);
    bounds.visible = layers.magnetopause; if (layers.magnetopause) { buildBounds(env); bounds.quaternion.setFromUnitVectors(UPY, P.sunDir); }
    belts.visible = layers.belts; if (layers.belts) belts.quaternion.setFromUnitVectors(UPZ, mhat);
    P.windHidden = layers.wind && camera.position.length() > 250;
    wind.visible = layers.wind && !P.windHidden; if (wind.visible) updateWind(env, Math.min(dt, 0.05), true);
    well.visible = wellMarks.visible = layers.gravwell; if (layers.gravwell && (wellFrame++ % 8 === 0)) buildWell(env);
    hillG.visible = layers.hill;
    if (layers.hill) {
      const hp = hillCircle.userData.pos, ex = V3().copy(P.sunDir).negate(), ey = V3().crossVectors(P.nrm, ex);
      for (let j = 0; j <= 256; j++) { const a = j / 256 * 2 * Math.PI; _a.copy(ex).multiplyScalar(Math.cos(a) * env.hill_re).addScaledVector(ey, Math.sin(a) * env.hill_re); hp.set([_a.x, _a.y, _a.z], 3 * j); }
      hillCircle.geometry.attributes.position.needsUpdate = true; hillCircle.geometry.setDrawRange(0, 257);
      hillSphere.scale.setScalar(env.hill_re);
      L1.position.copy(P.sunDir).multiplyScalar(env.hill_re); L2.position.copy(P.sunDir).multiplyScalar(-env.hill_re);
      L1.scale.setScalar(env.hill_re * 0.012); L2.scale.setScalar(env.hill_re * 0.012);
    }
    // gravity pulls on Earth
    const rm = r * AU_KM * 1000; P.aSun = GM_SUN / (rm * rm);
    P.aMoon = P.moonPos ? GM_MOON / (P.moonPos.length() * RE_KM * 1000) ** 2 : NaN;
    const alen = (a) => 1.2 + 1.3 * Math.log10(a / 1e-6);
    arSun.visible = layers.gravvec; arMoon.visible = layers.gravvec && !!P.moonPos;
    if (layers.gravvec) {
      arSun.position.copy(P.sunDir).multiplyScalar(1.02); arSun.setDirection(P.sunDir); arSun.setLength(alen(P.aSun), 0.45, 0.25);
      if (P.moonPos) { const md = V3().copy(P.moonPos).normalize(); arMoon.position.copy(md).multiplyScalar(1.02); arMoon.setDirection(md); arMoon.setLength(alen(P.aMoon), 0.4, 0.22); }
    }
    // Mars (recorded heliocentric state, relative to Earth)
    P.marsPos = null; P.marsDistAU = NaN; P.aMars = NaN;
    if (S.mars) {
      const rel = V3(...S.mars), dAU = rel.length(), dRE = dAU * AU_KM / RE_KM;
      P.marsDistAU = dAU; P.marsDir = rel.clone().normalize();
      P.marsPos = P.marsDir.clone().multiplyScalar(Math.min(dRE, 40000));
      P.aMars = GM_MARS / (dAU * AU_KM * 1000) ** 2;
      marsDotGeom.attributes.position.setXYZ(0, P.marsPos.x, P.marsPos.y, P.marsPos.z); marsDotGeom.attributes.position.needsUpdate = true;
      const near = dAU < 0.3 && dRE < 40000 && (camera.position.length() > 150 || dAU < 2 * S.marsCaution);   // spheres only when zoomed out or Mars really near
      marsSafeS.visible = marsCautS.visible = near;
      if (near) { marsSafeS.position.copy(P.marsPos); marsCautS.position.copy(P.marsPos); marsSafeS.scale.setScalar(S.marsSafe * AU_KM / RE_KM); marsCautS.scale.setScalar(S.marsCaution * AU_KM / RE_KM); }
      P.marsSafe = S.marsSafe;
    }
    marsG.visible = layers.mars && !!P.marsPos;
    arMars.visible = layers.gravvec && !!P.marsPos;
    if (arMars.visible) { arMars.position.copy(P.marsDir).multiplyScalar(1.02); arMars.setDirection(P.marsDir); arMars.setLength(Math.max(0.7, alen(P.aMars)), 0.35, 0.2); }
    arVel.visible = layers.motion;
    if (layers.motion) { _a.crossVectors(P.nrm, P.vhat).normalize(); arVel.position.copy(P.vhat).multiplyScalar(1.05); arVel.setDirection(P.vhat); arVel.setLength(4.2, 0.5, 0.28); }
    P.pushRecent = layers.motion && S.recentPush;
    arPush.visible = !!P.pushRecent;
    if (arPush.visible) { const pd = V3().copy(eh).negate(); arPush.position.copy(pd).multiplyScalar(1.05); arPush.setDirection(pd); arPush.setLength(3.2, 0.45, 0.26); }
    // auto-zoom when a large-scale layer was just switched on
    if (P.wantZoom) {
      const z = P.wantZoom; P.wantZoom = null;
      if (z === 'hill') { const side = V3().crossVectors(P.nrm, P.sunDir); camera.position.copy(side).multiplyScalar(0.75).addScaledVector(P.sunDir, -0.45).addScaledVector(P.nrm, 0.75).normalize().multiplyScalar(2.6 * env.hill_re); controls.update(); }
    }
    P.S = S;
  }

  function placeLabels() {
    if (!layers.labels) { hideLabels(); return; }
    const env = P.env, S = P.S, cd = camera.position.length();
    const near = cd < 25, mid = cd < 260;          // hide small-scale labels when zoomed out (they would pile up on Earth)
    label('sun', sunCore.position, `Sun · ${(2 * env.sun_ang_rad / D2R).toFixed(2)}° wide from here`);
    label('earth', V3(0, 0, 1.1), near ? 'Earth (true size; 1 unit = 1 Earth radius = 6,371 km)' : 'Earth');
    label('moon', layers.moon && P.moonPos ? V3().copy(P.moonPos).add(V3(0, 0, MOON_R * 1.4)) : null, P.moonPos ? `Moon · ${Math.round(P.moonPos.length() * RE_KM).toLocaleString('en-US')} km` : '');
    label('ast', near && P.astFrac > 0 ? V3().copy(P.astPos).addScaledVector(P.nrm, P.astR + 0.15) : null,
      `asteroid · ${(100 * P.astFrac).toFixed(P.astFrac < 0.1 ? 2 : 1)} % left · Ø ${Math.round(2 * P.astRTrue * RE_KM).toLocaleString('en-US')} km${P.astEnlarged ? ' (drawn enlarged)' : ''}`);
    label('arms', near && layers.arms ? V3().copy(P.nrm).multiplyScalar(-1.4).addScaledVector(V3().copy(P.sunDir).negate(), RING_R) : null, P.groupThrow ? 'all 8 arms throwing ONE packet together (illustration)' : 'launcher ring + 8 arms (illustration)');
    label('mp', mid && layers.magnetopause ? V3().copy(P.sunDir).multiplyScalar(env.r0_re) : null, `magnetopause · ${env.r0_re.toFixed(1)} R_E`);
    label('bs', mid && layers.magnetopause ? V3().copy(P.sunDir).multiplyScalar(env.rbs_re).addScaledVector(P.nrm, 2) : null, `bow shock · ${env.rbs_re.toFixed(1)} R_E`);
    label('belts', near && layers.belts ? V3().copy(mhat).cross(P.nrm).normalize().multiplyScalar(4.8).addScaledVector(P.nrm, 1.5) : null, 'radiation belts (schematic)');
    label('field', near && layers.field ? V3().copy(mhat).multiplyScalar(2.6) : null, 'magnetic north (dipole axis)');
    label('wind', mid && layers.wind ? V3().copy(P.sunDir).multiplyScalar(Math.min(45, env.rbs_re * 2.2)).addScaledVector(P.nrm, 12) : null, `solar wind → ${env.n_cm3.toFixed(2)} /cm³, 400 km/s`);
    label('hill', layers.hill ? V3().crossVectors(P.nrm, P.sunDir).multiplyScalar(env.hill_re) : null, `Hill sphere · ${(env.hill_re * RE_KM / 1e6).toFixed(2)} million km`);
    label('l1', layers.hill ? L1.position : null, 'L1 (toward Sun)');
    label('l2', layers.hill ? L2.position : null, 'L2');
    label('well', layers.gravwell && P.wellLabelPos ? P.wellLabelPos : null, 'gravity well (log height)');
    label('gsun', near && layers.gravvec ? V3().copy(P.sunDir).multiplyScalar(1 + 1.2 + 1.3 * Math.log10(P.aSun / 1e-6)) : null, `Sun pull ${P.aSun.toExponential(2)} m/s²`);
    label('gmoon', near && layers.gravvec && P.moonPos ? V3().copy(P.moonPos).normalize().multiplyScalar(1 + 1.2 + 1.3 * Math.log10(P.aMoon / 1e-6)) : null, `Moon pull ${P.aMoon.toExponential(2)} m/s²`);
    label('mars', layers.mars && P.marsPos ? P.marsPos : null, P.marsPos ? `Mars · ${P.marsDistAU.toFixed(4)} AU = ${(P.marsDistAU * AU_KM / 1e6).toFixed(2)} million km${P.marsDistAU < P.marsSafe ? ' – INSIDE SAFE DISTANCE' : ''}` : '');
    label('gmars', near && layers.gravvec && P.marsPos ? V3().copy(P.marsDir).multiplyScalar(1 + Math.max(0.7, 1.2 + 1.3 * Math.log10(P.aMars / 1e-6)) + 0.3) : null, P.marsPos ? `Mars pull ${P.aMars.toExponential(2)} m/s²` : '');
    label('vel', near && layers.motion ? V3().copy(P.vhat).multiplyScalar(5.3) : null, Number.isFinite(P.speedKms) ? `orbital velocity ${P.speedKms.toFixed(2)} km/s` : 'direction of motion');
    label('push', near && P.pushRecent ? V3().copy(P.ehNow).multiplyScalar(-4.3) : null, 'push on Earth (recoil)');
    if (P.power && layers.beams) {
      const b = beams[P.beamView ? P.beamArm : 2], pw = P.power, sol = P.solar, nA = sol.n_arms || 8;
      const perArm = pw.inW / nA, inten = perArm / (Math.PI * (sol.dr_km * 500) ** 2);
      label('beam', b.ok ? V3().copy(b.A.tip).addScaledVector(b.src, P.beamView ? 1.6 : 6) : null,
        pw.inW > 0 ? `microwave beam ${sol.freq_ghz} GHz · ${pw.inW / nA > 0 ? (perArm).toExponential(2) : 0} W into this arm` : `microwave beam ${sol.freq_ghz} GHz · stand-by pilot beam (no launch right now – no power flowing)`);
      label('sat', b.ok && b.satPos ? b.satPos : null, 'solar collector satellite sending this beam (drawn ~15× enlarged)');
      label('dish', (P.beamView || near) && b.ok ? b.dish.position : null,
        `rectenna Ø ${sol.dr_km.toLocaleString('en-US')} km (true size)` + (pw.inW > 0 ? ` · ${inten.toExponential(1)} W/m² = ${(inten / 1361).toExponential(1)}× sunlight` : ''));
    } else { label('beam', null, ''); label('dish', null, ''); label('sat', null, ''); }
    if (P.solar && layers.collectors && belt.visible) {
      let bi = 0, bd = -2; for (let i = 0; i < NB; i += 7) { _a.set(beltPos[3 * i], beltPos[3 * i + 1], beltPos[3 * i + 2]).normalize(); const d = _a.dot(P.nrm) * 0.3 + _a.dot(V3().crossVectors(P.nrm, P.sunDir)); if (d > bd) { bd = d; bi = i; } }
      label('belt', V3(beltPos[3 * bi], beltPos[3 * bi + 1], beltPos[3 * bi + 2]), `collector belt (satellites drawn enlarged) · ${P.solar.orbit_au} AU from the Sun · ${(P.solar.count / 1e6).toFixed(0)} million collectors (~${Math.round(P.solar.each_side_km).toLocaleString('en-US')} km each)`);
    } else label('belt', null, '');
    resolveLabels();
  }

  function render() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    camera.aspect = w / Math.max(1, h);
    const shift = Math.min(170, 0.14 * w);          // the HUD covers the left part: shift Earth right
    camera.setViewOffset(w, h, -shift, 0, w, h);
    camera.updateProjectionMatrix();
    controls.update();
    camera.updateMatrixWorld();          // labels must use this frame's camera (it turns with the Sun direction)
    placeLabels();
    renderer.render(scene, camera);
  }
  function setLayer(k, on) {
    const was = layers[k]; layers[k] = on;
    if (on && !was && (k === 'gravwell' || k === 'hill')) P.wantZoom = 'hill';
    if (!on && !was) return;
    if (!on && (k === 'gravwell' || k === 'hill') && !layers.gravwell && !layers.hill && camera.position.length() > 200) zoomTo(30);
  }
  function setActive(on) { controls.enabled = on; if (!on) hideLabels(); }
  return { scene, camera, controls, layers, P, update, render, look, zoomTo, setLayer, setActive, hideLabels, setBeamView };
}
