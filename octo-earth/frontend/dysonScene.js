// OCTO-EARTH view "Dyson beam failure" (update p37; since p38 failed attempt 4 on failed.html) - a cinematic, CONCEPTUAL animation with a documented force model.
//
// A Dyson swarm of collector satellites (the same satellite model as the solar-collector views, instanced) harvests
// sunlight; energy paths run into ONE concentrator, which emits ONE beam to ONE giant concave silver reflector held at
// 8 mounts by the 8 real OCTO-EARTH arms (shared rig octoArms.js, 2 Earth radii each). The reflector is turned so
// that the beam is reflected onto Earth:
//
//     d = unit vector of the incident beam (concentrator -> reflector)
//     n = normalize(r0 - d)  with r0 = unit vector reflector -> Earth's centre   (front normal of the mirror)
//     r = d - 2 (d . n) n                                                        (reflected ray, computed every frame
//                                                                                 from the ACTUAL mirror normal)
// Forces of a light beam of power P (c = speed of light), perfect reflection, full absorption at Earth:
//     on the reflector (carried by the arms into Earth):   F_arms = (P / c) (d - r)
//     on Earth where the reflected beam is absorbed:       F_abs  = (P / c) r
//     total on Earth + mechanism:                          F      = (P / c) d   (away from the Sun)
// P_max = 1.29e25 W (the E2 beam power of the OCTO-EARTH report)  ->  F = 4.3e16 N, a = F / M_earth = 7.2e-9 m/s^2.
// Earth's displacement is integrated from that acceleration with the animation time COMPRESSED (1 s of animation =
// 0.25 years); its size is drawn at true scale relative to Earth (not exaggerated). Distances between Sun, swarm,
// concentrator and Earth are NOT to scale.
//
// Arm loads: each arm carries its share of |F_arms|, weighted by its lever (longer reach = more load). The arm
// capacity is a STYLIZED, ASSUMED threshold (ARM_CAP_FRAC of the full-power load): overload above 100 %, structural
// failure at FAIL_STRESS. The failure cascade, fragments and tumbling are illustrative, but every motion is a
// deterministic function of time, so Pause / Replay are exact.
import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import { createArmRig } from './octoArms.js';
import { createRealEarth, createStarfield, createSun, makeSpaceEnv } from './earthAssets.js';
import { satelliteGeometry, satQuat, glowTex } from './collector.js';
import { createLabels, createCineCamera, renderShifted, hudRow, hudBar, sm, fmtE } from './cineKit.js';
import { DYSON, F_REQ } from './failedPhysics.js';

const V3 = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const C_LIGHT = 2.998e8, M_EARTH = 5.972e24, RE_M = 6.371e6, YEAR_S = 3.156e7;
// Reflector geometry in Earth radii (found by a numeric search, re-checked by checkDysonGeometry() and the tests):
// ring axis u at ALPHA from +x (away from the Sun) in the ecliptic, reflector centre at H along u, mounts at radius M.
export const DY = {
  P_MAX: 1.29e25, ALPHA_DEG: 51, H: 1.55, M: 0.98, DISC_R: 1.02, PHASE_DEG: 22.5,
  SUN_R: 6, EARTH_X: 70, ER: 2.5, CONC_X: 26, N_SATS: 400, N_PATHS: 48,
  T_COLLECT: 5, T_CONC: 9, T_ALIGN: 12, T_RAMP_END: 20, P_ALIGN: 0.05, YEARS_PER_S: 0.25,
  ARM_CAP_FRAC: 0.55, FAIL_STRESS: 1.55, T_END: 40,
};
/** Unit vectors of the mechanism (Earth-centred, Earth radii). */
export function dyFrame() {
  const a = DY.ALPHA_DEG * Math.PI / 180, u = V3(Math.cos(a), Math.sin(a), 0), e2 = V3(0, 0, 1), e1 = V3().crossVectors(e2, u).normalize();
  return { u, e1, e2 };
}
/** Mirror reflection r = d - 2 (d . n) n. */
export function reflect(d, n, out = V3()) { return out.copy(d).addScaledVector(n, -2 * d.dot(n)); }
/** Front normal that sends incident direction d to direction r0: n = normalize(r0 - d). */
export function alignNormal(d, r0, out = V3()) { return out.subVectors(r0, d).normalize(); }

function segDist(p, a, b) { const ab = V3().subVectors(b, a); const t = Math.max(0, Math.min(1, V3().subVectors(p, a).dot(ab) / ab.lengthSq())); return p.distanceTo(V3().copy(a).addScaledVector(ab, t)); }
/** Geometry check: reach of every arm, beam clearance from Earth, reflected ray hits Earth's centre. */
export function checkDysonGeometry(earthShiftRE = 0) {
  const { u, e1, e2 } = dyFrame();
  const Cl = V3((DY.CONC_X - DY.EARTH_X) / DY.ER - earthShiftRE, 0, 0);
  const Rc = V3().copy(u).multiplyScalar(DY.H), d = V3().subVectors(Rc, Cl).normalize(), r0 = V3().copy(u).negate();
  const n = alignNormal(d, r0), r = reflect(d, n);
  let worstReach = 0, minMountR = 9;
  for (let k = 0; k < 8; k++) {
    const ph = (k * 45 + DY.PHASE_DEG) * Math.PI / 180, rk = V3().copy(e1).multiplyScalar(Math.cos(ph)).addScaledVector(e2, Math.sin(ph));
    const base = V3().copy(rk).multiplyScalar(1.30), pk = V3().copy(rk).addScaledVector(n, -rk.dot(n)).normalize(), mt = V3().copy(Rc).addScaledVector(pk, DY.M);
    worstReach = Math.max(worstReach, mt.distanceTo(base)); minMountR = Math.min(minMountR, mt.length());
  }
  return {
    worstReach, minMountR, earthClear: segDist(V3(), Cl, Rc), tiltDeg: Math.acos(Math.min(1, n.dot(r0))) * 180 / Math.PI,
    reflErrDeg: Math.acos(Math.min(1, r.dot(r0))) * 180 / Math.PI, d: d.toArray(), n: n.toArray(), r: r.toArray(),
    F_N: DY.P_MAX / C_LIGHT, a_ms2: DY.P_MAX / C_LIGHT / M_EARTH,
  };
}
const STATUS = ['COLLECTING', 'CONCENTRATING', 'BEAM ACTIVE', 'OVERLOAD', 'STRUCTURAL FAILURE'];

export function createDysonView(renderer, canvas, container) {
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0x010207);
  const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 30000); camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true; controls.minDistance = 2; controls.maxDistance = 400; controls.enabled = false;
  const ER = DY.ER, E0 = V3(DY.EARTH_X, 0, 0), CW = V3(DY.CONC_X, 0, 0);

  // ---------------------------------------------------------------- Sun, stars, light
  scene.add(createStarfield(12000, 5200, 7));
  scene.add(new THREE.AmbientLight(0x707a90, 0.10));
  const sun = createSun(DY.SUN_R); scene.add(sun.group);
  const sunKey = new THREE.DirectionalLight(0xfff2dd, 2.2); scene.add(sunKey); scene.add(sunKey.target);   // light on the Earth system

  // ---------------------------------------------------------------- Dyson swarm: instanced collector satellites
  const satGeo = satelliteGeometry(true);
  const satMat = new THREE.MeshStandardMaterial({ vertexColors: true, metalness: 0.45, roughness: 0.4, emissive: 0x5a5c60, emissiveIntensity: 0.55, side: THREE.DoubleSide });
  const sats = new THREE.InstancedMesh(satGeo, satMat, DY.N_SATS); sats.frustumCulled = false; scene.add(sats);
  const orb = [...Array(DY.N_SATS)].map((_, i) => {
    const h = (k) => { const x = Math.sin((i + 1) * 12.9898 + k * 78.233) * 43758.5453; return x - Math.floor(x); };
    const plane = i % 10, inc = (plane - 4.5) * 7.5 * Math.PI / 180, node = plane * 0.63 + 0.3 * h(1);
    const r = 10 + 8 * h(2);
    return { r, inc, node, ph: 2 * Math.PI * h(3), w: 0.22 * Math.pow(10 / r, 1.5), pos: V3() };
  });
  const _q = new THREE.Quaternion(), _m = new THREE.Matrix4(), _sc = V3(), _a = V3(), _b = V3(), _c = V3(), _d = V3(), NORTH = V3(0, 0, 1);
  function satPos(o, t, out) {
    const th = o.ph + o.w * t; out.set(o.r * Math.cos(th), o.r * Math.sin(th), 0);
    out.applyAxisAngle(V3(1, 0, 0), o.inc).applyAxisAngle(NORTH, o.node); return out;
  }
  // energy paths from 48 collectors into the concentrator's receiving dish, with travelling pulses
  const PATH_SUB = 14, PULSES_PER = 3;
  const pathIdx = [...Array(DY.N_PATHS)].map((_, k) => Math.floor(k * DY.N_SATS / DY.N_PATHS) + 3);
  const pathPos = new Float32Array(DY.N_PATHS * PATH_SUB * 6), pathGeo = new THREE.BufferGeometry(); pathGeo.setAttribute('position', new THREE.BufferAttribute(pathPos, 3));
  const paths = new THREE.LineSegments(pathGeo, new THREE.LineBasicMaterial({ color: 0xffe2a8, transparent: true, opacity: 0.12, blending: THREE.AdditiveBlending, depthWrite: false }));
  paths.frustumCulled = false; scene.add(paths);
  const pulsePos = new Float32Array(DY.N_PATHS * PULSES_PER * 3), pulseGeo = new THREE.BufferGeometry(); pulseGeo.setAttribute('position', new THREE.BufferAttribute(pulsePos, 3));
  const pulses = new THREE.Points(pulseGeo, new THREE.PointsMaterial({ color: 0xfff0c8, size: 0.7, map: glowTex(), transparent: true, opacity: 0.0, blending: THREE.AdditiveBlending, depthWrite: false }));
  pulses.frustumCulled = false; scene.add(pulses);
  const DISH_IN = V3().copy(CW).add(V3(-1.1, 0, 0));
  function pathPoint(P, s, out, k) {     // quadratic curve from collector P into the dish, bulging away from the Sun-Earth line
    _c.addVectors(P, DISH_IN).multiplyScalar(0.5); _c.z += 3 * Math.sin(k * 1.7); _c.y += 2.5 * Math.cos(k * 1.3);
    const u = 1 - s; return out.set(0, 0, 0).addScaledVector(P, u * u).addScaledVector(_c, 2 * u * s).addScaledVector(DISH_IN, s * s);
  }

  // ---------------------------------------------------------------- concentrator (receiving dish + emitter)
  const metal = new THREE.MeshStandardMaterial({ color: 0xb8c0cc, metalness: 0.8, roughness: 0.3, emissive: 0x10141a });
  const conc = new THREE.Group(); conc.position.copy(CW); scene.add(conc);
  const dishPts = []; for (let i = 0; i <= 16; i++) { const r = 1.6 * i / 16; dishPts.push(new THREE.Vector2(r, 0.35 * (r / 1.6) ** 2)); }
  const recv = new THREE.Mesh(new THREE.LatheGeometry(dishPts, 40), new THREE.MeshStandardMaterial({ color: 0xd8c89a, metalness: 0.9, roughness: 0.25, side: THREE.DoubleSide, emissive: 0x2a2010 }));
  recv.rotation.z = Math.PI / 2; recv.position.x = -0.9; conc.add(recv);              // opens towards the Sun (-x)
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.45, 0.6, 1.4, 24), metal); body.rotation.z = Math.PI / 2; conc.add(body);
  const emitter = new THREE.Group(); conc.add(emitter);
  const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.38, 1.6, 20), metal); barrel.position.y = 0.9; emitter.add(barrel);
  const lens = new THREE.Mesh(new THREE.CircleGeometry(0.22, 24), new THREE.MeshBasicMaterial({ color: 0xfff1d0 })); lens.position.y = 1.71; lens.rotation.x = -Math.PI / 2; emitter.add(lens);
  const concGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex('rgba(255,236,190,1)', 'rgba(255,170,60,0)'), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
  concGlow.scale.setScalar(4); conc.add(concGlow);

  // ---------------------------------------------------------------- Earth system (Earth-centred group, Earth radii)
  const sys = new THREE.Group(); sys.scale.setScalar(ER); sys.position.copy(E0); scene.add(sys);
  const earth = createRealEarth(renderer, { spinRate: 0.05 }); sys.add(earth.group); earth.group.rotation.y = 23.44 * Math.PI / 180;
  const F = dyFrame();
  const rig = createArmRig({ look: 'cinematic', nseg: 10, fragments: 96 });
  sys.add(rig.group); rig.setFrame(F.e1, F.e2, F.u, DY.PHASE_DEG);

  // reflector: thick concave silver dish, rim frame, back spokes + hub, 8 mounts
  const envTex = makeSpaceEnv(renderer, V3(-1, 0, 0));
  const silver = new THREE.MeshStandardMaterial({ color: 0xe4e9f0, metalness: 1.0, roughness: 0.14, envMap: envTex, envMapIntensity: 1.4, side: THREE.DoubleSide });
  const frameMat = new THREE.MeshStandardMaterial({ color: 0x5c6672, metalness: 0.75, roughness: 0.35, envMap: envTex, envMapIntensity: 0.6 });
  const refl = new THREE.Group(); sys.add(refl);
  {
    const R = DY.DISC_R, sag = 0.11, th = 0.05, prof = [];
    for (let i = 0; i <= 24; i++) { const r = R * i / 24; prof.push(new THREE.Vector2(r, sag * (r / R) ** 2)); }       // front (concave) face
    for (let i = 24; i >= 0; i--) { const r = R * i / 24; prof.push(new THREE.Vector2(r, sag * (r / R) ** 2 - th - 0.02 * (1 - r / R))); }   // back face
    const g = new THREE.LatheGeometry(prof, 96); g.rotateX(Math.PI / 2);         // lathe axis y -> local z (= front normal)
    const dish = new THREE.Mesh(g, silver); refl.add(dish);
    const rim = new THREE.Mesh(new THREE.TorusGeometry(R, 0.035, 10, 96), frameMat); rim.position.z = sag - 0.02; refl.add(rim);
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.13, 0.14, 16), frameMat); hub.rotation.x = Math.PI / 2; hub.position.z = -th - 0.08; refl.add(hub);
    for (let k = 0; k < 8; k++) {
      const a = k * Math.PI / 4 + 0.2, sp = new THREE.Mesh(new THREE.BoxGeometry(R * 0.96, 0.03, 0.035), frameMat);
      sp.position.set(Math.cos(a) * R * 0.5, Math.sin(a) * R * 0.5, -th - 0.03 + 0.5 * sag * 0.25); sp.rotation.z = a; refl.add(sp);
    }
  }
  // nominal orientation -> mount positions in the reflector frame (each mount faces its arm)
  const qOf = (n, out) => { const m2 = V3().copy(NORTH).addScaledVector(n, -NORTH.dot(n)).normalize(), m1 = V3().crossVectors(m2, n); return out.setFromRotationMatrix(new THREE.Matrix4().makeBasis(m1, m2, n)); };
  const Rc = V3().copy(F.u).multiplyScalar(DY.H);
  const geo0 = checkDysonGeometry(0), n0 = V3(...geo0.n), q0 = qOf(n0, new THREE.Quaternion()), q0i = q0.clone().invert();
  const mountsLocal = rig.arms.map((arm) => { const rk = arm.dir; const pk = V3().copy(rk).addScaledVector(n0, -rk.dot(n0)).normalize(); return pk.multiplyScalar(DY.M).applyQuaternion(q0i).setZ(0.09); });
  const mountMesh = new THREE.InstancedMesh(new THREE.BoxGeometry(0.16, 0.16, 0.12), frameMat, 8);
  mountsLocal.forEach((p, k) => { _q.setFromAxisAngle(NORTH, Math.atan2(p.y, p.x)); _m.compose(p, _q, _sc.set(1, 1, 1)); mountMesh.setMatrixAt(k, _m); });
  refl.add(mountMesh);
  // silver chips from the reflector (pre-created pool)
  const NCH = 24, chips = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 0.7), silver, NCH); chips.frustumCulled = false; sys.add(chips);
  const chipSt = [...Array(NCH)].map((_, i) => ({ t0: 1e9, p0: V3(), v: V3(), ax: V3(Math.sin(i * 1.3), Math.cos(i * 2.1), Math.sin(i * 0.7)).normalize(), w: 1.5 + (i % 5), s: 0.05 + 0.03 * (i % 3) }));

  // ---------------------------------------------------------------- beams: core + glow + pulses (one incident, one reflected)
  const beamGlowMat = () => new THREE.ShaderMaterial({
    uniforms: { col: { value: new THREE.Color(1.0, 0.86, 0.6) }, power: { value: 0 } }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: 'varying vec3 vN; varying vec3 vV; void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }',
    fragmentShader: 'uniform vec3 col; uniform float power; varying vec3 vN; varying vec3 vV; void main(){ float f = pow(abs(dot(normalize(vN), normalize(vV))), 2.2); gl_FragColor = vec4(col * f * power, f * power); }',
  });
  function makeBeam() {
    const g = new THREE.Group(); g.visible = false; scene.add(g);
    const cyl = new THREE.CylinderGeometry(1, 1, 1, 20, 1, true);
    const core = new THREE.Mesh(cyl, new THREE.MeshBasicMaterial({ color: 0xfffaf0, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); g.add(core);
    const glow = new THREE.Mesh(cyl, beamGlowMat()); g.add(glow);
    const NP = 26, pp = new Float32Array(NP * 3), pg = new THREE.BufferGeometry(); pg.setAttribute('position', new THREE.BufferAttribute(pp, 3));
    const pts = new THREE.Points(pg, new THREE.PointsMaterial({ color: 0xffffff, size: 0.9, map: glowTex(), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
    pts.frustumCulled = false; scene.add(pts);
    const hit = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex('rgba(255,245,220,1)', 'rgba(255,180,80,0)'), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    hit.visible = false; scene.add(hit);
    return { g, core, glow, pts, pp, NP, hit, a: V3(), b: V3() };
  }
  const beamIn = makeBeam(), beamOut = makeBeam();
  function setBeam(B, a, b, power, t, hitEnd) {
    const len = a.distanceTo(b); B.g.visible = power > 0.002 && len > 1e-3; B.pts.visible = B.g.visible; B.hit.visible = B.g.visible && !!hitEnd;
    if (!B.g.visible) return;
    B.a.copy(a); B.b.copy(b);
    _d.subVectors(b, a).normalize(); _q.setFromUnitVectors(V3(0, 1, 0), _d);
    B.g.position.addVectors(a, b).multiplyScalar(0.5); B.g.quaternion.copy(_q);
    const pw = Math.min(1, Math.max(0, power));
    B.core.scale.set(0.035 + 0.05 * pw, len, 0.035 + 0.05 * pw); B.core.material.opacity = 0.35 + 0.65 * pw;
    B.glow.scale.set(0.22 + 0.32 * pw, len, 0.22 + 0.32 * pw); B.glow.material.uniforms.power.value = 0.25 + 0.75 * pw;
    for (let i = 0; i < B.NP; i++) { const s = ((t * 22 + i * len / B.NP) % len + len) % len; _c.copy(a).addScaledVector(_d, s); B.pp.set([_c.x, _c.y, _c.z], 3 * i); }
    B.pts.geometry.attributes.position.needsUpdate = true; B.pts.material.opacity = 0.3 + 0.7 * pw;
    if (hitEnd) { B.hit.position.copy(b); B.hit.scale.setScalar(1.2 + 2.2 * pw); B.hit.material.opacity = 0.5 + 0.5 * pw; }
  }
  // force indicator on Earth (arrow along the total force)
  const arrowMat = new THREE.MeshBasicMaterial({ color: 0xa8f0a0, transparent: true, opacity: 0.9 });
  const arrow = new THREE.Group(); scene.add(arrow);
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 1, 10), arrowMat); arrow.add(shaft);
  const head = new THREE.Mesh(new THREE.ConeGeometry(0.22, 0.6, 14), arrowMat); arrow.add(head);

  // ---------------------------------------------------------------- simulation state (deterministic)
  const S = {
    active: false, running: false, started: false, t: 0, wall: 0, lastPanel: -1, labels: true, autoCam: true,
    status: 0, P: 0, charge: 0, x_m: 0, v_ms: 0, F: V3(), Fmag: 0, Fabs: 0, Farms: 0, stress: new Array(8).fill(0), maxStress: 0,
    tOver: null, tFail: null, tDetach: null, tSafe: null, broken: [], pending: [], alignErr: 0, hitsDisc: false, hitsEarth: false,
    reflPos: V3().copy(Rc), reflQ: q0.clone(), Cl: V3(), d: V3(1, 0, 0), n: n0.clone(), r: V3(), hit: V3(), outEnd: V3(), ended: false,
  };
  // lever weights: arms reaching farther carry more of the reflector load (normalised to mean 1)
  const reach = mountsLocal.map((p, k) => p.clone().applyQuaternion(q0).add(Rc).distanceTo(rig.arms[k].base));
  const meanReach = reach.reduce((s, x) => s + x, 0) / 8, lever = reach.map((x) => (x / meanReach) ** 2);
  const leverSum = lever.reduce((s, x) => s + x, 0); for (let k = 0; k < 8; k++) lever[k] *= 8 / leverSum;
  const breakOrder = [...Array(8).keys()].sort((a, b) => lever[b] - lever[a]);
  const F_ARM_FULL = (DY.P_MAX / C_LIGHT) * 2 * Math.cos(geo0.tiltDeg * Math.PI / 180) / 8;   // mean arm load at full power
  const ARM_CAP = DY.ARM_CAP_FRAC * F_ARM_FULL * Math.max(...lever);                          // assumed (stylized) capacity: the most loaded arm reaches 100 % at ARM_CAP_FRAC of full power
  const wobbleAx1 = V3(1, 0, 0), wobbleAx2 = V3(0, 1, 0);
  let kickAx = V3(1, 0, 0), tumbleAx = V3(0.3, 0.8, 0.5).normalize(), detVel = V3(), detPos = V3(), detQ = new THREE.Quaternion();

  function powerAt(t) {
    if (t < DY.T_CONC) return 0;
    let p = t < DY.T_ALIGN ? DY.P_ALIGN : DY.P_ALIGN + (1 - DY.P_ALIGN) * Math.min(1, (t - DY.T_ALIGN) / (DY.T_RAMP_END - DY.T_ALIGN));
    if (S.tSafe !== null && t > S.tSafe) p *= Math.max(0, 1 - (t - S.tSafe) / 1.5);
    return p;
  }
  const _qa = new THREE.Quaternion(), _qb = new THREE.Quaternion(), _n = V3(), _r0 = V3();
  /** Reflector pose at time t (Earth-centred frame). */
  function reflectorPose(t, d) {
    const r0 = _r0.copy(Rc).negate().normalize();
    alignNormal(d, r0, _n); const qT = qOf(_n, _qa);
    if (S.tDetach !== null && t >= S.tDetach) {
      const dt = t - S.tDetach;
      S.reflPos.copy(detPos).addScaledVector(detVel, dt).addScaledVector(detVel, 0.15 * dt * dt);
      S.reflQ.copy(detQ).premultiply(_qb.setFromAxisAngle(tumbleAx, 0.55 * dt + 0.08 * dt * dt));
      return;
    }
    S.reflPos.copy(Rc);
    if (t < DY.T_CONC + 0.5) S.reflQ.copy(qOf(r0, _qb));                                              // stowed: facing Earth squarely
    else if (t < DY.T_ALIGN) S.reflQ.copy(qOf(r0, _qb)).slerp(qT, sm((t - DY.T_CONC - 0.5) / (DY.T_ALIGN - DY.T_CONC - 0.5)));
    else S.reflQ.copy(qT);
    // wobble under load, then the failure kicks
    const w = Math.max(0, S.maxStress - 0.75) / 0.8, amp = 0.028 * Math.min(1.3, w);
    if (amp > 0) S.reflQ.multiply(_qb.setFromAxisAngle(wobbleAx1, amp * Math.sin(17 * t))).multiply(_qb.setFromAxisAngle(wobbleAx2, amp * 0.8 * Math.sin(13 * t + 1)));
    if (S.tFail !== null && t >= S.tFail) {
      const dt = t - S.tFail, kick = 0.2 * (1 - Math.exp(-6 * dt)) + (dt > 0.55 ? 0.16 * (1 - Math.exp(-5 * (dt - 0.55))) : 0) + (dt > 1.1 ? 0.2 * (1 - Math.exp(-4 * (dt - 1.1))) : 0);
      S.reflQ.premultiply(_qb.setFromAxisAngle(kickAx, kick));
      S.reflPos.addScaledVector(F.u, 0.05 * dt * dt);
    }
  }
  function earthWorld(out) { return out.copy(E0).addScaledVector(V3(1, 0, 0), S.x_m / RE_M * ER); }
  const _ew = V3();

  /** Optics + forces at time t (Earth-centred frame): where the beam hits, the reflected ray, forces, arm loads. */
  function optics(t) {
    earthWorld(_ew);
    S.Cl.subVectors(CW, _ew).divideScalar(ER);
    S.d.subVectors(Rc, S.Cl).normalize();                    // the concentrator aims at the nominal reflector centre
    reflectorPose(t, S.d);
    S.n.set(0, 0, 1).applyQuaternion(S.reflQ);                // actual front normal
    // incident beam vs reflector disc
    const den = S.d.dot(S.n); S.hitsDisc = false;
    if (Math.abs(den) > 1e-6) {
      const s = _a.subVectors(S.reflPos, S.Cl).dot(S.n) / den;
      if (s > 0) { S.hit.copy(S.Cl).addScaledVector(S.d, s); S.hitsDisc = S.hit.distanceTo(S.reflPos) <= DY.DISC_R && den < 0; }
    }
    reflect(S.d, S.n, S.r);
    // reflected ray vs Earth (unit sphere at the origin)
    S.hitsEarth = false;
    if (S.hitsDisc) {
      const b = S.hit.dot(S.r), c = S.hit.lengthSq() - 1, disc = b * b - c;
      if (disc > 0 && -b - Math.sqrt(disc) > 0) { S.hitsEarth = true; S.outEnd.copy(S.hit).addScaledVector(S.r, -b - Math.sqrt(disc)); }
      else S.outEnd.copy(S.hit).addScaledVector(S.r, 40);
    }
    S.alignErr = Math.acos(Math.max(-1, Math.min(1, S.r.dot(_b.copy(S.hit).negate().normalize())))) * 180 / Math.PI;
    S.P = powerAt(t);
    const Pc = S.P * DY.P_MAX / C_LIGHT, attached = S.tDetach === null || t < S.tDetach;
    S.F.set(0, 0, 0); S.Farms = 0; S.Fabs = 0;
    if (S.hitsDisc && attached) { _c.subVectors(S.d, S.r).multiplyScalar(Pc); S.F.add(_c); S.Farms = _c.length(); }
    if (S.hitsEarth) { _c.copy(S.r).multiplyScalar(Pc); S.F.add(_c); S.Fabs = _c.length(); }
    S.Fmag = S.F.length();
    let mx = 0;
    for (let k = 0; k < 8; k++) {
      const brokenK = S.broken.includes(k), intact = 8 - S.broken.length;
      S.stress[k] = brokenK || !attached ? 0 : (S.Farms / Math.max(1, intact)) * lever[k] / ARM_CAP;
      mx = Math.max(mx, S.stress[k]);
    }
    S.maxStress = mx;
  }

  function step(h) {
    S.t += h; const t = S.t;
    optics(t);
    // Earth: a = F / M_earth along the force, animation time compressed (1 s = YEARS_PER_S years)
    const hs = h * DY.YEARS_PER_S * YEAR_S, ax = S.F.x / M_EARTH;      // along +x (the force is almost exactly +x)
    S.v_ms += ax * hs; S.x_m += S.v_ms * hs;
    earth.update(h);
    S.charge = t < DY.T_COLLECT ? 0.2 * t / DY.T_COLLECT : t < DY.T_CONC ? 0.2 + 0.8 * (t - DY.T_COLLECT) / (DY.T_CONC - DY.T_COLLECT) : (S.tSafe !== null && t > S.tSafe ? Math.max(0, 1 - (t - S.tSafe) / 1.5) : 1);
    if (S.tOver === null && S.maxStress >= 1) S.tOver = t;
    if (S.tFail === null && S.maxStress >= DY.FAIL_STRESS) {
      S.tFail = t; S.tSafe = t + 4.6;
      const k0 = breakOrder[0], mp = _a.copy(mountsLocal[k0]).applyQuaternion(S.reflQ);
      kickAx = V3().crossVectors(S.n, mp).normalize();
      S.pending.push({ t, ks: [k0] }, { t: t + 0.55, ks: breakOrder.slice(1, 3) }, { t: t + 1.1, ks: breakOrder.slice(3, 5) });
      S.tDetachPlanned = t + 1.6;
    }
    if (S.tDetach === null && S.tDetachPlanned && t >= S.tDetachPlanned) {
      S.tDetach = t; detPos.copy(S.reflPos); detQ.copy(S.reflQ);
      detVel.subVectors(S.d, S.r).normalize().multiplyScalar(0.35).addScaledVector(F.u, 0.25);   // pushed by the beam, away from Earth
      tumbleAx = V3().crossVectors(kickAx, S.n).add(kickAx).normalize();
      for (let i = 0; i < NCH; i++) {                                                              // silver chips break off the rim
        const c = chipSt[i], a = i / NCH * 2 * Math.PI;
        c.t0 = t + 0.02 * (i % 6); c.p0.set(Math.cos(a) * DY.DISC_R, Math.sin(a) * DY.DISC_R, 0.05).applyQuaternion(S.reflQ).add(S.reflPos);
        c.v.copy(detVel).addScaledVector(_a.copy(c.p0).sub(S.reflPos).normalize(), 0.25 + 0.05 * (i % 4)).addScaledVector(S.n, 0.08 * ((i % 3) - 1));
      }
    }
    if (t >= DY.T_END && !S.ended) { S.ended = true; S.running = false; }
    S.status = S.tFail !== null ? 4 : S.maxStress >= 1 ? 3 : t >= DY.T_CONC ? 2 : t >= DY.T_COLLECT ? 1 : 0;
  }

  // ---------------------------------------------------------------- drawing
  const targets = [...Array(8)].map(() => V3()), opens = new Array(8).fill(0.06), restT = [...Array(8)].map(() => V3()), relStart = [...Array(8)].map(() => V3());
  let relCaptured = false;
  const toWorld = (p, out) => out.copy(p).multiplyScalar(ER).add(_ew);
  const _w1 = V3(), _w2 = V3(), _w3 = V3();
  function draw() {
    const t = S.t; earthWorld(_ew); sys.position.copy(_ew);
    optics(t);
    sun.update(t); earth.setSun(_a.copy(_ew).negate().normalize());
    sunKey.position.copy(_ew).addScaledVector(_a, 60); sunKey.target.position.copy(_ew);
    // swarm
    for (let i = 0; i < DY.N_SATS; i++) {
      const o = orb[i]; satPos(o, t, o.pos);
      satQuat(_a.copy(o.pos).negate(), NORTH, _q); _m.compose(o.pos, _q, _sc.setScalar(0.3)); sats.setMatrixAt(i, _m);
    }
    sats.instanceMatrix.needsUpdate = true;
    // energy paths + pulses
    const flow = t < DY.T_COLLECT ? 0.25 : (S.tSafe !== null && t > S.tSafe ? Math.max(0.1, 1 - (t - S.tSafe) / 2) : 1);
    let pi = 0, qi = 0;
    for (let k = 0; k < DY.N_PATHS; k++) {
      const P = orb[pathIdx[k]].pos;
      for (let s = 0; s < PATH_SUB; s++) { pathPoint(P, s / PATH_SUB, _w1, k); pathPoint(P, (s + 1) / PATH_SUB, _w2, k); pathPos.set([_w1.x, _w1.y, _w1.z, _w2.x, _w2.y, _w2.z], 6 * pi++); }
      for (let j = 0; j < PULSES_PER; j++) { const s = ((t * (0.18 + 0.25 * flow) + j / PULSES_PER + k * 0.137) % 1); pathPoint(P, s, _w1, k); pulsePos.set([_w1.x, _w1.y, _w1.z], 3 * qi++); }
    }
    pathGeo.attributes.position.needsUpdate = true; pulseGeo.attributes.position.needsUpdate = true;
    paths.material.opacity = 0.05 + 0.13 * flow; pulses.material.opacity = 0.25 + 0.75 * flow;
    // concentrator aim (emitter along the incident beam) + glow
    _q.setFromUnitVectors(V3(0, 1, 0), S.d); emitter.quaternion.copy(_q);
    concGlow.material.opacity = 0.15 + 0.85 * S.charge; concGlow.scale.setScalar(2.5 + 3 * S.charge);
    // reflector
    refl.position.copy(S.reflPos); refl.quaternion.copy(S.reflQ);
    // incident beam: from the emitter lens towards the reflector (front travels at the start)
    const lensW = _w3.copy(CW).addScaledVector(S.d, 1.7);
    if (S.P > 0) {
      const tOn = t - DY.T_CONC, maxLen = 200, front = Math.min(maxLen, tOn * 60);
      const hitW = toWorld(S.hit, _w1), toHit = lensW.distanceTo(hitW);
      const endLen = S.hitsDisc ? Math.min(front, toHit) : front;
      setBeam(beamIn, lensW, _w2.copy(lensW).addScaledVector(S.d, endLen), S.P, t, S.hitsDisc && front >= toHit);
      if (S.hitsDisc && front >= toHit) {
        const outEnd = toWorld(S.outEnd, _w2), outLen = Math.min(hitW.distanceTo(outEnd), (front - toHit));
        setBeam(beamOut, hitW, _w1.copy(hitW).addScaledVector(_d.copy(S.r), outLen), S.P, t, S.hitsEarth && outLen >= hitW.distanceTo(outEnd) - 1e-3);
      } else setBeam(beamOut, lensW, lensW, 0, t, false);
    } else { setBeam(beamIn, lensW, lensW, 0, t, false); setBeam(beamOut, lensW, lensW, 0, t, false); }
    // arms: claws on the 8 mounts while attached; broken arms fly; released arms recoil to rest
    for (const ev of S.pending) {
      if (!ev.done && t >= ev.t) {
        ev.done = true;
        for (const k of ev.ks) {
          if (S.broken.includes(k)) continue;
          S.broken.push(k);
          const s = 6, p = rig.arms[k].pts[s];
          rig.breakArm(k, s, ev.t, _a.subVectors(rig.arms[k].pts[10], rig.arms[k].base).normalize().multiplyScalar(0.25).addScaledVector(F.u, 0.18), 1.6 + 0.4 * k, 0.5, k);
          rig.spawnFragments(p, 8, ev.t, _b.copy(F.u).multiplyScalar(0.1), 0.45, 0.035, k);
        }
      }
    }
    const detached = S.tDetach !== null && t >= S.tDetach;
    if (detached && !relCaptured) { relCaptured = true; for (let k = 0; k < 8; k++) relStart[k].copy(rig.arms[k].tip); }
    for (let k = 0; k < 8; k++) {
      rig.restTarget(k, t, restT[k]);
      if (!detached) { targets[k].copy(mountsLocal[k]).applyQuaternion(S.reflQ).add(S.reflPos); opens[k] = 0.06; }
      else {
        const k2 = Math.min(1, (t - S.tDetach) / 1.4), e = 1 - Math.exp(-5 * k2) * Math.cos(7 * k2);        // recoil with overshoot
        targets[k].lerpVectors(relStart[k], restT[k], Math.min(1.15, e)); opens[k] = 0.06 + 0.8 * Math.min(1, k2 * 3);
      }
    }
    rig.pose(targets, { open: opens, strain: S.stress, t });
    // reflector chips
    for (let i = 0; i < NCH; i++) {
      const c = chipSt[i];
      if (t < c.t0) { _m.compose(_a.set(0, 0, 0), _q.identity(), _sc.set(0, 0, 0)); chips.setMatrixAt(i, _m); continue; }
      const dt = t - c.t0; _q.setFromAxisAngle(c.ax, c.w * dt); _m.compose(_a.copy(c.p0).addScaledVector(c.v, dt), _q, _sc.setScalar(c.s)); chips.setMatrixAt(i, _m);
    }
    chips.instanceMatrix.needsUpdate = true;
    // force arrow
    const fr = S.Fmag / (DY.P_MAX / C_LIGHT);
    arrow.visible = fr > 0.003;
    if (arrow.visible) {
      _d.copy(S.F).normalize(); const L = 1.2 + 6 * fr;
      arrow.position.copy(_ew).addScaledVector(_d, ER * 1.08); arrow.quaternion.setFromUnitVectors(V3(0, 1, 0), _d);
      shaft.scale.set(1, L, 1); shaft.position.y = L / 2; head.position.y = L + 0.3;
    }
    earth.update(0);
  }

  // ---------------------------------------------------------------- camera presets, labels, HUD
  const eW = () => earthWorld(V3());
  const CAM_DIR = V3().copy(n0).multiplyScalar(0.6).add(V3(0, 0, 0.65)).addScaledVector(V3().crossVectors(NORTH, F.u).normalize(), 0.45).normalize();
  const presets = {
    overview: () => ({ pos: [38, -88, 44], target: [36, 0, 0] }),
    swarm: () => ({ pos: [-4, -40, 17], target: [6, 0, 0] }),
    concentrator: () => ({ pos: [CW.x - 9, -12, 6], target: [CW.x + 4, 0, 0] }),
    beam: () => { const e = eW(); return { pos: [(CW.x + e.x) / 2, -34, 13], target: [(CW.x + e.x) / 2 + 4, 0, 0] }; },
    // looking at the mirror's front face from above / the Sun side, so Earth does not hide the reflector
    reflector: () => { const e = eW(), c = V3().copy(Rc).multiplyScalar(0.55 * ER).add(e), p = V3().copy(c).addScaledVector(CAM_DIR, 15); return { pos: p.toArray(), target: c.toArray() }; },
    failure: () => { const e = eW(), c = V3().copy(Rc).multiplyScalar(0.7 * ER).add(e), p = V3().copy(c).addScaledVector(CAM_DIR, 24).add(V3(0, -6, 0)); return { pos: p.toArray(), target: c.toArray() }; },
    earth: () => { const e = eW(), p = V3().copy(e).addScaledVector(CAM_DIR, 21).add(V3(0, -9, 0)); return { pos: p.toArray(), target: [e.x + 1.5, e.y + 1.5, e.z] }; },
  };
  const cam = createCineCamera(camera, controls, presets, (on) => { S.autoCam = on; S.lastPanel = -1; });
  cam.go('overview', 0);
  function autoPreset() {
    const t = S.t;
    if (!S.started) return 'overview';
    if (t < DY.T_COLLECT) return 'swarm';
    if (t < DY.T_CONC) return 'concentrator';
    if (t < DY.T_ALIGN + 1) return 'beam';
    if (S.tFail === null) return 'reflector';
    if (t < S.tFail + 7) return 'failure';
    return 'earth';
  }
  const lbl = createLabels(container, canvas, camera, { sun: '#ffcc33', swarm: '#e0e4ea', conc: '#ffe2a8', beam: '#fff1d0', refl: '#dfe9ff', earth: '#9fd0ff', force: '#a8f0a0', out: '#fff1d0' });
  const panel = document.createElement('div'); panel.id = 'dyPanel'; panel.className = 'small cinepanel'; panel.hidden = true; container.appendChild(panel);
  panel.addEventListener('click', (e) => {
    const b = e.target.closest('[data-dy]'); if (!b) return;
    const a = b.dataset.dy;
    if (a === 'start') start(); else if (a === 'pause') togglePause(); else if (a === 'reset') reset(); else if (a === 'replay') replayFailure();
    else if (a === 'auto') { cam.setAuto(true); cam.go(autoPreset()); }
    else if (presets[a]) { cam.setAuto(false); cam.go(a); }
    S.lastPanel = -1;
  });
  function renderPanel() {
    const st = STATUS[S.status], t = S.t;
    const stCls = S.status >= 4 ? 'bad' : S.status === 3 ? 'warn' : S.status === 2 ? 'good' : '';
    const intact = 8 - S.broken.length, released = S.tDetach !== null ? intact : 0;
    const concTxt = !S.started ? 'idle' : S.tSafe !== null && t > S.tSafe ? (S.P > 0 ? 'SAFING - shutting the beam down' : 'SAFED (beam shut down)') : t < DY.T_COLLECT ? `charging ${Math.round(100 * S.charge)} %` : t < DY.T_CONC ? `concentrating ${Math.round(100 * S.charge)} %` : `emitting · ${Math.round(100 * S.charge)} %`;
    const beamTxt = S.P > 0 ? `${fmtE(S.P * DY.P_MAX, 2)} W (${Math.round(100 * S.P)} % of P<sub>max</sub>)` : 'off';
    const alTxt = !S.started || t < DY.T_CONC ? 'stowed (facing Earth)' : !S.hitsDisc ? 'LOST - beam misses the reflector' : !S.hitsEarth ? `MISALIGNED ${S.alignErr.toFixed(1)}° - reflected beam misses Earth` : t < DY.T_ALIGN ? `aligning · error ${S.alignErr.toFixed(2)}°` : (S.alignErr > 3 ? `OFF-CENTRE ${S.alignErr.toFixed(1)}° (still hits Earth)` : `locked on Earth · error ${S.alignErr.toFixed(2)}°`);
    const dispKm = S.x_m / 1000, simYr = Math.max(0, t - DY.T_CONC) * DY.YEARS_PER_S;
    const armTxt = S.tFail === null ? `${intact}/8 intact · peak load ${Math.round(100 * S.maxStress)} % of capacity` : S.tDetach === null ? `${S.broken.length} of 8 BROKEN · cascade` : `${S.broken.length} broken, ${released} released · reflector lost`;
    const runTxt = S.started && !S.running && !S.ended ? 'Resume' : 'Pause';
    const h = `<div class="cineh">DYSON ENERGY TRANSFER <span class="dim">(conceptual animation)</span></div>
      <div class="btns small"><button data-dy="start" title="Run the sequence from the start.">Start</button><button data-dy="pause" title="Freeze / continue the animation (the camera still moves).">${runTxt}</button><button data-dy="reset" title="Back to the beginning (everything rebuilt from the same deterministic state).">Reset</button><button data-dy="replay" title="Jump to just before the overload and play the failure again.">Replay failure</button></div>
      <div class="cstage ${stCls}">${st}${S.running || !S.started ? '' : S.ended ? ' · sequence complete' : ' · PAUSED'}</div>
      ${hudRow('Collector array', `${DY.N_SATS} satellites drawn · ${!S.started ? 'idle' : S.tSafe !== null && t > S.tSafe + 2 ? 'idle' : 'harvesting'}`, 'The Dyson swarm: collector satellites on orbits around the Sun (instanced; a real swarm would have millions). Energy paths run into the concentrator.')}
      ${hudRow('Concentrator', concTxt, 'Collects the swarm power and emits ONE beam aimed at the reflector centre.')}
      ${hudRow('Incident beam', beamTxt, 'Beam power ramps from 5 % (alignment) to 100 % = 1.29e25 W (the E2 beam power of the report).')}
      ${hudRow('Reflector alignment', alTxt, 'Error = angle between the actual reflected ray r = d - 2(d.n)n and the direction to Earth’s centre. n is the actual mirror normal (including wobble and failure).', S.hitsEarth || !S.started || t < DY.T_CONC ? '' : 'bad')}
      ${hudRow('Force on Earth', `${fmtE(S.Fmag, 2)} N`, 'Total force on Earth + mechanism: (P/c)(d - r) through the arms + (P/c) r where the reflected beam is absorbed = (P/c) d when aligned.')}
      ${hudRow('Earth displacement', `${fmtE(dispKm, 0)} km · Δv ${S.v_ms.toFixed(3)} m/s`, 'Integrated from a = F / M_earth with the animation time compressed (1 s = 0.25 yr). Drawn at true scale relative to Earth’s size (not exaggerated). For comparison: Earth’s orbital speed is 29,780 m/s.')}
      ${hudRow('Needed for 2.7 AU', `${fmtE(F_REQ, 2)} N · full beam = ${(100 * DYSON.frac).toFixed(1)} %`, 'Average push for 1 -> 2.7 AU in 10,000 years (F_req = M dv / T, dv = 11,658 m/s). Even with all 8 arms intact the full beam gives only this fraction.', 'bad')}
      ${hudRow('Earth heating (if absorbed)', `~${fmtE(DYSON.Tsurf, 0)} K`, 'The reflected beam is absorbed by Earth: global energy balance T = 288 K ((S + P)/S)^(1/4) - see model 1 (laser push).', 'bad')}
      ${hudRow('Simulated time', `${simYr.toFixed(2)} yr (compressed, 1 s = ${DY.YEARS_PER_S} yr)`, 'Simulated time since the beam came on - NOT wall-clock time.')}
      ${hudRow('Arm structural status', armTxt, `Each arm carries its share of |F_arms| weighted by its reach. Capacity is an ASSUMED stylized threshold that the most loaded arm reaches at ${DY.ARM_CAP_FRAC * 100} % of full power (${fmtE(ARM_CAP, 2)} N per arm); failure at ${Math.round(DY.FAIL_STRESS * 100)} %.`, S.status >= 4 ? 'bad' : S.status === 3 ? 'warn' : '')}
      <div class="crow"><span class="cl">Peak arm load</span><span class="cv">${hudBar(S.maxStress / DY.FAIL_STRESS, S.maxStress >= 1 ? '#ff6a4d' : S.maxStress > 0.75 ? '#ffb44d' : '#7fc8ff')}</span></div>
      <div class="btns small cam">Camera: <button data-dy="overview">overview</button><button data-dy="swarm">swarm</button><button data-dy="concentrator">concentrator</button><button data-dy="beam">beam</button><button data-dy="reflector">reflector</button><button data-dy="earth">Earth</button><button data-dy="auto" class="${S.autoCam ? 'on' : ''}" title="Automatic cinematic camera (dragging the view turns it off).">auto</button></div>
      <div class="csub dim">Conceptual: Sun, swarm, concentrator and Earth distances are not to scale; Earth, the 8 arms (2 R<sub>E</sub>) and the reflector (diameter ${fmtE(2 * DY.DISC_R * 6371, 0)} km) are to scale with each other. Full power: F = P/c = ${fmtE(DY.P_MAX / C_LIGHT, 2)} N, a = ${fmtE(DY.P_MAX / C_LIGHT / M_EARTH, 2)} m/s<sup>2</sup> = ${(DY.P_MAX / C_LIGHT / M_EARTH * YEAR_S).toFixed(2)} m/s per year. The arm capacity, cascade and fragments are illustrative. Even if the arms held, the full beam gives ${(100 * DYSON.frac).toFixed(1)} % of the ${fmtE(F_REQ, 2)} N needed, and the absorbed ${fmtE(DY.P_MAX, 2)} W would heat Earth to ~${fmtE(DYSON.Tsurf, 0)} K.</div>`;
    if (panel.innerHTML !== h) panel.innerHTML = h;
  }
  function placeLabels() {
    const e = earthWorld(V3());
    lbl.set('sun', V3(0, 0, DY.SUN_R + 2.5), 'Sun (not to scale)');
    lbl.set('swarm', orb[17].pos, `Dyson swarm · ${DY.N_SATS} collectors drawn`);
    lbl.set('conc', V3().copy(CW).add(V3(0, 0, 2.2)), 'concentrator');
    lbl.set('beam', beamIn.g.visible ? V3().lerpVectors(beamIn.a, beamIn.b, 0.55).add(V3(0, 0, 1.0)) : null, 'incident beam (one)');
    lbl.set('refl', V3().copy(S.reflPos).multiplyScalar(ER).add(e).add(V3(0, 0, ER * 1.25)), S.tDetach !== null ? 'reflector - torn loose, tumbling' : 'reflector · held at 8 mounts by the 8 arms');
    lbl.set('earth', V3().copy(e).add(V3(0, 0, -ER * 1.25)), 'Earth');
    lbl.set('force', arrow.visible ? V3().copy(arrow.position).addScaledVector(_d.copy(S.F).normalize(), 3) : null, `F = ${fmtE(S.Fmag, 2)} N`);
    lbl.set('out', beamOut.g.visible ? V3().lerpVectors(beamOut.a, beamOut.b, 0.5).add(V3(0, 0, -0.8)) : null, S.hitsEarth ? 'reflected beam → Earth' : 'reflected beam (misaligned)');
    lbl.declutter();
  }

  // ---------------------------------------------------------------- controls
  let acc = 0; const H = 1 / 120;
  function reset() {
    Object.assign(S, {
      running: false, started: false, t: 0, status: 0, P: 0, charge: 0, x_m: 0, v_ms: 0, Fmag: 0, Fabs: 0, Farms: 0, maxStress: 0,
      tOver: null, tFail: null, tDetach: null, tDetachPlanned: null, tSafe: null, broken: [], pending: [], ended: false, lastPanel: -1,
    });
    S.stress.fill(0); S.F.set(0, 0, 0); acc = 0; relCaptured = false;
    for (const c of chipSt) c.t0 = 1e9;
    rig.reset(); optics(0);
    if (S.autoCam) cam.go('overview', 1.2);
  }
  function start() { reset(); S.started = true; S.running = true; if (S.autoCam) cam.go('swarm', 2.2); }
  function togglePause() { if (!S.started) { start(); return; } if (S.ended) return; S.running = !S.running; S.lastPanel = -1; }
  /** Rebuild the deterministic state up to shortly before the overload, then play on. */
  function replayFailure() {
    reset(); S.started = true;
    const tTo = DY.T_ALIGN + (DY.T_RAMP_END - DY.T_ALIGN) * ((DY.ARM_CAP_FRAC - DY.P_ALIGN) / (1 - DY.P_ALIGN)) - 2.0;   // ~2 s before the overload
    while (S.t < tTo) step(H);
    S.running = true; S.lastPanel = -1;
    if (S.autoCam) cam.go('reflector', 1.0);
  }
  function update(dt) {
    dt = Math.min(0.1, Math.max(0, dt));
    if (S.running) { acc += dt; let n = 0; while (acc >= H && n < 30) { step(H); acc -= H; n++; } if (n >= 30) acc = 0; }
    if (S.autoCam && !cam.state.tr) { const want = autoPreset(); if (cam.state.current !== want) cam.go(want, 2.4); }
    cam.update(dt);
    draw();
    S.wall += dt; if (S.wall - S.lastPanel > 0.25 || S.lastPanel < 0) { S.lastPanel = S.wall; renderPanel(); }
  }
  function render() { renderShifted(renderer, scene, camera, canvas, controls, panel); placeLabels(); }
  function setActive(on) { S.active = on; controls.enabled = on; panel.hidden = !on; if (!on) lbl.hide(); if (on) S.lastPanel = -1; }
  function setLabels(on) { lbl.state.on = on; if (!on) lbl.hide(); }
  reset();
  return {
    scene, camera, controls, S, rig, update, render, setActive, setLabels, hideLabels: lbl.hide, start, reset, togglePause, replayFailure, panel, cam,
    counts: () => ({ beams: [beamIn, beamOut].length, reflectors: 1, sats: sats.count, sceneChildren: scene.children.length, sysChildren: sys.children.length }),
    geometry: () => checkDysonGeometry(S.x_m / RE_M), lever, ARM_CAP,
  };
}
