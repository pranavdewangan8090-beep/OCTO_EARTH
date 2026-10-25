// OCTO-EARTH failed attempt 2 - "Hydrogen harvesting" (update p38; replaces the p37 arm-net animation).
// Independent model: no launcher arms, no asteroid.
//
// Idea: a magnetic ram-scoop (a superconducting ring whose field funnels charged particles) sits sunward of Earth,
// outside the magnetosphere, and collects solar-wind hydrogen (protons). The hydrogen is fused to helium in a fusion
// drive on Earth; the exhaust pushes Earth along its orbit. Everything is IDEAL: every proton crossing the scoop
// area is caught, all of it is fused, and the full fusion energy (0.71 % of the mass) becomes exhaust speed:
//     v_e = c sqrt(2 * 0.0071) = 3.6e7 m/s (12 % of light speed)
//     hydrogen needed for dv = 11,658 m/s (rocket equation):  m = M_earth (exp(dv / v_e) - 1) = 1.9e21 kg
//     collected:  mdot = rho_sw v_sw pi R^2,   rho_sw v_sw = 5 protons/cm^3 * m_p * 400 km/s = 3.3e-15 kg/m^2/s
// A 100 Earth-radius scoop collects 4.3 t/s -> 1.3e15 kg in 10,000 years (7e-7 of the need). Even the ENTIRE solar wind
// of the Sun (1.5e9 kg/s, all directions) for 10,000 years is only 24 % of the need (a = 1.22 AU).
//
// Units: Earth radii, Earth at the origin, Sun towards -x, Earth's orbital motion +y. The scoop is drawn at 20 R_E with a
// drawn radius that grows with log(R) - NOT to scale. Protons are sparse markers (each stands for a huge number of real
// protons); most never reach the scoop and flow around the bow shock. Time-lapse: 30 s of animation = 10,000 years.
import * as THREE from './vendor/three.module.js';
import { makeBase, createArrow, shueSurface, shue, V3, clamp, lerp, hash } from './failKit.js';
import { glowTex } from './collector.js';
import { hudRow, hudBar, fmtE } from './cineKit.js';
import { K, HYD, DV_REQ, F_REQ, hydrogen, hydrogenWholeSun } from './failedPhysics.js';

export const HZ = { T_ANIM: 30, X_SCOOP: -20, X_TANK: -16.6, X_START: -48, X_END: 40, N_P: 170, FLOW_T: 9, MP_R0: 10, MP_A: 0.58, BS_R0: 13, BS_A: 0.75 };
const drawnR = (R) => clamp(2.2 + 1.2 * Math.log10(Math.max(R, 1)), 3.2, 7.4);

export function createHydrogenScoopView(renderer, canvas, container) {
  const ctl = `<div class="fa-slider" title="Radius of the scoop's collecting area in Earth radii (logarithmic slider)."><span class="dim">Scoop radius</span> <input type="range" id="hzR" min="1" max="4" step="0.01" value="2"> <b class="mono" id="hzRv"></b></div>
    <div class="btns small"><button data-hz="whole" title="Compare: collect EVERY proton the Sun emits in all directions (an impossible collector enclosing the Sun).">Whole solar wind (impossible)</button></div>`;
  const B = makeBase(renderer, canvas, container, {
    id: 'hz', title: '2 &middot; HYDROGEN HARVESTING <span class="dim">(ram-scoop + fusion drive)</span>', controlsHtml: ctl,
    cam: { pos: [-7, -40, 18], target: [-9, 0, 0] }, maxDist: 220,
    labels: { scoop: '#c9b6ff', tank: '#9fd8ff', earth: '#9fd0ff', mp: '#7fb0ff', bs: '#ffb48a', wind: '#ffe6a0', drive: '#ffd18a' },
    note: 'Ideal case: every proton crossing the scoop is caught, fully fused, and all fusion energy becomes exhaust speed. Scoop size and distance NOT to scale (drawn radius grows with log R); proton dots are sparse markers. Solar wind: 5 protons/cm^3 at 400 km/s (typical near 1 AU).',
  });
  const { scene, earth } = B;
  const S = { t: 0, running: false, started: false, ended: false, R: 100, whole: false, wall: 0, lastDyn: -1 };

  // ---------------------------------------------------------------- magnetosphere + bow shock
  const mp = shueSurface(HZ.MP_R0, HZ.MP_A, 0x4f86ff, 0.16); scene.add(mp);
  const bs = shueSurface(HZ.BS_R0, HZ.BS_A, 0xff9a5a, 0.12, 2.0); scene.add(bs);
  const rhoShock = (x) => {                            // radius of the bow shock surface at position x (Sun at -x)
    if (x < -HZ.BS_R0) return 0; let lo = 0, hi = 2.6;
    for (let i = 0; i < 30; i++) { const m = (lo + hi) / 2; if (-shue(HZ.BS_R0, HZ.BS_A, m) * Math.cos(m) > x) hi = m; else lo = m; }
    const th = (lo + hi) / 2; return shue(HZ.BS_R0, HZ.BS_A, th) * Math.sin(th);
  };
  const RHO = []; for (let i = 0; i <= 200; i++) RHO.push(rhoShock(HZ.X_START + (HZ.X_END - HZ.X_START) * i / 200));
  const rhoAt = (x) => { const f = clamp((x - HZ.X_START) / (HZ.X_END - HZ.X_START), 0, 1) * 200, i = Math.min(199, Math.floor(f)); return lerp(RHO[i], RHO[i + 1], f - i); };

  // ---------------------------------------------------------------- scoop ring, funnel field, tank, fuel line, drive
  const scoop = new THREE.Group(); scoop.position.x = HZ.X_SCOOP; scene.add(scoop);
  const ringMat = new THREE.MeshStandardMaterial({ color: 0xb8c0cc, metalness: 0.8, roughness: 0.3, emissive: 0x3a2a90, emissiveIntensity: 0.6 });
  const ring = new THREE.Mesh(new THREE.TorusGeometry(1, 0.045, 16, 120), ringMat); ring.rotation.y = Math.PI / 2; scoop.add(ring);
  const tank = new THREE.Group(); tank.position.x = HZ.X_TANK; scene.add(tank);
  const shell = new THREE.Mesh(new THREE.CylinderGeometry(0.9, 0.9, 2.6, 40, 1, true), new THREE.MeshStandardMaterial({ color: 0x9fb4cc, metalness: 0.3, roughness: 0.2, transparent: true, opacity: 0.28, side: THREE.DoubleSide, depthWrite: false }));
  shell.rotation.z = Math.PI / 2; tank.add(shell);
  for (const s of [-1, 1]) { const cap = new THREE.Mesh(new THREE.CircleGeometry(0.9, 40), new THREE.MeshStandardMaterial({ color: 0x8090a4, metalness: 0.7, roughness: 0.4, side: THREE.DoubleSide })); cap.position.x = s * 1.3; cap.rotation.y = Math.PI / 2; tank.add(cap); }
  const fill = new THREE.Mesh(new THREE.CylinderGeometry(0.86, 0.86, 1, 40), new THREE.MeshBasicMaterial({ color: 0x5fc8ff, transparent: true, opacity: 0.85 }));
  fill.rotation.z = Math.PI / 2; tank.add(fill);
  const struts = new THREE.Group(); scene.add(struts);
  const strutMat = new THREE.MeshStandardMaterial({ color: 0x8a93a2, metalness: 0.6, roughness: 0.4 });
  const strutMeshes = []; for (let k = 0; k < 4; k++) { const m = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 1, 6), strutMat); struts.add(m); strutMeshes.push(m); }
  const fieldLines = new THREE.Group(); scene.add(fieldLines);
  const fieldMat = new THREE.LineBasicMaterial({ color: 0xa48cff, transparent: true, opacity: 0.45 });
  const fuelGeo = new THREE.BufferGeometry().setFromPoints([V3(HZ.X_TANK + 1.3, 0, 0), V3(-4, -3, 0.5), V3(0, -1.05, 0)]);
  const fuel = new THREE.Line(fuelGeo, new THREE.LineDashedMaterial({ color: 0x6fd0ff, dashSize: 0.4, gapSize: 0.3, transparent: true, opacity: 0.7 })); fuel.computeLineDistances(); scene.add(fuel);
  const drive = new THREE.Group(); drive.position.set(0, -1.0, 0); scene.add(drive);   // nozzle points -y (push +y)
  const nozzle = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.11, 0.16, 20, 1, true), new THREE.MeshStandardMaterial({ color: 0xc0c6d0, metalness: 0.8, roughness: 0.3, side: THREE.DoubleSide }));
  nozzle.position.y = -0.06; drive.add(nozzle);
  const flame = new THREE.Mesh(new THREE.ConeGeometry(0.08, 1, 20, 1, true), new THREE.MeshBasicMaterial({ color: 0xffc070, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false }));
  flame.rotation.z = Math.PI; drive.add(flame);
  const flameGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex('rgba(255,220,160,1)', 'rgba(255,140,60,0)'), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  flameGlow.position.y = -0.16; drive.add(flameGlow);
  const thrustArrow = createArrow(0xa8f0a0, 0.04); scene.add(thrustArrow.group);
  const sunArrow = createArrow(0xffe6a0, 0.05); scene.add(sunArrow.group); sunArrow.set(V3(-46, 12, 9), V3(1, 0, 0), 7);

  // ---------------------------------------------------------------- protons (pure function of time)
  const NP = HZ.N_P, pp = new Float32Array(NP * 3), pc = new Float32Array(NP * 3);
  const pgeo = new THREE.BufferGeometry(); pgeo.setAttribute('position', new THREE.BufferAttribute(pp, 3)); pgeo.setAttribute('color', new THREE.BufferAttribute(pc, 3));
  const pts = new THREE.Points(pgeo, new THREE.PointsMaterial({ size: 7, sizeAttenuation: false, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, map: glowTex('rgba(255,255,255,1)', 'rgba(255,255,255,0)') }));
  pts.frustumCulled = false; scene.add(pts);
  const P = []; for (let i = 0; i < NP; i++) P.push({ b: 17 * Math.sqrt(hash(i, 7)), ph: 6.2832 * hash(i, 8), ofs: hash(i, 9), sp: 0.85 + 0.3 * hash(i, 10) });
  /** Radius of a funnelled path (impact radius b, capture radius Rc, ring radius Rd) at position x. */
  function funnelRho(b, Rc, Rd, x) {
    const xs = HZ.X_SCOOP, x0 = xs - 12;
    if (x <= x0) return b;
    const rRing = (b / Rc) * Rd * 0.9;
    if (x <= xs) { const k = (x - x0) / (xs - x0), s = k * k * (3 - 2 * k); return lerp(b, rRing, s); }
    const k = clamp((x - xs) / (HZ.X_TANK - xs), 0, 1), s = k * k * (3 - 2 * k); return lerp(rRing, 0.25 * b / Rc, s);
  }
  function buildField(Rd) {
    for (const c of fieldLines.children) c.geometry.dispose(); fieldLines.clear();
    const Rc = 1.55 * Rd;
    for (let k = 0; k < 16; k++) {
      const ph = (k + (k >= 8 ? 0.5 : 0)) * Math.PI / 4, b = Rc * (k >= 8 ? 0.62 : 0.98), pts3 = [];
      for (let i = 0; i <= 60; i++) { const x = lerp(HZ.X_START + 4, HZ.X_TANK, i / 60), r = funnelRho(b, Rc, Rd, x); pts3.push(V3(x, r * Math.cos(ph), r * Math.sin(ph))); }
      fieldLines.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts3), fieldMat));
    }
  }
  let builtRd = -1;
  function layout() {
    const Rd = S.whole ? 7.4 : drawnR(S.R);
    if (Math.abs(Rd - builtRd) < 1e-3) return Rd; builtRd = Rd;
    ring.scale.setScalar(Rd); buildField(Rd);
    for (let k = 0; k < 4; k++) {
      const ph = k * Math.PI / 2 + Math.PI / 4, a = V3(HZ.X_SCOOP, Rd * Math.cos(ph), Rd * Math.sin(ph)), b = V3(HZ.X_TANK - 1.3, 0.8 * Math.cos(ph), 0.8 * Math.sin(ph));
      const m = strutMeshes[k], d = V3().subVectors(b, a); m.position.copy(a).addScaledVector(d, 0.5); m.scale.set(1, d.length(), 1); m.quaternion.setFromUnitVectors(V3(0, 1, 0), d.normalize());
    }
    return Rd;
  }

  function model() { return S.whole ? hydrogenWholeSun() : hydrogen(S.R); }
  function state() {
    const h = model(), yrs = (S.started ? clamp(S.t / HZ.T_ANIM, 0, 1) : 0) * K.T_MISSION_Y, ts = yrs * K.YEAR;
    const m = h.mdot * ts, dv = HYD.VE * Math.log(1 + m / K.M_E);
    const a = 1 / (1 - dv / Math.sqrt(K.GM_SUN / K.AU)) ** 2;
    return { h, yrs, ts, m, dv, a, frac: m / HYD.M_NEED, thrust: h.mdot * HYD.VE };
  }
  function draw(dt) {
    const Rd = layout(), s = state(), tv = S.t + S.idle;
    const Rc = 1.55 * Rd; let caught = 0;
    for (let i = 0; i < NP; i++) {
      const p = P[i], cap = p.b < Rc, xEnd = cap ? HZ.X_TANK : HZ.X_END, len = xEnd - HZ.X_START;
      const u = ((tv * p.sp / HZ.FLOW_T + p.ofs) % 1 + 1) % 1, x = HZ.X_START + u * (HZ.X_END - HZ.X_START);
      let r, col;
      if (cap && x > xEnd) { r = 0; pp.set([HZ.X_TANK, 0, 0], 3 * i); pc.set([0, 0, 0], 3 * i); continue; }
      if (cap) { r = funnelRho(p.b, Rc, Rd, x); col = x > HZ.X_SCOOP - 12 ? [0.55, 0.85, 1.0] : [1.0, 0.92, 0.65]; if (x > HZ.X_SCOOP) caught++; }
      else { const rs = rhoAt(x) + 0.5; r = p.b >= rs ? p.b : Math.sqrt(rs * rs + 0.15 * p.b * p.b); col = r > p.b + 0.05 ? [1.0, 0.55, 0.3] : [1.0, 0.8, 0.35]; }
      const fade = Math.min(1, (x - HZ.X_START) / 4, (xEnd - x) / 3 + (cap ? 0.15 : 0));
      pp.set([x, r * Math.cos(p.ph), r * Math.sin(p.ph)], 3 * i); pc.set(col.map((c) => c * clamp(fade, 0, 1)), 3 * i);
    }
    pgeo.attributes.position.needsUpdate = true; pgeo.attributes.color.needsUpdate = true;
    // tank fill: LINEAR fraction of the hydrogen needed (a thin sliver is always drawn so the tank is visibly "almost empty")
    const fl = clamp(s.frac, 0.004, 1); fill.scale.set(1, 2.5 * fl, 1); fill.position.x = -1.25 + 1.25 * fl;
    const on = S.started && s.m > 0;
    const thrustLen = on ? clamp(0.12 + 0.07 * Math.log10(Math.max(1, s.thrust / 1e9)), 0.12, 0.9) : 0;
    flame.visible = flameGlow.visible = on; flame.scale.set(1, thrustLen, 1); flame.position.y = -0.14 - thrustLen / 2;
    flameGlow.scale.setScalar(0.25 + thrustLen * 0.6);
    thrustArrow.set(V3(0, 1.1, 0), V3(0, 1, 0), on ? 0.4 + 2.2 * clamp(s.thrust / F_REQ, 0, 1) : 0);
    ringMat.emissiveIntensity = S.started ? 0.9 + 0.2 * Math.sin(tv * 3) : 0.35;
    fuel.material.opacity = on ? 0.75 : 0.2;
    earth.update(S.running ? dt : 0);
    S.caughtVis = caught;
    return s;
  }

  // ---------------------------------------------------------------- panel
  const slider = B.panel.querySelector('#hzR'), rv = B.panel.querySelector('#hzRv'), wholeBtn = B.panel.querySelector('[data-hz="whole"]');
  slider.addEventListener('input', () => { S.R = Math.pow(10, +slider.value); if (S.whole) { S.whole = false; wholeBtn.classList.remove('on'); } S.lastDyn = -1; dyn(); });
  wholeBtn.addEventListener('click', () => { S.whole = !S.whole; wholeBtn.classList.toggle('on', S.whole); S.lastDyn = -1; dyn(); });
  function dyn() {
    const s = state(), h = s.h, end = hydrogen(S.R), wsun = hydrogenWholeSun();
    rv.textContent = S.whole ? '(whole Sun)' : `${fmtE(S.R, 0)} R_E`;
    if (!S.started) B.setStage('READY - press Start', '');
    else if (S.ended) B.setStage(`10,000 YEARS: ${fmtE(100 * s.frac, 2)} % OF THE FUEL NEEDED - Earth at ${s.a.toFixed(s.a < 1.001 ? 6 : 3)} AU`, 'bad');
    else B.setStage(`COLLECTING - ${Math.round(s.yrs).toLocaleString('en-US')} years` + (S.running ? '' : ' - PAUSED'), 'warn');
    const logF = clamp(Math.log10(Math.max(1, s.m)) / Math.log10(HYD.M_NEED), 0, 1);
    let x = '';
    x += hudRow('Simulated time', `${Math.round(s.yrs).toLocaleString('en-US')} of 10,000 years`, 'Time-lapse: 30 s of animation = 10,000 years (simulated, not wall-clock).');
    x += hudRow('Collector', S.whole ? 'every proton the Sun emits' : `scoop radius ${fmtE(S.R, 0)} R<sub>E</sub> = ${fmtE(S.R * K.R_E / 1000, 0)} km`, 'Area that catches the wind: pi R^2. The drawn ring is not to scale.');
    x += hudRow('Solar wind', `${HYD.N_CM3} protons/cm<sup>3</sup> at ${HYD.V_SW / 1000} km/s`, `Mass flux ${fmtE(HYD.MASS_FLUX, 2)} kg per m^2 per second. Interstellar hydrogen is ~1000 x thinner still.`);
    x += hudRow('Hydrogen collected', `${fmtE(h.mdot, 2)} kg/s`, S.whole ? 'The Sun loses ~1.5e9 kg/s as solar wind in all directions.' : 'mdot = mass flux x pi R^2 (every proton caught).');
    x += hudRow('In the tank', `${fmtE(s.m, 2)} kg`, 'Collected mass so far.');
    x += hudRow('Needed', `${fmtE(HYD.M_NEED, 2)} kg`, `Rocket equation with the ideal fusion exhaust v_e = ${fmtE(HYD.VE, 2)} m/s: m = M (exp(dv/v_e) - 1), dv = ${DV_REQ.toFixed(0)} m/s.`);
    x += `<div class="crow" title="Linear fraction of the hydrogen needed."><span class="cl">Tank (linear)</span><span class="cv">${fmtE(100 * s.frac, 2)} % ${hudBar(s.frac, '#5fc8ff')}</span></div>`;
    x += `<div class="crow" title="Logarithmic gauge: 0 = 1 kg, full = the mass needed. Each step of 1/21 is a factor of 10."><span class="cl">Tank (log scale)</span><span class="cv">${(21.3 * logF).toFixed(1)} of 21.3 decades ${hudBar(logF, '#a48cff')}</span></div>`;
    x += hudRow('Fusion drive thrust', `${fmtE(s.thrust, 2)} N = ${fmtE(100 * s.thrust / F_REQ, 2)} % of F<sub>req</sub>`, 'Thrust = mdot v_e (burning the hydrogen as fast as it arrives).', 'bad');
    x += hudRow('Speed gained', `${fmtE(s.dv, 3)} of ${DV_REQ.toFixed(0)} m/s`, 'dv = v_e ln(1 + m / M_earth).', 'bad');
    x += hudRow('Orbit now', `${s.a.toFixed(s.a < 1.001 ? 7 : 4)} AU (target 2.7)`, 'Slow spiral: orbit radius from the speed gained.', 'bad');
    x += hudRow('Wind push on the scoop', `${fmtE(h.windPush, 2)} N`, 'Momentum of the caught wind (mdot v_sw). Points away from the Sun (radial), so it barely changes the orbit size; negligible anyway.');
    x += `<div class="fa-sec">After 10,000 years</div>`;
    x += hudRow(`Scoop ${fmtE(S.R, 0)} R<sub>E</sub>`, `${fmtE(100 * end.frac, 2)} % of need &middot; ${end.a10k.toFixed(end.a10k < 1.001 ? 6 : 3)} AU`, 'Ideal collection for the full mission.', 'bad');
    x += hudRow('Whole solar wind', `${(100 * wsun.frac).toFixed(0)} % of need &middot; ${wsun.a10k.toFixed(3)} AU`, 'Every proton the Sun emits, all directions, for 10,000 years.', 'bad');
    x += `<div class="fa-verdict">FAILS: space is too empty. Even catching the Sun's entire wind for 10,000 years gives 24 % of the fuel (Earth reaches ${wsun.a10k.toFixed(2)} AU, not 2.7).</div>`;
    B.setDyn(x);
  }
  function placeLabels() {
    const Rd = S.whole ? 7.4 : drawnR(S.R);
    B.lbl.set('scoop', V3(HZ.X_SCOOP, 0, Rd + 0.8), S.whole ? 'collector (stands for: every proton of the Sun)' : `magnetic scoop, radius ${fmtE(S.R, 0)} R_E (not to scale)`);
    B.lbl.set('tank', V3(HZ.X_TANK, 0, -1.6), 'hydrogen tank (fill = linear)');
    B.lbl.set('earth', V3(0, 0, -1.5), 'Earth');
    B.lbl.set('drive', S.started ? V3(0, -1.6, 0.4) : null, 'fusion drive (thrust tiny)');
    B.lbl.set('mp', V3(-2, 0, 10.6), 'magnetopause');
    B.lbl.set('bs', V3(-4, 0, 14.5), 'bow shock - most of the wind flows around');
    B.lbl.set('wind', V3(-46, 12, 10.5), 'solar wind (from the Sun)');
    B.lbl.declutter();
  }
  function reset() { Object.assign(S, { t: 0, running: false, started: false, ended: false, lastDyn: -1, idle: 0 }); draw(0); dyn(); }
  function start() { reset(); S.started = true; S.running = true; }
  function togglePause() { if (!S.started || S.ended) { start(); return; } S.running = !S.running; S.lastDyn = -1; }
  function update(dt) {
    dt = Math.min(0.1, Math.max(0, dt));
    if (S.running) { S.t += dt; if (S.t >= HZ.T_ANIM) { S.t = HZ.T_ANIM; S.running = false; S.ended = true; S.lastDyn = -1; } }
    else if (!S.started) S.idle += dt;            // before Start the wind already flows (nothing is collected)
    draw(dt); B.updateCam(dt);
    S.wall += dt; if (S.lastDyn < 0 || S.wall - S.lastDyn > 0.25) { S.lastDyn = S.wall; dyn(); }
  }
  function render() { B.render(); placeLabels(); }
  S.idle = 0; reset();
  return {
    key: 'hydrogen', S, B, update, render, start, reset, togglePause, setActive: B.setActive, panel: B.panel, state,
    info: {
      title: 'Hydrogen harvesting (ram-scoop + fusion drive)',
      idea: 'A magnetic scoop outside the magnetosphere funnels solar-wind hydrogen into a tank; a fusion drive on Earth burns it (0.12 c exhaust, ideal) and pushes Earth along its orbit.',
      fails: [`Fuel needed (rocket equation): ${fmtE(HYD.M_NEED, 2)} kg of hydrogen.`,
        `The solar wind carries only ${fmtE(HYD.MASS_FLUX, 1)} kg per m^2 per second: a 100 R_E scoop collects ${fmtE(hydrogen(100).m10k, 1)} kg in 10,000 years (${fmtE(hydrogen(100).frac, 1)} of the need).`,
        `Even the Sun's ENTIRE wind for 10,000 years = ${(100 * hydrogenWholeSun().frac).toFixed(0)} % of the need: Earth reaches ${hydrogenWholeSun().a10k.toFixed(2)} AU.`],
    },
  };
}
