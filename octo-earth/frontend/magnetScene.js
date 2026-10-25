// OCTO-EARTH failed attempt 3 - "Superconducting magnetic arms" (update p38). Independent model: its own arm design
// (8 straight cryostat booms), not the launcher arms; no asteroid.
//
// Idea: 8 superconducting booms around Earth carry a huge current I; in the Sun's (interplanetary) magnetic field B the
// Lorentz force F = I L x B pushes them - and Earth - along the orbit.
// Why it fails (all numbers from failedPhysics.js):
//  * The booms are INSIDE Earth's magnetosphere (they reach 3.25 R_E; the magnetopause is ~10 R_E sunward). The Sun's
//    field does not reach them. Earth's own dipole field does, but the force on a radial current in it is tangential
//    (r x z), the 8 forces cancel, and in any case Earth's field is made by Earth's core: pushing on it is pushing on
//    yourself (equal and opposite reaction on the core) -> NET FORCE 0.
//  * Hypothetical "IMF reaches the arms" (best case: every boom perpendicular to B, force counted along the orbit,
//    circuits closed through the solar-wind plasma): F = I * 8 L * B_IMF(r), B_IMF = 5 nT at 1 AU falling ~1/r.
//    A superconductor quenches when its own field exceeds ~45 T: for a 1 m cable I_max = 2 pi a B_c / mu0 = 2.25e8 A
//    -> F = 1.1e8 N, Earth moves 61 m in 10,000 years. The current needed is 4.4e17 A per arm (a 300 R_E thick
//    cable), and because B falls with distance even that only reaches 1.81 AU in 10,000 years; ~8.6e17 A would be
//    needed to arrive in time.
// The orbit curve is integrated (RK4) from da/dt = 2 a^1.5 F(a) / (M sqrt(GM)). Units: Earth radii, Earth at the origin,
// Sun towards -x, orbital motion +y, Earth's axis +z. Time-lapse: 30 s of animation = 10,000 years.
import * as THREE from './vendor/three.module.js';
import { makeBase, createArrow, shueSurface, V3, clamp, lerp, hash, drawChart } from './failKit.js';
import { glowTex } from './collector.js';
import { hudRow, hudBar, fmtE } from './cineKit.js';
import { K, MAG, F_REQ, imf, magnet, magForce, spiral } from './failedPhysics.js';

export const MZ = { T_ANIM: 30, R_IN: 1.25, R_OUT: 3.25, BOOM_R: 0.075, MP_R0: 10, MP_A: 0.58, N_PULSE: 7, PHASE: Math.PI / 8 };
const L10 = (x) => Math.log10(Math.max(x, 1e-30));
const IMF_SP = 3.2;                                      // spacing of the drawn IMF lines

export function createMagnetView(renderer, canvas, container) {
  const ctl = `<div class="fa-slider" title="Current in each boom (logarithmic slider)."><span class="dim">Current per arm</span> <input type="range" id="mzI" min="6" max="18" step="0.02" value="${L10(MAG.I_MAX).toFixed(2)}"> <b class="mono" id="mzIv"></b></div>
    <div class="btns small"><button data-mz="imf" title="Pretend the Sun's field reaches the arms (it does not: they are inside the magnetosphere). Best-case geometry.">IMF reaches arms (hypothetical)</button><button data-mz="quench" title="Pretend the superconductor survives any current.">Ignore quench</button></div>
    <canvas id="mzChart" class="fa-chart" title="Orbit size a(t) over the 10,000-year mission. Dashed: with the current that gives the required push at 1 AU."></canvas>`;
  const B = makeBase(renderer, canvas, container, {
    id: 'mz', title: '3 &middot; SUPERCONDUCTING MAGNETIC ARMS <span class="dim">(Lorentz force)</span>', controlsHtml: ctl,
    cam: { pos: [7.5, -11, 6.5], target: [0, 0, 0] }, maxDist: 120, minDist: 2,
    labels: { boom: '#bfe0ff', net: '#a8f0a0', mp: '#7fb0ff', imf: '#ffd27f', dip: '#7fc8ff', earth: '#9fd0ff', quench: '#ff8a6a' },
    note: 'Best case in the hypothetical mode: all boom currents perpendicular to the field, the force counted fully along the orbit, circuits closed through the solar-wind plasma. Parker spiral field: B_r = 3.5 nT/r^2, B_phi = B_r * Omega r / v_sw. Quench limit: field at the surface of a 1 m cable = 45 T. Field lines drawn schematically; booms and Earth to scale.',
  });
  const { scene, earth } = B;
  const S = { t: 0, running: false, started: false, ended: false, I: MAG.I_MAX, imf: false, ignoreQ: false, wall: 0, lastDyn: -1, idle: 0 };

  // ---------------------------------------------------------------- magnetosphere, dipole lines, IMF lines
  const mp = shueSurface(MZ.MP_R0, MZ.MP_A, 0x4f86ff, 0.14); scene.add(mp);
  const shueR = (x, y, z) => { const r = Math.hypot(x, y, z), c = -x / Math.max(r, 1e-6); return MZ.MP_R0 * Math.pow(2 / (1 + c), MZ.MP_A); };
  const dip = []; const dipMat = new THREE.LineBasicMaterial({ color: 0x5fb0ff, transparent: true, opacity: 0.22 });
  for (const Ls of [1.7, 2.3, 3.2, 4.4, 6, 8]) for (let k = 0; k < 8; k++) {
    const lon = k * Math.PI / 4 + 0.2, lm = Math.acos(Math.sqrt(1 / Ls)), pts = [];
    for (let i = 0; i <= 64; i++) {
      const la = -lm + 2 * lm * i / 64, r = Ls * Math.cos(la) ** 2, p = V3(r * Math.cos(la) * Math.cos(lon), r * Math.cos(la) * Math.sin(lon), r * Math.sin(la));
      if (r < shueR(p.x, p.y, p.z) * 0.97) pts.push(p); else if (pts.length) break;
    }
    if (pts.length > 2) { const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), dipMat); scene.add(l); dip.push(l); }
  }
  const imfUni = { r0: { value: MZ.MP_R0 }, al: { value: MZ.MP_A }, clip: { value: 1 }, op: { value: 0.55 } };
  const imfMat = new THREE.ShaderMaterial({
    uniforms: imfUni, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: 'varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }',
    fragmentShader: `uniform float r0; uniform float al; uniform float clip; uniform float op; varying vec3 vW;
      void main(){ float r = length(vW); float c = -vW.x / max(r, 1e-4); float rm = r0 * pow(2.0 / (1.0 + c), al);
        if (clip > 0.5 && r < rm * 1.04) discard;
        float a = op * smoothstep(46.0, 22.0, r); gl_FragColor = vec4(vec3(1.0, 0.8, 0.45) * a, a); }`,
  });
  const imfGroup = new THREE.Group(); scene.add(imfGroup);
  {
    const seg = [], d = V3(1, -1, 0).normalize(), n = V3(1, 1, 0).normalize();     // Parker spiral at 1 AU: ~45 deg
    for (const z of [-7.5, -3.75, 0, 3.75, 7.5]) for (let j = -16; j <= 16; j++) {
      const c = V3().copy(n).multiplyScalar(j * IMF_SP).add(V3(0, 0, z));
      for (let i = -24; i < 24; i++) { seg.push(V3().copy(c).addScaledVector(d, i * 2.5), V3().copy(c).addScaledVector(d, (i + 1) * 2.5)); }
    }
    imfGroup.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(seg), imfMat));
  }
  // ---------------------------------------------------------------- 8 straight cryostat booms + hub
  const rig = new THREE.Group(); scene.add(rig);
  const hub = new THREE.Mesh(new THREE.TorusGeometry(MZ.R_IN, 0.05, 12, 120), new THREE.MeshStandardMaterial({ color: 0xa8b0bc, metalness: 0.8, roughness: 0.3 })); rig.add(hub);
  const L = MZ.R_OUT - MZ.R_IN, booms = [], dirs = [];
  const coreMat = new THREE.MeshBasicMaterial({ color: 0x3fa0ff });
  const jacketMat = new THREE.MeshPhysicalMaterial({ color: 0xdfe8f2, metalness: 0.2, roughness: 0.15, transparent: true, opacity: 0.32, depthWrite: false });
  const bandMat = new THREE.MeshStandardMaterial({ color: 0xb8c2ce, metalness: 0.85, roughness: 0.3, emissive: 0x000000 });
  const tipMat = new THREE.MeshStandardMaterial({ color: 0x8090a0, metalness: 0.7, roughness: 0.35 });
  const bandGeo = new THREE.TorusGeometry(MZ.BOOM_R * 1.05, 0.012, 6, 24);
  for (let k = 0; k < 8; k++) {
    const ph = MZ.PHASE + k * Math.PI / 4, dir = V3(Math.cos(ph), Math.sin(ph), 0); dirs.push(dir);
    const g = new THREE.Group(); g.quaternion.setFromUnitVectors(V3(0, 1, 0), dir); rig.add(g);
    const core = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, L, 12), coreMat); core.position.y = MZ.R_IN + L / 2; g.add(core);
    const jacket = new THREE.Mesh(new THREE.CylinderGeometry(MZ.BOOM_R, MZ.BOOM_R, L, 20, 1, true), jacketMat); jacket.position.y = MZ.R_IN + L / 2; g.add(jacket);
    for (let b = 0; b <= 8; b++) { const band = new THREE.Mesh(bandGeo, bandMat); band.rotation.x = Math.PI / 2; band.position.y = MZ.R_IN + L * b / 8; g.add(band); }
    const tip = new THREE.Mesh(new THREE.SphereGeometry(0.11, 16, 10), tipMat); tip.position.y = MZ.R_OUT; g.add(tip);
    booms.push(g);
  }
  const pulseMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.045, 10, 6), new THREE.MeshBasicMaterial({ color: 0xcfeaff }), 8 * MZ.N_PULSE); rig.add(pulseMesh);
  const puffs = []; const puffTex = glowTex('rgba(240,245,255,0.9)', 'rgba(240,245,255,0)');
  for (let k = 0; k < 8; k++) { const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: puffTex, transparent: true, depthWrite: false, opacity: 0 })); scene.add(sp); puffs.push(sp); }
  const arrows = []; for (let k = 0; k < 8; k++) { const a = createArrow(0xa8f0a0, 0.03); scene.add(a.group); arrows.push(a); }
  const netArrow = createArrow(0xffffff, 0.06); scene.add(netArrow.group);
  const netRing = new THREE.Mesh(new THREE.RingGeometry(0.16, 0.2, 32), new THREE.MeshBasicMaterial({ color: 0xa8f0a0, side: THREE.DoubleSide, transparent: true, opacity: 0.9, depthTest: false }));
  netRing.renderOrder = 5; scene.add(netRing);

  // ---------------------------------------------------------------- physics state (cached per setting)
  let cache = { key: '' };
  function model() {
    const quench = S.I > MAG.I_MAX && !S.ignoreQ, Ieff = quench ? 0 : S.I, key = `${S.I}|${S.imf}|${S.ignoreQ}`;
    if (cache.key !== key) {
      const best = magnet(Math.max(Ieff, 1));
      const run = S.imf && Ieff > 0 ? spiral((r) => magForce(Ieff, r), K.T_MISSION_Y, 1500) : { samples: [[0, 1], [K.T_MISSION_Y, 1]], aEnd: 1, tReach: null };
      cache = { key, quench, Ieff, run, best, Fnet: S.imf ? magForce(Ieff, 1) : 0, Fboom: Ieff * MAG.L_ARM * MAG.B_EQ / (MZ.R_IN + L / 2) ** 3, Bself: K.MU0 * S.I / (2 * Math.PI * MAG.A_CABLE) };
    }
    return cache;
  }
  const aAt = (run, yrs) => { const sm = run.samples, n = sm.length - 1, f = clamp(yrs / K.T_MISSION_Y, 0, 1) * n, i = Math.min(n - 1, Math.floor(f)); return lerp(sm[i][1], sm[i + 1][1], f - i); };
  function state() { const m = model(), yrs = (S.started ? clamp(S.t / MZ.T_ANIM, 0, 1) : 0) * K.T_MISSION_Y; return { m, yrs, a: aAt(m.run, yrs) }; }
  const refRun = spiral((r) => magForce(MAG.I_REQ, r), K.T_MISSION_Y, 1500);

  function draw(dt) {
    const s = state(), m = s.m, tv = S.t + S.idle, on = S.started && m.Ieff > 0;
    // magnetosphere / IMF appearance
    imfUni.clip.value = S.imf ? 0 : 1; mp.material.uniforms.op.value = S.imf ? 0.04 : 0.14;
    imfGroup.position.set(((tv * 1.6) % (IMF_SP * Math.SQRT2)), 0, 0);
    // booms: blue core when superconducting, red when quenched
    const qVis = S.started && m.quench;
    coreMat.color.setHex(qVis ? 0xff3a1a : on ? 0x47b4ff : 0x2a4a70);
    bandMat.emissive.setHex(qVis ? 0x501000 : 0x000000);
    const sp = on ? 0.15 + 0.12 * clamp(L10(m.Ieff) - 6, 0, 12) : 0;
    const _m = new THREE.Matrix4();
    for (let k = 0; k < 8; k++) for (let j = 0; j < MZ.N_PULSE; j++) {
      const u = ((tv * sp + j / MZ.N_PULSE) % 1 + 1) % 1, r = MZ.R_IN + u * L, d = dirs[k];
      _m.makeTranslation(d.x * r, d.y * r, 0); if (!on) _m.scale(V3(0.001, 0.001, 0.001)); pulseMesh.setMatrixAt(k * MZ.N_PULSE + j, _m);
    }
    pulseMesh.instanceMatrix.needsUpdate = true;
    for (let k = 0; k < 8; k++) {
      const pf = puffs[k], d = dirs[k];
      const age = ((tv * 0.7 + hash(k, 3)) % 1);
      pf.position.set(d.x * (MZ.R_IN + L * (0.3 + 0.5 * hash(k, 4))), d.y * (MZ.R_IN + L * (0.3 + 0.5 * hash(k, 4))), 0.1 + age * 0.6);
      pf.scale.setScalar(0.3 + age * 0.9); pf.material.opacity = qVis ? 0.55 * (1 - age) : 0;
    }
    // Lorentz arrows at the boom midpoints
    const mid = MZ.R_IN + L / 2;
    for (let k = 0; k < 8; k++) {
      const d = dirs[k], o = V3(d.x * mid, d.y * mid, 0.12);
      if (!on) { arrows[k].set(o, d, 0); continue; }
      if (!S.imf) arrows[k].set(o, V3(d.y, -d.x, 0), 0.55 + 0.04 * clamp(L10(m.Fboom), 0, 25));   // I r x B_dipole(+z): tangential
      else arrows[k].set(o, V3(0, 1, 0), clamp(0.15 + 0.06 * (L10(m.Fnet / 8) - 6), 0.15, 1.6));    // best case: along the orbit
    }
    const nl = on && S.imf ? clamp(0.3 + 0.12 * (L10(m.Fnet) - 6), 0.3, 3) : 0;
    netArrow.set(V3(0, 1.15, 0), V3(0, 1, 0), nl);
    netRing.visible = on && !S.imf; netRing.position.set(0, 0, 1.3); netRing.lookAt(B.camera.position);
    earth.update(S.running ? dt : 0);
    return s;
  }

  // ---------------------------------------------------------------- panel + chart
  const slider = B.panel.querySelector('#mzI'), iv = B.panel.querySelector('#mzIv'), chart = B.panel.querySelector('#mzChart');
  const bImf = B.panel.querySelector('[data-mz="imf"]'), bQ = B.panel.querySelector('[data-mz="quench"]');
  slider.addEventListener('input', () => { S.I = Math.pow(10, +slider.value); S.lastDyn = -1; dyn(); });
  bImf.addEventListener('click', () => { S.imf = !S.imf; bImf.classList.toggle('on', S.imf); S.lastDyn = -1; dyn(); });
  bQ.addEventListener('click', () => { S.ignoreQ = !S.ignoreQ; bQ.classList.toggle('on', S.ignoreQ); S.lastDyn = -1; dyn(); });
  function plot(s) {
    drawChart(chart, {
      x0: 0, x1: 11000, y0: 0.95, y1: 3.1, xTicks: [0, 2500, 5000, 7500, 10000], yTicks: [1, 1.5, 2, 2.5, 3], xLabel: 'years',
      hMarks: [{ y: 2.7, col: '#7fe08a', label: 'target 2.7 AU' }], vMarks: [{ x: 10000, col: '#ff8a6a', label: 'deadline' }],
      series: [{ data: refRun.samples, col: '#9fb6d6', dash: [4, 3], w: 1.4, label: `I_req ${fmtE(MAG.I_REQ, 1)} A: ${refRun.aEnd.toFixed(2)} AU` },
        { data: s.m.run.samples, col: '#ffd27f', w: 2.2, label: `this setting: ${s.m.run.aEnd.toFixed(s.m.run.aEnd < 1.001 ? 7 : 2)} AU` }],
      marker: S.started ? { x: s.yrs, y: s.a, col: '#ffffff' } : null,
    });
  }
  function dyn() {
    const s = state(), m = s.m;
    iv.textContent = `${fmtE(S.I, 2)} A`;
    const qTxt = m.quench ? 'QUENCHED - superconductor lost, current 0' : S.I > MAG.I_MAX ? 'above the limit (quench ignored)' : 'superconducting';
    if (!S.started) B.setStage('READY - press Start', '');
    else if (m.quench) B.setStage(`QUENCH: ${fmtE(S.I, 1)} A per arm exceeds ${fmtE(MAG.I_MAX, 1)} A - no current, no force`, 'bad');
    else if (!S.imf) B.setStage(`CURRENT ON - forces internal, net force 0 - Earth stays at 1 AU` + (S.running ? '' : S.ended ? ' (end)' : ' - PAUSED'), 'bad');
    else B.setStage(`HYPOTHETICAL BEST CASE - ${Math.round(s.yrs).toLocaleString('en-US')} y: ` + (s.a < 1.001 ? `Earth moved ${fmtE((s.a - 1) * K.AU, 1)} m` : `a = ${s.a.toFixed(3)} AU`) + (S.running ? '' : S.ended ? ' (end)' : ' - PAUSED'), s.m.run.tReach !== null ? 'warn' : 'bad');
    let x = '';
    x += hudRow('Simulated time', `${Math.round(s.yrs).toLocaleString('en-US')} of 10,000 years`, 'Time-lapse: 30 s of animation = 10,000 years (simulated, not wall-clock).');
    x += hudRow('Field at the booms', S.imf ? `IMF ${fmtE(imf(s.a) * 1e9, 2)} nT (hypothetical)` : `Earth's dipole ${fmtE(MAG.B_EQ / (MZ.R_IN + L / 2) ** 3 * 1e6, 2)} &micro;T`, 'Real case: the booms (1.25-3.25 R_E) are deep inside the magnetosphere (magnetopause ~10 R_E sunward); only Earth\u2019s own field acts on them.');
    x += hudRow('Sun\u2019s field (IMF)', `${fmtE(imf(1) * 1e9, 2)} nT at 1 AU &rarr; ${fmtE(imf(2.7) * 1e9, 2)} nT at 2.7 AU`, 'Parker spiral: B_r = 3.5 nT / r^2, B_phi = B_r Omega r / v_sw. Falls ~1/r, so the push weakens as Earth moves out.');
    x += hudRow('Cable self-field', `${fmtE(m.Bself, 2)} T (limit ${MAG.B_CRIT} T)`, 'B = mu0 I / (2 pi a) at the surface of a 1 m radius cable. Above ~45 T even the best superconductors stop superconducting (quench).', m.Bself > MAG.B_CRIT ? 'bad' : 'good');
    x += hudRow('Booms', qTxt, `Quench limit for a 1 m cable: I_max = 2 pi a B_c / mu0 = ${fmtE(MAG.I_MAX, 2)} A.`, m.quench ? 'bad' : '');
    if (!S.imf) {
      x += hudRow('Force on each boom', `${fmtE(m.Fboom, 2)} N, tangential`, 'Radial current x northward dipole field = sideways (tangential) force; the 8 forces cancel (only a twist).');
      x += hudRow('Net force on Earth + arms', '0 N', 'Earth\u2019s field is produced by Earth\u2019s core: every push on the booms has an equal and opposite push on the core. Internal forces cannot move the centre of mass.', 'bad');
    } else {
      x += hudRow('Push (best case)', `${fmtE(m.Fnet, 2)} N = ${fmtE(100 * m.Fnet / F_REQ, 2)} % of F<sub>req</sub>`, 'F = I * 8 * 2 R_E * B_IMF at 1 AU, all along the orbit.', m.Fnet < F_REQ ? 'bad' : 'warn');
      x += hudRow('Arrives at 2.7 AU', m.run.tReach !== null ? `after ${Math.round(m.run.tReach).toLocaleString('en-US')} years` : 'not within 10,000 years', 'RK4 integration of da/dt = 2 a^1.5 F(a) / (M sqrt(GM)) with B falling with distance.', m.run.tReach !== null ? 'warn' : 'bad');
    }
    if (S.imf) x += hudRow('Earth moved outward', `${fmtE((s.a - 1) * K.AU, 2)} m`, 'Change of the orbit radius so far (1 AU = 1.496e11 m).', s.a < K.A1 ? 'bad' : 'warn');
    x += hudRow('Orbit now / at 10,000 y', `${s.a.toFixed(s.a < 1.001 ? 8 : 3)} / ${m.run.aEnd.toFixed(m.run.aEnd < 1.001 ? 8 : 3)} AU`, 'Target: 2.7 AU.', m.run.aEnd < K.A1 ? 'bad' : 'warn');
    x += `<div class="fa-sec">What it would take</div>`;
    x += hudRow('Current for F<sub>req</sub> at 1 AU', `${fmtE(MAG.I_REQ, 2)} A = ${fmtE(MAG.I_REQ / MAG.I_MAX, 1)} x I<sub>max</sub>`, `A cable carrying it below 45 T would need a radius of ${fmtE(MAG.A_FOR_IREQ / K.R_E, 0)} Earth radii.`, 'bad');
    x += hudRow('Even then', `a = ${refRun.aEnd.toFixed(2)} AU after 10,000 y`, 'Because the Sun\u2019s field weakens with distance.', 'bad');
    x += hudRow('Current to arrive in time', `${fmtE(MAG.I_ARRIVE, 2)} A`, 'Best case, quench ignored, IMF assumed to reach the arms.', 'bad');
    x += `<div class="fa-verdict">FAILS: inside the magnetosphere the net force is 0; even outside, the best real cable gives ${fmtE(magForce(MAG.I_MAX, 1), 1)} N (Earth moves ${fmtE((magnet(MAG.I_MAX).aEnd - 1) * K.AU, 0)} m in 10,000 years).</div>`;
    B.setDyn(x); plot(s);
  }
  function placeLabels() {
    const s = state();
    B.lbl.set('boom', V3(dirs[1].x * 3.5, dirs[1].y * 3.5, 0.35), 'superconducting boom (cryostat, 2 R_E)');
    B.lbl.set('net', S.started && s.m.Ieff > 0 ? (S.imf ? V3(0, 1.6 + 0.3, 0.6) : V3(0, 0, 1.65)) : null, S.imf ? `net push ${fmtE(s.m.Fnet, 1)} N (best case)` : 'net force = 0 (forces internal)');
    B.lbl.set('quench', S.started && s.m.quench ? V3(dirs[5].x * 3.6, dirs[5].y * 3.6, 0.5) : null, 'QUENCH');
    B.lbl.set('mp', V3(-9.6, 0, 2.6), S.imf ? 'magnetopause (ignored: hypothetical)' : 'magnetopause - Sun\u2019s field stops here');
    B.lbl.set('imf', V3(-14, -10, 7.5), 'Sun\u2019s magnetic field (Parker spiral), flowing with the solar wind');
    B.lbl.set('dip', V3(0, -5.2, 2.2), 'Earth\u2019s dipole field');
    B.lbl.set('earth', V3(0, 0, -1.4), 'Earth');
    B.lbl.declutter();
  }
  function reset() { Object.assign(S, { t: 0, running: false, started: false, ended: false, lastDyn: -1 }); draw(0); dyn(); }
  function start() { reset(); S.started = true; S.running = true; }
  function togglePause() { if (!S.started || S.ended) { start(); return; } S.running = !S.running; S.lastDyn = -1; }
  function update(dt) {
    dt = Math.min(0.1, Math.max(0, dt));
    if (S.running) { S.t += dt; if (S.t >= MZ.T_ANIM) { S.t = MZ.T_ANIM; S.running = false; S.ended = true; S.lastDyn = -1; } }
    else if (!S.started) S.idle += dt;
    draw(dt); B.updateCam(dt);
    S.wall += dt; if (S.lastDyn < 0 || S.wall - S.lastDyn > 0.25) { S.lastDyn = S.wall; dyn(); }
  }
  function render() { B.render(); placeLabels(); }
  reset();
  return {
    key: 'magnet', S, B, update, render, start, reset, togglePause, setActive: (on) => { B.setActive(on); if (on) { S.lastDyn = -1; } }, panel: B.panel, state,
    info: {
      title: 'Superconducting magnetic arms (Lorentz force)',
      idea: '8 superconducting booms around Earth carry an enormous current; in the Sun\u2019s magnetic field the Lorentz force F = I L x B pushes them, and Earth with them, along the orbit.',
      fails: ['The booms sit inside Earth\u2019s magnetosphere: the Sun\u2019s field never reaches them. Earth\u2019s own field gives only internal forces - net force 0.',
        `Even if the Sun\u2019s field reached them: the cable quenches above ${fmtE(MAG.I_MAX, 2)} A, giving ${fmtE(magForce(MAG.I_MAX, 1), 1)} N - Earth moves ${fmtE((magnet(MAG.I_MAX).aEnd - 1) * K.AU, 0)} m in 10,000 years.`,
        `The push needed takes ${fmtE(MAG.I_REQ, 1)} A per arm, and since the field weakens with distance, even that only reaches ${refRun.aEnd.toFixed(2)} AU by the deadline.`],
    },
  };
}
