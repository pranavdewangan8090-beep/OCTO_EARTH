// OCTO-EARTH failed attempt 1 - "Laser push" (update p38). Independent model; not the asteroid launcher.
//
// Idea: a huge laser array behind Earth (trailing side) shines on Earth; light carries momentum, so absorbed light
// pushes with F = P / c. To raise Earth's orbit from 1 AU to 2.7 AU in 10,000 years the push must average
// F_req = 2.2e17 N, so P = F_req * c = 6.6e25 W - 17 % of the Sun's total output and 5.4e8 times the sunlight Earth
// absorbs today. All that power ends up as heat in Earth.
//
// Mode "required power": a LOGARITHMIC simulated clock (0.1 s ... 46 days in 45 s of animation). The delivered energy
// E = P t is compared with the energy each stage of destruction needs (failedPhysics.js LASER.E_*): atmosphere blown
// off, oceans boiled, crust melted, mantle melted, gravitational binding energy (Earth dispersed). Earth's motion is
// integrated from a = F / M (drawn at true scale: it has moved ~260 km when the energy to unbind it has arrived).
// Mode "safe +2 K": the largest beam that warms Earth by only 2 K (3.4e15 W) on a LINEAR 10,000-year time-lapse.
// The power slider only changes the calculator table. Every visual is a deterministic function of the animation time.
import * as THREE from './vendor/three.module.js';
import { makeBase, createBeam, createArrow, V3, clamp, lerp, hash, fmtDur } from './failKit.js';
import { createSun } from './earthAssets.js';
import { hudRow, hudBar, fmtE, sm } from './cineKit.js';
import { K, LASER, F_REQ, DV_REQ, laser, laserMotion, aAfterDv, SUN_ABSORBED } from './failedPhysics.js';

export const LZ = { T_ANIM: 45, T_SAFE: 30, T0: 0.1, STATION_Y: -12, BEAM_R: 1.0, N_PLUME: 700 };
const REQ = laser(LASER.P_REQ), SAFE = laser(LASER.P_SAFE);
LZ.T_END = REQ.tUnbind * 1.06;
const STAGES = [
  { t: 0, name: 'BEAM ON - air and surface heating', cls: 'warn' },
  { t: REQ.tAtm, name: 'ATMOSPHERE BLOWN OFF - oceans boiling', cls: 'warn' },
  { t: REQ.tOceans, name: 'OCEANS BOILED AWAY - crust melting', cls: 'bad' },
  { t: REQ.tCrust, name: 'CRUST MOLTEN - mantle melting', cls: 'bad' },
  { t: REQ.tMantle, name: 'WHOLE PLANET MOLTEN - rock vaporising', cls: 'bad' },
  { t: REQ.tUnbind, name: 'ENERGY TO UNBIND EARTH DELIVERED - destroyed before it moved', cls: 'bad' },
];
/** Simulated time [s] at animation time ta in the required-power mode (logarithmic clock). */
export const laserClock = (ta) => LZ.T0 * Math.pow(LZ.T_END / LZ.T0, clamp(ta / LZ.T_ANIM, 0, 1));
const L10 = (x) => Math.log10(Math.max(x, 1e-30));
const ramp = (t, t0, t1) => clamp((L10(t) - L10(t0)) / (L10(t1) - L10(t0)), 0, 1);  // 0..1 between two times, log scale

export function createLaserView(renderer, canvas, container) {
  const ctl = `<div class="btns small"><span class="dim">Beam power:</span><button data-lz="req" class="on" title="Power needed to reach 2.7 AU in 10,000 years: P = F_req c.">Required (${fmtE(LASER.P_REQ, 1)} W)</button><button data-lz="safe" title="Largest beam that warms Earth by only 2 K.">Safe +2 K (${fmtE(LASER.P_SAFE, 1)} W)</button></div>
    <div class="fa-slider" title="Calculator only: try any beam power; the table below updates (the animation keeps its mode)."><span class="dim">Calculator power</span> <input type="range" id="lzP" min="12" max="27" step="0.05" value="${L10(LASER.P_REQ).toFixed(2)}"> <b class="mono" id="lzPv"></b></div>`;
  const B = makeBase(renderer, canvas, container, {
    id: 'lz', title: '1 &middot; LASER PUSH <span class="dim">(radiation pressure)</span>', controlsHtml: ctl,
    cam: { pos: [-21, -3.5, 7.5], target: [0, -5.4, 0] }, maxDist: 160,
    labels: { st: '#9fe8ff', beam: '#bff3ff', earth: '#9fd0ff', plume: '#ffc89a', push: '#a8f0a0', sun: '#ffcc33' },
    note: `Energy milestones assume all beam energy stays in Earth (re-radiation would slow the later stages, but the surface would already be at ~${fmtE(REQ.Tsurf, 0)} K, far above the boiling point of rock, ~3,000 K). Laser drawn visible (a real one would be infrared). Station size and distance not to scale; Earth's motion is to scale.`,
  });
  const { scene, earth } = B;
  const S = { t: 0, running: false, started: false, ended: false, mode: 'req', calcP: LASER.P_REQ, wall: 0, lastDyn: -1 };

  const sun = createSun(9); sun.group.position.set(-600, 60, 40); scene.add(sun.group);
  // ---------------------------------------------------------------- laser array station (faces +y, towards Earth)
  const st = new THREE.Group(); st.position.set(0, LZ.STATION_Y, 0); scene.add(st);
  const metal = new THREE.MeshStandardMaterial({ color: 0x9aa3b0, metalness: 0.7, roughness: 0.35 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x252b36, metalness: 0.4, roughness: 0.6 });
  const hexP = []; const SP = 0.36;
  for (let q = -3; q <= 3; q++) for (let r = -3; r <= 3; r++) { const x = SP * (q + r / 2), z = SP * r * 0.866; if (Math.hypot(x, z) <= 1.15) hexP.push([x, z]); }
  const emit = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.15, 0.17, 0.32, 6), metal, hexP.length);
  const lens = new THREE.InstancedMesh(new THREE.CircleGeometry(0.12, 18), new THREE.MeshBasicMaterial({ color: 0xc9f6ff }), hexP.length);
  const m4 = new THREE.Matrix4(), q4 = new THREE.Quaternion(), qL = new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0));
  hexP.forEach(([x, z], i) => {
    emit.setMatrixAt(i, m4.compose(V3(x, 0, z), q4.identity(), V3(1, 1, 1)));
    lens.setMatrixAt(i, m4.compose(V3(x, 0.165, z), qL, V3(1, 1, 1)));
  });
  st.add(emit); st.add(lens);
  const back = new THREE.Mesh(new THREE.CylinderGeometry(1.3, 1.3, 0.12, 6), dark); back.position.y = -0.2; st.add(back);
  for (const s of [-1, 1]) {                                   // radiator wings (the waste heat of the lasers)
    const w = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.03, 1.1), new THREE.MeshStandardMaterial({ color: 0x3a4152, metalness: 0.3, roughness: 0.7, emissive: 0x401006, emissiveIntensity: 0 }));
    w.position.set(s * 3.1, -0.45, 0); st.add(w); st.userData['wing' + s] = w;
    const strut = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 1.5), metal); strut.rotation.z = Math.PI / 2; strut.position.set(s * 1.3, -0.45, 0); st.add(strut);
  }
  const lensGlow = new THREE.PointLight(0x9fe8ff, 0, 12, 1.5); lensGlow.position.set(0, 1.2, 0); st.add(lensGlow);

  // ---------------------------------------------------------------- beam, push arrow, plume, halo
  const beam = createBeam(0x8fe6ff); scene.add(beam.group);
  const push = createArrow(0xa8f0a0, 0.05); scene.add(push.group);
  const N = LZ.N_PLUME, ppos = new Float32Array(N * 3), pcol = new Float32Array(N * 3), psz = new Float32Array(N);
  const pg = new THREE.BufferGeometry(); pg.setAttribute('position', new THREE.BufferAttribute(ppos, 3)); pg.setAttribute('color', new THREE.BufferAttribute(pcol, 3)); pg.setAttribute('sz', new THREE.BufferAttribute(psz, 1));
  const pmat = new THREE.ShaderMaterial({
    uniforms: { op: { value: 0 }, px: { value: renderer.getPixelRatio() } }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, vertexColors: true,
    vertexShader: 'attribute float sz; uniform float px; varying vec3 vC; void main(){ vC = color; vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_PointSize = sz * px * 60.0 / max(1.0, -mv.z); gl_Position = projectionMatrix * mv; }',
    fragmentShader: 'uniform float op; varying vec3 vC; void main(){ vec2 d = gl_PointCoord - 0.5; float a = smoothstep(0.5, 0.0, length(d)) * op; gl_FragColor = vec4(vC * a, a); }',
  });
  const plume = new THREE.Points(pg, pmat); plume.frustumCulled = false; scene.add(plume);
  const halo = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 32), new THREE.ShaderMaterial({
    uniforms: { op: { value: 0 } }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: 'varying vec3 vN; varying vec3 vW; void main(){ vN = normalize(mat3(modelMatrix) * normal); vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }',
    fragmentShader: 'uniform float op; varying vec3 vN; varying vec3 vW; void main(){ vec3 V = normalize(cameraPosition - vW); float f = 1.0 - abs(dot(normalize(vN), V)); float a = op * (0.25 + 1.5 * pow(f, 1.8)); gl_FragColor = vec4(vec3(1.0, 0.72, 0.4) * a, a); }',
  })); scene.add(halo);
  const HEAT = V3(0, -1, 0);                                    // the beam heats the trailing hemisphere (facing -y)
  earth.uniforms.heatDir.value.copy(HEAT);

  // ---------------------------------------------------------------- state -> visuals (pure function of S.t and mode)
  function state() {
    if (S.mode === 'safe') {
      const yrs = clamp(S.t / LZ.T_SAFE, 0, 1) * K.T_MISSION_Y, ts = yrs * K.YEAR;
      const m = laserMotion(LASER.P_SAFE, ts), a = aAfterDv(m.dv);
      return { ts, yrs, P: LASER.P_SAFE, E: LASER.P_SAFE * ts, dv: m.dv, x: (a - 1) * K.AU, a, stage: -1 };
    }
    const ts = S.started ? laserClock(S.t) : 0, m = laserMotion(LASER.P_REQ, ts);
    let stage = 0; for (let i = 0; i < STAGES.length; i++) if (ts >= STAGES[i].t) stage = i;
    return { ts, P: LASER.P_REQ, E: LASER.P_REQ * ts, dv: m.dv, x: m.x, stage };
  }
  function draw(dt) {
    const s = state(), U = earth.uniforms, on = S.started, req = S.mode === 'req', ts = s.ts;
    const tVis = S.t;                                            // animation time drives flicker / particle motion
    earth.group.position.set(0, req ? s.x / K.R_E : 0, 0);       // true scale (the safe-mode shift is ~6 m: invisible)
    const ep = earth.group.position;
    // heating (required mode only)
    const kA = on && req ? ramp(ts, 0.1, REQ.tAtm) : 0, kO = on && req ? ramp(ts, REQ.tAtm, REQ.tOceans) : 0;
    const kC = on && req ? ramp(ts, REQ.tOceans, REQ.tCrust) : 0, kM = on && req ? ramp(ts, REQ.tCrust, REQ.tMantle) : 0, kU = on && req ? ramp(ts, REQ.tMantle, REQ.tUnbind) : 0;
    U.heatSpot.value = on && req ? clamp(0.2 + 0.3 * kA + 0.2 * kO + 0.15 * kC + 0.15 * kM, 0, 1) : 0;
    U.spotCos.value = lerp(lerp(0.75, 0.05, kA), -0.6, kC);
    U.scorch.value = on && req ? sm(kO) : 0;
    U.heatGlobal.value = on && req ? clamp(0.3 * kC + 0.35 * kM + 0.35 * kU, 0, 1) : 0;
    earth.cloudUniforms.fade.value = 1 - sm(kO * 1.4);
    earth.atmUniforms.hot.value = on && req ? clamp(0.6 * kA + 0.4 * kM, 0, 1) * (1 - 0.5 * kO + 0.5 * kM) : 0;
    // beam
    const bi = !on ? 0 : req ? 1.0 * (1 - 0.25 * kU) : 0.22;
    beam.setEnds(V3(0, LZ.STATION_Y + 0.25, 0), V3(0, ep.y - 0.35, 0), req ? LZ.BEAM_R : 0.55);
    beam.set(bi * (0.85 + 0.15 * Math.sin(tVis * 23)), tVis);
    lens.material.color.setHex(on ? 0xffffff : 0x5d7f8a); lensGlow.intensity = on ? (req ? 40 : 6) : 0;
    for (const sgn of [-1, 1]) st.userData['wing' + sgn].material.emissiveIntensity = on ? (req ? 0.8 : 0.15) : 0;
    push.set(V3(0, ep.y + 1.15, 0), V3(0, 1, 0), on ? (req ? 1.6 : 0.7) : 0);
    // plume: steam while the oceans boil, then rock vapour; particles leave the heated hemisphere
    const pStr = on && req ? clamp(0.5 * kA + 0.9 * (kO > 0 && kO < 1 ? 1 : kO >= 1 ? 0.6 : 0) + 0.8 * kM, 0, 1.2) : 0;
    pmat.uniforms.op.value = pStr * 0.55;
    const cosMax = lerp(0.85, 0.2, kO), rock = clamp(kC * 1.5 + kM, 0, 1);
    for (let i = 0; i < N; i++) {
      const life = 2.2 + 2.0 * hash(i, 1), age = ((tVis / life + hash(i, 2)) % 1 + 1) % 1;
      const cz = lerp(1, cosMax, hash(i, 3)), ph = 6.2832 * hash(i, 4), rr = Math.sqrt(1 - cz * cz);
      const dx = rr * Math.cos(ph), dz = rr * Math.sin(ph), dy = -cz;            // direction around -y
      const R = 1.03 + age * (1.2 + 2.2 * hash(i, 5)) * (0.6 + rock);
      ppos[3 * i] = dx * R; ppos[3 * i + 1] = ep.y + dy * R; ppos[3 * i + 2] = dz * R;
      const fade = (1 - age) * (age < 0.08 ? age / 0.08 : 1);
      const c = [lerp(0.75, 1.0, rock) * fade, lerp(0.85, 0.55, rock) * fade, lerp(1.0, 0.25, rock) * fade];
      pcol.set(c, 3 * i); psz[i] = 0.7 + 1.6 * age;
    }
    pg.attributes.position.needsUpdate = true; pg.attributes.color.needsUpdate = true; pg.attributes.sz.needsUpdate = true;
    halo.position.copy(ep); halo.scale.setScalar(1.06 + 0.9 * sm(kU)); halo.material.uniforms.op.value = on && req ? 0.25 * kM + 0.6 * kU : 0;
    earth.update(S.running ? dt : 0); sun.update(tVis);
    return s;
  }

  // ---------------------------------------------------------------- panel
  const slider = B.panel.querySelector('#lzP'), pv = B.panel.querySelector('#lzPv');
  slider.addEventListener('input', () => { S.calcP = Math.pow(10, +slider.value); S.lastDyn = -1; dyn(); });
  B.ctlEl.addEventListener('click', (e) => {
    const b = e.target.closest('[data-lz]'); if (!b) return;
    S.mode = b.dataset.lz; for (const x of B.ctlEl.querySelectorAll('[data-lz]')) x.classList.toggle('on', x === b);
    reset(); if (S.mode === 'safe') B.fly([-13, -3, 5], [0, -4.5, 0]); else B.fly([-21, -3.5, 7.5], [0, -5.4, 0]);
  });
  function dyn() {
    const s = state(), req = S.mode === 'req', c = laser(S.calcP);
    pv.textContent = `${fmtE(S.calcP, 2)} W`;
    if (!S.started) B.setStage('READY - press Start', '');
    else if (req) B.setStage(STAGES[s.stage].name + (S.running ? '' : S.ended ? ' (end)' : ' - PAUSED'), STAGES[s.stage].cls);
    else B.setStage(`SAFE BEAM - Earth fine, but moved ${fmtE(s.x, 1)} m in ${Math.round(s.yrs).toLocaleString('en-US')} years` + (S.running ? '' : S.ended ? ' (end)' : ' - PAUSED'), s.yrs >= K.T_MISSION_Y ? 'bad' : 'good');
    const milestones = [['atmosphere blown off', REQ.tAtm, LASER.E_ATMOSPHERE], ['oceans boiled', REQ.tOceans, LASER.E_OCEANS], ['crust melted', REQ.tCrust, LASER.E_CRUST], ['mantle melted', REQ.tMantle, LASER.E_MANTLE], ['Earth unbound (dispersed)', REQ.tUnbind, LASER.E_UNBIND]]
      .map(([n, t, E]) => hudRow(n, `${fmtE(E, 2)} J at ${fmtDur(t)} ${req && S.started && s.ts >= t ? '<b class="fa-bad">REACHED</b>' : '<span class="dim">pending</span>'}`, 'Energy this stage needs (simple estimate, see README) and the beam time that delivers it at the required power.', req && S.started && s.ts >= t ? 'bad' : '')).join('');
    let h = '';
    if (req) {
      h += hudRow('Simulated time', `${S.started ? fmtDur(s.ts) : '0 s'} <span class="dim">(log clock)</span>`, 'Simulated time since the beam came on - NOT wall-clock time. The clock is logarithmic: 45 s of animation cover 0.1 s to 46 days.');
      h += hudRow('Beam power', `${fmtE(s.P, 2)} W = ${(100 * s.P / K.L_SUN).toFixed(1)} % of the Sun`, 'P = F_req c: the power whose light pressure gives the average push needed for 2.7 AU in 10,000 years.');
      h += hudRow('vs. sunlight on Earth', `${fmtE(s.P / SUN_ABSORBED, 2)} x`, `Earth absorbs ${fmtE(SUN_ABSORBED, 2)} W of sunlight today. Every watt of the beam also becomes heat.`, 'bad');
      h += hudRow('Push on Earth', `${fmtE(REQ.F, 2)} N (= F<sub>req</sub>)`, 'F = P / c for fully absorbed light.');
      h += hudRow('Energy delivered', `${fmtE(s.E, 2)} J`, 'E = P t.');
      h += hudRow('Earth moved so far', `${fmtE(s.x / 1000, 2)} km &middot; &Delta;v ${fmtE(s.dv, 2)} m/s`, `From a = F/M = ${fmtE(REQ.acc, 2)} m/s^2. The full trip needs Δv = ${DV_REQ.toFixed(0)} m/s over 10,000 years.`);
      h += hudRow('Surface temperature', `~${fmtE(REQ.Tsurf, 0)} K`, 'Global energy balance (288 K today) with the beam added: T = 288 K x ((S + P)/S)^(1/4).', 'bad');
      h += `<div class="fa-sec">Energy budget milestones</div>${milestones}`;
    } else {
      h += hudRow('Simulated time', `${Math.round(s.yrs).toLocaleString('en-US')} of 10,000 years <span class="dim">(linear)</span>`, 'Linear time-lapse: 30 s of animation = 10,000 years.');
      h += hudRow('Beam power', `${fmtE(s.P, 2)} W`, 'Chosen so that Earth warms by only 2 K: P = S_abs ((290/288)^4 - 1).');
      h += hudRow('Warming', `+${LASER.T_SAFE_DK} K (safe)`, 'Global energy balance.', 'good');
      h += hudRow('Push on Earth', `${fmtE(SAFE.F, 2)} N = ${fmtE(100 * SAFE.F / F_REQ, 1)} % of F<sub>req</sub>`, 'F = P / c.', 'bad');
      h += hudRow('Earth moved outward', `${fmtE(s.x, 2)} m (orbit 1 &rarr; ${s.a.toFixed(9)} AU)`, 'Change of the orbit radius from the speed gained (slow spiral).');
      h += hudRow('Progress to 2.7 AU', `${fmtE(100 * (s.a - 1) / 1.7, 2)} % ${hudBar((s.a - 1) / 1.7, '#ff6a4d')}`, 'Fraction of the 1.7 AU that must be covered.', 'bad');
    }
    h += `<div class="fa-sec">Calculator (slider): ${fmtE(c.P, 2)} W</div>`;
    h += hudRow('Push / needed', `${fmtE(100 * c.fracOfNeed, 2)} %`, 'F = P/c compared with F_req = 2.2e17 N.', c.fracOfNeed < 1 ? 'bad' : '');
    h += hudRow('Orbit after 10,000 y', `${c.a10k.toFixed(c.a10k < 1.001 ? 7 : 3)} AU`, 'Slow spiral with the speed gained in 10,000 years.', c.a10k < K.A1 ? 'bad' : '');
    h += hudRow('Surface temperature', `~${fmtE(c.Tsurf, 0)} K`, 'Global energy balance with this beam absorbed (288 K today). Water boils at 373 K, rock melts at ~1,500 K.', c.Tsurf > 300 ? 'bad' : 'good');
    h += hudRow('Oceans boiled after', fmtDur(c.tOceans), 'Time for the beam to deliver the energy to heat and boil all oceans (ignores re-radiation).', c.tOceans < 1e4 * K.YEAR ? 'bad' : '');
    h += `<div class="fa-verdict">FAILS: enough push means ${fmtE(REQ.Tsurf, 0)} K and a vaporised planet; a beam Earth survives moves it ${fmtE((SAFE.a10k - 1) * K.AU, 1)} m in 10,000 years.</div>`;
    B.setDyn(h);
  }
  function placeLabels() {
    const ep = earth.group.position, req = S.mode === 'req';
    B.lbl.set('st', V3(0, LZ.STATION_Y, 1.6), 'laser array (beams at Earth’s trailing side)');
    B.lbl.set('beam', S.started ? V3(0, LZ.STATION_Y * 0.55, 1.15) : null, req ? `beam ${fmtE(LASER.P_REQ, 1)} W` : `safe beam ${fmtE(LASER.P_SAFE, 1)} W`);
    B.lbl.set('earth', V3(0, ep.y, -1.35), 'Earth');
    B.lbl.set('push', S.started ? V3(0, ep.y + 2.0, 0.5) : null, `push ${fmtE(req ? REQ.F : SAFE.F, 1)} N (orbit direction)`);
    B.lbl.set('plume', S.started && req && pmat.uniforms.op.value > 0.1 ? V3(0, ep.y - 2.6, 1.4) : null, earth.uniforms.heatGlobal.value > 0.2 ? 'rock vapour' : 'steam / blown-off air');
    B.lbl.set('sun', V3(-600, 60, 52), 'Sun (direction)');
    B.lbl.declutter();
  }
  // ---------------------------------------------------------------- controls
  function reset() { Object.assign(S, { t: 0, running: false, started: false, ended: false, lastDyn: -1 }); draw(0); dyn(); }
  function start() { reset(); S.started = true; S.running = true; }
  function togglePause() { if (!S.started || S.ended) { start(); return; } S.running = !S.running; S.lastDyn = -1; }
  function update(dt) {
    dt = Math.min(0.1, Math.max(0, dt));
    if (S.running) { S.t += dt; const T = S.mode === 'req' ? LZ.T_ANIM : LZ.T_SAFE; if (S.t >= T) { S.t = T; S.running = false; S.ended = true; S.lastDyn = -1; } }
    draw(dt); B.updateCam(dt);
    S.wall += dt; if (S.lastDyn < 0 || S.wall - S.lastDyn > 0.25) { S.lastDyn = S.wall; dyn(); }
  }
  function render() { B.render(); placeLabels(); }
  reset();
  return {
    key: 'laser', S, B, update, render, start, reset, togglePause, setActive: B.setActive, panel: B.panel, state,
    info: {
      title: 'Laser push (radiation pressure)',
      idea: 'A giant laser array behind Earth shines on its trailing side. Light carries momentum, so the absorbed beam pushes Earth forward along its orbit (F = P/c), slowly spiralling it outward.',
      fails: [`Needed push ${fmtE(F_REQ, 2)} N means a beam of ${fmtE(LASER.P_REQ, 2)} W (${(100 * LASER.P_REQ / K.L_SUN).toFixed(0)} % of the Sun, ${fmtE(LASER.P_REQ / SUN_ABSORBED, 1)} x the sunlight Earth absorbs).`,
        `All of it turns into heat: oceans boiled in ${fmtDur(REQ.tOceans)}, energy to unbind Earth in ${fmtDur(REQ.tUnbind)} - Earth has moved only ${fmtE(laserMotion(LASER.P_REQ, REQ.tUnbind).x / 1000, 0)} km by then.`,
        `A beam Earth survives (+2 K, ${fmtE(LASER.P_SAFE, 1)} W) moves the orbit ${fmtE((SAFE.a10k - 1) * K.AU, 1)} m in 10,000 years.`],
    },
  };
}
