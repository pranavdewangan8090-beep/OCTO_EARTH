// OCTO-EARTH overview details + climate / energy-ledger / belt panel.
// Everything shown here comes from the Python records:
//   * meta.belt   : the assigned orbits of the 12 simulated collectors (BeltKeeper.tracked)
//   * rec.belt    : their deviations from the assigned slots WITH station keeping and WITHOUT (faint grey ghosts),
//                   20 snapshots per decision interval as [radius - R (km), angle along the orbit (rad), height (km)]
//   * rec.clim_track / rec.T_surface_K / rec.mirror_W : the two-box climate model and the space mirrors
// It also draws the launcher ring + 8 arms on the overview Earth (same design as the Earth POV, scaled with the
// enlarged Earth marker) and the ring of space mirrors (drawn close to Earth; their true orbit is 20 Earth radii).
//   * rec.ledger  : the energy ledger of the interval
// The belt's other collectors (instanced satellites) move on circular orbits inside the configured band; they are a
// picture of the belt, the 12 tracked ones are the simulated ones.
import * as THREE from './vendor/three.module.js';
import { satelliteGeometry, satQuat, glowTex } from './collector.js';
import { earthTexture } from './pov.js';
import { RING_R, GRAB_Z, strokeLength } from './launcher.js';
import { createArmRig } from './octoArms.js';

const AU_KM = 1.495978707e8;
const N_INST = 900;                       // drawn belt satellites (illustration of the 5 million)
const MAX_REV_PER_S = 0.4;                // belt motion is frozen above this many orbits per wall-clock second
const V3 = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const UPY = V3(0, 1, 0), NORTH = V3(0, 0, 1);

function hash(i, k) { let h = Math.imul(i + 1, 374761393) ^ Math.imul(k + 7, 668265263); h = Math.imul(h ^ (h >>> 13), 1274126177); return ((h ^ (h >>> 16)) >>> 0) / 4294967296; }

export function createOvExtras(scene, S, earth, EARTH_R) {
  // ------------------------------------------------------------- Earth detail (procedural texture, tilt, spin, air glow)
  earth.material.visible = false;
  const tilt = new THREE.Group(); earth.add(tilt);
  tilt.quaternion.setFromUnitVectors(UPY, V3(0, -Math.sin(23.44 * Math.PI / 180), Math.cos(23.44 * Math.PI / 180)));
  const spin = new THREE.Group(); tilt.add(spin);
  const earthMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.85, metalness: 0, emissive: 0x0b1d3a, emissiveIntensity: 0.55 });
  const body = new THREE.Mesh(new THREE.SphereGeometry(EARTH_R, 64, 32), earthMat); spin.add(body);
  const air = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex('rgba(110,170,255,0.55)', 'rgba(60,120,255,0)'), depthWrite: false, blending: THREE.AdditiveBlending }));
  air.scale.setScalar(EARTH_R * 3.1); earth.add(air);
  let texDone = false;

  // ------------------------------------------------------------- belt satellites (instanced)
  const satGeo = satelliteGeometry(true);          // no colours in the solar-system view
  const satMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.1, emissive: 0x2a2a2a, emissiveIntensity: 0.5 });
  const sats = new THREE.InstancedMesh(satGeo, satMat, N_INST); sats.frustumCulled = false; sats.visible = false; scene.add(sats);
  const inst = [];
  // ------------------------------------------------------------- 12 simulated collectors, their orbits and grey ghosts
  // No colour coding: the simulated collectors are solid silver, the 'without station keeping' copies are faint
  // see-through grey ghosts joined to them by a thin dashed grey line.
  const trkMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.5, emissive: 0x3a3a3a, emissiveIntensity: 0.8 });
  const ghostMat = new THREE.MeshBasicMaterial({ color: 0x9a9a9a, transparent: true, opacity: 0.4, depthWrite: false });
  const trk = new THREE.InstancedMesh(satGeo, trkMat, 12), ghost = new THREE.InstancedMesh(satGeo, ghostMat, 12);
  trk.frustumCulled = ghost.frustumCulled = false; trk.visible = ghost.visible = false; scene.add(trk); scene.add(ghost);
  const haloTex = glowTex('rgba(235,235,235,0.45)', 'rgba(235,235,235,0)'), ghostTex = glowTex('rgba(190,190,190,0.2)', 'rgba(190,190,190,0)');
  const halos = [], ghalos = [];
  for (let k = 0; k < 12; k++) {
    const h = new THREE.Sprite(new THREE.SpriteMaterial({ map: haloTex, depthWrite: false, blending: THREE.AdditiveBlending })); h.scale.setScalar(0.32); h.visible = false; scene.add(h); halos.push(h);
    const g = new THREE.Sprite(new THREE.SpriteMaterial({ map: ghostTex, depthWrite: false, blending: THREE.AdditiveBlending })); g.scale.setScalar(0.3); g.visible = false; scene.add(g); ghalos.push(g);
  }
  const orbPos = new Float32Array(12 * 129 * 2 * 3), orbGeo = new THREE.BufferGeometry(); orbGeo.setAttribute('position', new THREE.BufferAttribute(orbPos, 3));
  const orbits = new THREE.LineSegments(orbGeo, new THREE.LineBasicMaterial({ color: 0xb0b0b0, transparent: true, opacity: 0.12, depthWrite: false }));
  orbits.frustumCulled = false; orbits.visible = false; scene.add(orbits);
  const linkPos = new Float32Array(12 * 6), linkGeo = new THREE.BufferGeometry(); linkGeo.setAttribute('position', new THREE.BufferAttribute(linkPos, 3));
  const links = new THREE.LineSegments(linkGeo, new THREE.LineDashedMaterial({ color: 0xa0a0a0, dashSize: 0.12, gapSize: 0.08, transparent: true, opacity: 0.45 }));
  links.frustumCulled = false; links.visible = false; scene.add(links);
  // ------------------------------------------------------------- the launcher ring and the 8 arms on Earth
  // Same design as in the Earth POV, built in units of Earth radii and scaled with the (enlarged) Earth marker, so the
  // arms stay in proportion to the Earth drawn here. They throw ONE packet together whenever packets are launched.
  const ARM = { RING: RING_R, NSEG: 10, N: 8, THROW_S: 1.3 };
  // the meshes, materials and arm shape come from the shared rig in octoArms.js (also used by the p37 cinematic views)
  const rig = createArmRig({ nseg: ARM.NSEG, look: 'overview' });
  const armGroup = rig.group; armGroup.scale.setScalar(EARTH_R); earth.add(armGroup);
  const _targets = [...Array(ARM.N)].map(() => V3()), _opens = new Array(ARM.N).fill(0.35);
  const thrownRock = new THREE.Mesh(new THREE.IcosahedronGeometry(0.24, 0),
    new THREE.MeshStandardMaterial({ color: 0xd09060, roughness: 0.9, emissive: 0x4a2410, emissiveIntensity: 0.8, transparent: true }));
  thrownRock.visible = false; armGroup.add(thrownRock);
  const armState = { phase: 1, dir: V3(1, 0, 0), G: V3(), F: V3(), u: V3(), w: V3(), arms: rig.arms };

  // ------------------------------------------------------------- the ring of space mirrors (enabler E3)
  // Drawn at 4.5 Earth radii from the Earth marker so it can be seen; the real ring is 20 Earth radii away, in the
  // plane through Earth perpendicular to the sunlight (it never passes through Earth's shadow).
  const N_MIR = 36, MIR_R = 4.5;
  const mirrorGroup = new THREE.Group(); mirrorGroup.scale.setScalar(EARTH_R); earth.add(mirrorGroup);
  const mirrorPanels = new THREE.InstancedMesh(new THREE.PlaneGeometry(0.32, 0.32),
    new THREE.MeshBasicMaterial({ color: 0xe8f2ff, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }), N_MIR);
  mirrorPanels.frustumCulled = false; mirrorGroup.add(mirrorPanels);
  const mirrorOrbit = new THREE.Mesh(new THREE.TorusGeometry(MIR_R, 0.025, 6, 160),
    new THREE.MeshBasicMaterial({ color: 0x9fd8ff, transparent: true, opacity: 0.35, depthWrite: false })); mirrorGroup.add(mirrorOrbit);
  // Light rays: for every mirror one incoming sunbeam (from the Sun's side) and its reflection. Drawn as thin,
  // very light lines; the brightness of each vertex is in the colour attribute.
  const rayPos = new Float32Array(N_MIR * 4 * 3), rayCol = new Float32Array(N_MIR * 4 * 3), rayGeo = new THREE.BufferGeometry();
  rayGeo.setAttribute('position', new THREE.BufferAttribute(rayPos, 3)); rayGeo.setAttribute('color', new THREE.BufferAttribute(rayCol, 3));
  const mirrorRays = new THREE.LineSegments(rayGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.4,
    depthWrite: false, blending: THREE.AdditiveBlending }));
  mirrorRays.frustumCulled = false; mirrorGroup.add(mirrorRays);
  mirrorGroup.visible = false;

  let key = '', trkKey = '', frozenT = null;
  const E = { P: [], Q: [], N: [], u0: [], inc: [], n: 0, R: 0.5 };
  const _q = new THREE.Quaternion(), _m = new THREE.Matrix4(), _p = V3(), _s = V3(), _d = V3(), _sun = V3();
  function basis(incR, nodeR) {
    const cO = Math.cos(nodeR), sO = Math.sin(nodeR), ci = Math.cos(incR), si = Math.sin(incR);
    const P = V3(cO, sO, 0), Q = V3(-sO * ci, cO * ci, si); return { P, Q, N: V3().crossVectors(P, Q) };
  }
  function place(mesh, k, pos, scale) {
    _sun.copy(pos).negate(); satQuat(_sun, NORTH, _q); _s.setScalar(scale); _m.compose(pos, _q, _s); mesh.setMatrixAt(k, _m);
  }
  function setupBelt(sol, belt) {
    const k2 = `${sol.orbit_au}|${sol.half_width_deg}|${sol.orbit_period_yr}`;
    if (k2 === key) return; key = k2;
    const hw = sol.half_width_deg * Math.PI / 180;
    inst.length = 0;
    for (let k = 0; k < N_INST; k++) {
      const inc = Math.asin(Math.sin(hw) * hash(k, 1)), node = 2 * Math.PI * hash(k, 2), u0 = 2 * Math.PI * hash(k, 3);
      inst.push({ ...basis(inc, node), u0, rj: 1 + 0.004 * (hash(k, 4) - 0.5) });
    }
    E.R = sol.orbit_au; E.n = 2 * Math.PI / (sol.orbit_period_yr || 0.354);
  }
  function setupTracked(belt) {
    if (!belt) return false;
    const k2 = JSON.stringify([belt.R_au, belt.inc_deg, belt.node_deg]);
    if (k2 === trkKey) return true; trkKey = k2;
    E.P = []; E.Q = []; E.N = []; E.u0 = belt.u0_rad; E.inc = belt.inc_deg; E.nT = belt.n_rad_yr; E.RT = belt.R_au;
    let o = 0;
    for (let k = 0; k < 12; k++) {
      const b = basis(belt.inc_deg[k] * Math.PI / 180, belt.node_deg[k] * Math.PI / 180); E.P.push(b.P); E.Q.push(b.Q); E.N.push(b.N);
      for (let j = 0; j < 128; j++) for (const jj of [j, j + 1]) {
        const a = jj / 128 * 2 * Math.PI; _p.copy(b.P).multiplyScalar(Math.cos(a)).addScaledVector(b.Q, Math.sin(a)).multiplyScalar(belt.R_au * S);
        orbPos.set([_p.x, _p.y, _p.z], o); o += 3;
      }
    }
    orbGeo.setDrawRange(0, o / 3); orbGeo.attributes.position.needsUpdate = true;
    return true;
  }
  // deviation of the tracked collectors at time t from the decision record that covers t
  function devAt(dec, t, which) {
    const b = dec && dec.belt; if (!b || !b.track || !b.track[which] || !b.t) return null;
    const ts = b.t, tr = b.track[which]; let k = 0;
    while (k < ts.length - 2 && ts[k + 1] < t) k++;
    const h = ts[k + 1] - ts[k], f = h > 0 ? Math.max(0, Math.min(1, (t - ts[k]) / h)) : 0;
    return tr[k].map((a, i) => a.map((x, c) => x + (tr[k + 1][i][c] - x) * f));
  }

  // ------------------------------------------------------------- arms: pose and the group throw (illustration)
  const _sc = new THREE.Vector3(), _u = new THREE.Vector3();          // (_q, _m are shared with the belt code above)
  const _e1 = V3(), _e2 = V3(), _T = V3(), _P1 = V3(), _P2 = V3(), _ab = V3(), _cp = V3(), _lift = V3(), _seg = V3(), _mid = V3();
  /** Put instance idx of an instanced mesh at pos, its long (y) axis along dir, scaled by (sx, sy, sz). */
  function placeInstance(mesh, idx, pos, dir, sx, sy, sz) {
    _q.setFromUnitVectors(UPY, _u.copy(dir).normalize()); _sc.set(sx, sy, sz); _m.compose(pos, _q, _sc); mesh.setMatrixAt(idx, _m);
  }
  /** Cubic Bezier point between p0 and p3 with control points p1 and p2. */
  function bezier(p0, p1, p2, p3, t, out) {
    const v = 1 - t;
    return out.set(0, 0, 0).addScaledVector(p0, v * v * v).addScaledVector(p1, 3 * v * v * t).addScaledVector(p2, 3 * v * t * t).addScaledVector(p3, t * t * t);
  }
  /**
   * Pose the 8 arms (each 2 Earth radii long). At rest they reach outward and sway a little. While packets are being
   * launched, all 8 close around one rock at the grab point under the asteroid (on the ring's axis, the same distance
   * from every arm root) and fling it along the recorded launch direction, then repeat.
   * Everything is in Earth radii inside armGroup (Earth's centre = origin, ecliptic north = +z).
   */
  function updateArms(st, dt, wall) {
    armGroup.visible = !st.hide && st.layers.arms !== false;
    if (!armGroup.visible) { thrownRock.visible = false; return; }
    _e1.copy(earth.position).normalize(); if (_e1.lengthSq() < 0.5) _e1.set(1, 0, 0);   // away from the Sun
    _e2.crossVectors(NORTH, _e1);
    rig.setFrame(_e1, _e2, NORTH, 0);           // ring plane + struts + root hubs
    const A = armState, sway = st.playing && st.speedY > 200 ? 0 : wall;
    // start a new throw while launches are happening (one throw stands for many packets at fast playback)
    if (A.phase >= 1 && st.launching && st.throwDir) {
      A.phase = 0; A.dir.set(...st.throwDir).normalize();
      // grab point on the ring axis under the asteroid (same distance from all 8 roots); release point as far along
      // the throw direction as all 8 arms (2 Earth radii long) can still hold the packet (update p36)
      A.G.copy(NORTH).multiplyScalar(GRAB_Z - 0.15);
      const roots = [...Array(ARM.N)].map((_, k) => { const a = k * Math.PI / 4; return V3().copy(_e1).multiplyScalar(Math.cos(a)).addScaledVector(_e2, Math.sin(a)).multiplyScalar(ARM.RING + 0.05); });
      A.stroke = strokeLength(A.G, A.dir, roots);
      A.F.copy(A.G).addScaledVector(A.dir, Math.max(0.05, A.stroke - 0.15));
      A.u.copy(Math.abs(A.dir.z) < 0.9 ? NORTH : _e1).cross(A.dir).normalize(); A.w.crossVectors(A.dir, A.u);
    }
    const throwing = A.phase < 1; let ph = 1;
    if (throwing) { A.phase = Math.min(1, A.phase + (st.playing ? dt : 0) / ARM.THROW_S); ph = A.phase; }
    const smooth = (k) => k * k * (3 - 2 * k);
    for (const arm of A.arms) {
      // rest pose: reaching outward, alternately above and below the ring, with a slow sway
      const _T = rig.restTarget(arm.k, sway, _targets[arm.k]);
      let open = 0.35;
      if (throwing) {
        const th = arm.k * 45 * Math.PI / 180, cage = _cp.copy(A.u).multiplyScalar(Math.cos(th)).addScaledVector(A.w, Math.sin(th)).multiplyScalar(0.28);
        const grab = _P1.copy(A.G).add(cage), fling = _P2.copy(A.F).addScaledVector(cage, 1.2), rest = _mid.copy(_T);
        if (ph < 0.36) { _T.lerpVectors(rest, grab, smooth(ph / 0.36)); open = 0.8; }
        else if (ph < 0.45) { _T.copy(grab); open = 0.8 - 0.75 * (ph - 0.36) / 0.09; }
        else if (ph < 0.66) { const k = (ph - 0.45) / 0.21; _T.lerpVectors(grab, fling, k * k); open = 0.05; }
        else { const k = (ph - 0.66) / 0.34; _T.lerpVectors(fling, rest, smooth(k)); open = 1 - 0.65 * k; }
      }
      _opens[arm.k] = open;
    }
    // every arm has the same length (2 Earth radii): it bends away from Earth when its claw is closer (armCurve)
    rig.pose(_targets, { open: _opens, t: wall });
    // the rock: held, flung, then flying away along the launch direction (fading out)
    thrownRock.visible = throwing && ph > 0.36;
    if (thrownRock.visible) {
      if (ph < 0.45) thrownRock.position.copy(A.G);
      else if (ph < 0.66) { const k = (ph - 0.45) / 0.21; thrownRock.position.lerpVectors(A.G, A.F, k * k); }
      else thrownRock.position.copy(A.F).addScaledVector(A.dir, 9 * (ph - 0.66));
      thrownRock.material.opacity = ph < 0.66 ? 1 : Math.max(0, 1 - (ph - 0.66) / 0.34);
      thrownRock.rotation.x += dt * 3; thrownRock.rotation.y += dt * 2;
    }
  }

  // ------------------------------------------------------------- mirror ring: panels, reflected light (illustration)
  /**
   * Draw the mirror ring in the plane through Earth perpendicular to the sunlight. The share of mirrors whose
   * reflected ray hits Earth = recorded mirror light / mirror capacity (the agent's action a4); the others turn their
   * light past Earth. Brightness of the rays follows the recorded mirror light.
   */
  function updateMirrors(st, wall) {
    const on = !st.hide && st.layers.mirrors !== false && st.mirrors && st.mirrors.on;
    mirrorGroup.visible = !!on; if (!on) return;
    _e1.copy(earth.position).normalize(); if (_e1.lengthSq() < 0.5) _e1.set(1, 0, 0);   // anti-Sun direction
    const sunward = V3().copy(_e1).negate();
    const b1 = V3().crossVectors(NORTH, _e1); if (b1.lengthSq() < 1e-6) b1.set(0, 1, 0); b1.normalize();
    const b2 = V3().crossVectors(_e1, b1);
    mirrorOrbit.quaternion.setFromUnitVectors(V3(0, 0, 1), _e1);
    const cap = Math.max(1e-9, st.mirrors.capRel || 1), aimed = Math.max(0, Math.min(1, (st.mirRel || 0) / cap));
    const nAimed = Math.round(aimed * N_MIR), spin = (st.playing ? 0 : 0.05 * wall);
    const P = V3(), toE = V3(), nrm = V3(), away = V3(), look = new THREE.Matrix4();
    for (let k = 0; k < N_MIR; k++) {
      const a = 2 * Math.PI * k / N_MIR + spin;
      P.copy(b1).multiplyScalar(Math.cos(a) * MIR_R).addScaledVector(b2, Math.sin(a) * MIR_R);
      toE.copy(P).negate().normalize();
      const hits = ((k * 7) % N_MIR) < nAimed;                  // 7 and 36 share no factor: aimed mirrors spread around the ring
      const outDir = hits ? toE : away.copy(toE).multiplyScalar(-0.2).addScaledVector(_e1, 1).normalize();
      nrm.copy(sunward).add(outDir).normalize();                                // mirror normal bisects Sun and target
      look.lookAt(V3(), nrm, Math.abs(nrm.z) < 0.95 ? NORTH : b1); _q.setFromRotationMatrix(look);
      _sc.setScalar(1); _m.compose(P, _q, _sc); mirrorPanels.setMatrixAt(k, _m);
      // incoming sunlight: a beam arriving from the Sun's side and ending on the mirror
      const LIN = 3.0;
      rayPos.set([P.x + sunward.x * LIN, P.y + sunward.y * LIN, P.z + sunward.z * LIN, P.x, P.y, P.z], 12 * k);
      // reflected light: aimed mirrors send it down to Earth's surface (Earth = radius 1 in this group);
      // mirrors that are not needed right now tilt it past Earth (drawn much fainter and short)
      const L = hits ? P.length() - 1.0 : 1.6;
      rayPos.set([P.x, P.y, P.z, P.x + outDir.x * L, P.y + outDir.y * L, P.z + outDir.z * L], 12 * k + 6);
      const bi = 0.55, br = hits ? 0.95 : 0.12;          // very light, nearly white (slightly warm) colour
      rayCol.set([0, 0, 0, bi, bi, bi * 0.94, br, br, br * 0.94, br, br, br * 0.94], 12 * k);   // incoming fades in from the Sun side
    }
    mirrorPanels.instanceMatrix.needsUpdate = true; rayGeo.attributes.position.needsUpdate = true; rayGeo.attributes.color.needsUpdate = true;
    mirrorRays.material.opacity = 0.18 + 0.27 * Math.min(1, Math.sqrt((st.mirRel || 0) / 1.0));   // brighter with more mirror light
  }

  let lastWall = performance.now() / 1000;
  function update(st) {
    // st: { t, speedY, playing, sol, belt, dec, layers: {collectors, tracked, mirrors, arms}, hide, mirRel, mirrors,
    //       launching, throwDir, ast }
    const wall = performance.now() / 1000, dt = Math.min(0.1, wall - lastWall); lastWall = wall;
    updateArms(st, dt, wall); updateMirrors(st, wall);
    if (!texDone) { earthMat.map = earthTexture(); earthMat.needsUpdate = true; texDone = true; }
    const t = st.t || 0;
    // Earth spin (sidereal day), frozen at fast playback like the POV view
    if (!(st.playing && st.speedY * 365.25 > 2)) spin.rotation.y = 2 * Math.PI * ((t * 365.25 / 0.99726968) % 1);
    air.visible = !st.hide;
    const sol = st.sol, on = !!(sol && sol.on) && !st.hide;
    // belt motion: frozen when it would spin faster than MAX_REV_PER_S orbits per wall-clock second (strobing)
    const revPerS = st.playing ? st.speedY / (sol && sol.orbit_period_yr || 0.354) : 0;
    const frozen = revPerS > MAX_REV_PER_S;
    if (!frozen) frozenT = null; else if (frozenT === null) frozenT = t;
    const tb = frozen ? frozenT : t;
    sats.visible = on && st.layers.collectors;
    if (sats.visible) {
      setupBelt(sol, st.belt);
      const R = E.R * S;
      for (let k = 0; k < N_INST; k++) {
        const c = inst[k], th = c.u0 + E.n * tb;
        _p.copy(c.P).multiplyScalar(Math.cos(th)).addScaledVector(c.Q, Math.sin(th)).multiplyScalar(R * c.rj);
        place(sats, k, _p, 0.022);
      }
      sats.instanceMatrix.needsUpdate = true;
    }
    const showTrk = on && st.layers.tracked && setupTracked(st.belt);
    trk.visible = orbits.visible = showTrk;
    let devOn = null, devOff = null;
    if (showTrk) {
      devOn = devAt(st.dec, t, 'on'); devOff = devAt(st.dec, t, 'off');
      ghost.visible = links.visible = !!devOff;
      for (let k = 0; k < 12; k++) {
        const th = E.u0[k] + E.nT * tb;
        const pos = (dv) => {
          const a = th + (dv ? dv[1] : 0), rr = (E.RT + (dv ? dv[0] / AU_KM : 0)) * S;
          return V3().copy(E.P[k]).multiplyScalar(Math.cos(a)).addScaledVector(E.Q[k], Math.sin(a)).multiplyScalar(rr).addScaledVector(E.N[k], dv ? dv[2] / AU_KM * S : 0);
        };
        const pOn = pos(devOn ? devOn[k] : null); place(trk, k, pOn, 0.045); halos[k].position.copy(pOn); halos[k].visible = true;
        if (devOff) {
          const pOff = pos(devOff[k]); place(ghost, k, pOff, 0.04); ghalos[k].position.copy(pOff); ghalos[k].visible = true;
          linkPos.set([pOn.x, pOn.y, pOn.z, pOff.x, pOff.y, pOff.z], 6 * k);
        } else ghalos[k].visible = false;
      }
      trk.instanceMatrix.needsUpdate = true; ghost.instanceMatrix.needsUpdate = true;
      if (devOff) { linkGeo.attributes.position.needsUpdate = true; links.computeLineDistances(); }
    } else { ghost.visible = links.visible = false; for (const h of halos) h.visible = false; for (const g of ghalos) g.visible = false; }
    if (showTrk && !devOff) for (const g of ghalos) g.visible = false;
    if (!showTrk) for (const h of halos) h.visible = false;
    return { frozen, devOn, devOff };
  }
  return { update, objects: { sats, trk, ghost, body, armGroup, mirrorGroup } };
}

// --------------------------------------------------------------------------- panel (rail drawer "Ledger")
const sci = (v, d = 2) => (v !== null && v !== undefined && Number.isFinite(+v)) ? (+v).toExponential(d).replace('e+', 'e') : 'N/A';
const num = (v, d = 1) => (v !== null && v !== undefined && Number.isFinite(+v)) ? (+v).toFixed(d) : 'N/A';
const int = (v) => (v !== null && v !== undefined && Number.isFinite(+v)) ? Math.round(v).toLocaleString('en-US') : 'N/A';
const esc = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

export const LEDGER_ROWS = [
  ['sun_on_collectors', 'Sunlight on the collectors', 'in', 'Sunlight hitting the built part of the belt (minus the clear lanes for Earth and Mars).'],
  ['reserve_used', 'Starting reserve used', 'in', 'Energy taken from the starting reserve (only when the beams cannot deliver enough).'],
  ['reflected', 'reflected by the panels', 'out', 'Sunlight the panels do not absorb (1 - absorptance); goes back to space.'],
  ['collector_heat', 'heat radiated by the panels', 'out', 'Absorbed sunlight not turned into used electricity; the panels radiate it (they run ~450 K). Includes the electricity that was available but not needed (the beams are throttled).'],
  ['manufacturing', 'manufacturing + replacement', 'out', 'Embodied energy of new and replacement collectors, paid from their own electricity.'],
  ['transmitter_heat', 'transmitter losses (heat)', 'out', 'Electricity -> microwave conversion losses at the collectors.'],
  ['beam_spill', 'microwaves missing the dishes', 'out', 'Diffraction spill (Goubau). A tiny share may hit Earth (enabler E2: beam spill control).'],
  ['rectenna_heat', 'rectenna losses (heat at the arms)', 'out', 'Microwaves absorbed by the dishes but not turned into electricity.'],
  ['packet_ke', 'kinetic energy of the packets', 'out', 'Useful work: the launcher accelerates the packets (and Earth recoils).'],
  ['launcher_heat', 'launcher losses (heat at the arms)', 'out', 'Launcher electricity not turned into packet kinetic energy (1 - launch efficiency).'],
  ['mirror_sun', 'Sunlight on the space mirrors', 'in2', 'Sunlight falling on the mirror ring (enabler E3). Separate from the power chain: the mirrors use no electricity.'],
  ['mirror_to_earth', 'mirror light onto Earth', 'out2', 'Reflected sunlight the mirrors aim at Earth (the agent sets how much, action a4). Earth absorbs / reflects it and radiates it to space.'],
  ['mirror_to_space', 'mirror light turned past Earth', 'out2', 'Reflected sunlight from the mirrors not needed right now; they are tilted so it passes Earth.'],
  ['mirror_heat', 'mirror heat', 'out2', 'Sunlight absorbed by the mirror film (1 - reflectivity); the mirrors radiate it to space.'],
];

export function climateAtDec(dec, t) {
  if (!dec || !dec.clim_track || !dec.clim_track.length) return dec ? { T: dec.T_surface_K, sun: dec.sun_rel, mirW: dec.mirror_W } : null;
  const tr = dec.clim_track; let k = 0;
  while (k < tr.length - 2 && tr[k + 1][0] < t) k++;
  const a = tr[k], b = tr[Math.min(k + 1, tr.length - 1)], h = b[0] - a[0], f = h > 0 ? Math.max(0, Math.min(1, (t - a[0]) / h)) : 1;
  return { T: a[1] + (b[1] - a[1]) * f, sun: a[2] + (b[2] - a[2]) * f, mirW: a[3] + (b[3] - a[3]) * f };
}

export function ledgerPanelHtml(o) {
  // o: { dec, decs, t, cum, meta, enablers, problems, belt, frozen }
  const { dec, meta } = o, cl = meta && meta.climate, sol = meta && meta.solar;
  let h = '';
  h += '<div class="note">Everything on this page is recorded by the Python physics for the mission being watched (hover a line for an explanation). The climate is part of the RL task: the agent decides how much of the space mirrors\' light goes to Earth (action a4).</div>';
  // climate
  h += '<h4>Climate (two-box energy-balance model)</h4>';
  if (!dec) h += '<div class="dim small">Start a mission (▶ Mission) to see the climate, the energy ledger and the collector belt.</div>';
  else {
    const c = climateAtDec(dec, o.t) || {}, ok = cl ? Math.abs(c.T - cl.target) <= cl.tol : true;
    h += `<table class="small"><tr><td title="Global-mean surface temperature from the climate model.">surface temperature</td><td class="${ok ? 'pass' : 'fail'}">${num(c.T, 1)} K (${num(c.T - 273.15, 1)} °C)</td><td class="l dim">target ${cl ? cl.target : '?'} ± ${cl ? cl.tol : '?'} K</td></tr>`
      + `<tr><td title="Deep-ocean temperature (slow reservoir, ~140-year time constant).">deep ocean</td><td>${num(dec.T_deep_K, 1)} K</td><td class="l dim">end of interval</td></tr>`
      + `<tr><td title="Sunlight reaching Earth at its simulated distance, relative to today.">sunlight</td><td>${num(c.sun, 3)} × today</td><td class="l dim">${num(1361 * c.sun, 0)} W/m²</td></tr>`
      + `<tr><td title="Sunlight the ring of space mirrors reflects onto Earth (enabler E3).">space mirrors</td><td>${num((c.mirW || 0) / (cl ? cl.earth_sun_W : 1.74e17), 3)} × today</td><td class="l dim">${sci(c.mirW)} W · can give up to ${num(dec.mirror_cap_rel, 2)} ×</td></tr>`
      + `<tr><td title="Climate model run with the same sunlight but no mirror light: where the temperature would settle.">without mirrors</td><td class="fail">${num(dec.T_no_mirror_K, 0)} K</td><td class="l dim">${esc(dec.life_no_mirror || '')}</td></tr>`
      + (dec.life ? `<tr><td title="Simple zonal profile (equator +30 K, poles -15 K vs the global mean, like today). Ice where colder than 263 K; growing where 278-313 K.">ice line · growing area</td><td>${num(dec.life.ice_line_deg, 0)}° · ${num(100 * dec.life.grow_frac, 0)} %</td><td class="l dim">${esc(dec.life.state)} (today: 70° · 75 %)</td></tr>` : '')
      + `<tr><td title="Total light the agent asks for in this decision interval (action a4).">agent's set-point</td><td>${num(dec.insol_set_rel, 3)} × today</td><td class="l dim">decision at year ${int(dec.t_years - 100)}</td></tr>`
      + `<tr><td title="Heat from the power chain that reaches Earth (E1 arm radiators, E2 beam spill control).">waste heat on Earth</td><td>${sci(dec.waste_heat_earth_W)} W</td><td class="l dim">${num(100 * dec.waste_heat_earth_W / (cl ? cl.earth_sun_W : 1.74e17), 2)} % of today's sunlight</td></tr>`
      + `<tr><td title="Simulated years so far with the surface temperature more than 15 K from the target.">years outside livable band</td><td>${num(dec.years_outside_livable, 0)}</td><td class="l dim">target ± ${cl ? cl.livable : 15} K</td></tr></table>`;
    h += '<canvas id="chClimate" class="chart" title="Surface temperature (orange) with the target band (green), sunlight (yellow) and mirror light (white) in units of today\'s sunlight (right axis). Simulated time on the x-axis."></canvas>';
  }
  // energy ledger
  h += '<h4>Energy ledger <span class="dim small">(every joule, sunlight → final form)</span></h4>';
  const L = dec && dec.ledger, C = o.cum;
  if (!L || !L.sun_on_collectors) h += '<div class="dim small">No collector energy recorded yet (or the collectors are switched off).</div>';
  else {
    const ein = (x) => (x.sun_on_collectors || 0) + (x.reserve_used || 0), ein2 = (x) => x.mirror_sun || 0;
    h += `<table class="small"><tr><th class="l">where it goes</th><th>this interval [J]</th><th>share</th><th>mission so far [J]</th></tr>`;
    for (const [k, lab, kind, tipTxt] of LEDGER_ROWS) {
      const v = L[k] || 0, vc = C ? C[k] || 0 : NaN;
      const isIn = kind === 'in' || kind === 'in2', den = kind === 'out2' ? ein2(L) : ein(L);
      if (kind === 'in2' && !(L.mirror_sun > 0)) continue;
      if (kind === 'out2' && !(L.mirror_sun > 0)) continue;
      h += `<tr title="${esc(tipTxt)}"><td class="l">${isIn ? '<b>IN</b> ' : '&nbsp;&nbsp;→ '}${esc(lab)}</td><td>${sci(v, 3)}</td><td>${isIn ? '' : (100 * v / Math.max(1e-300, den)).toPrecision(3) + ' %'}</td><td>${sci(vc, 3)}</td></tr>`;
    }
    h += `<tr><td class="l" title="|inputs - sum of all sinks| / inputs. Energy is conserved if this is at round-off level.">residual (conservation)</td><td>${sci(L.residual_rel, 1)}</td><td></td><td></td></tr></table>`;
    h += `<div class="small dim">Chain: panel electricity ${sci(L.panel_electricity, 2)} J → microwaves ${sci(L.microwaves_sent, 2)} J → captured ${sci(L.captured, 2)} J → arm electricity ${sci(L.arm_electricity, 2)} J = launcher ${sci(L.launcher_draw, 2)} J (− reserve share). The space mirrors are a separate flow of plain sunlight (no electricity). Heat reaching Earth: ${sci(L.heat_to_earth, 2)} J.</div>`;
    if (sol) h += `<div class="small">Panels: rated ${num(100 * sol.pv_rated, 0)} % at 25 °C → operating ${num(100 * sol.eta_pv_op, 1)} % (panel temperature ${int(sol.T_panel_K)} K: ×${num(sol.temp_factor, 3)}; age: ×${num(sol.degr_factor, 3)}). Belt built now: ${num(100 * (dec.built || 0), 1)} % (from ${num(100 * sol.built0, 0)} % at year 0, complete after ${int(sol.build_years)} y). Fresh material over the mission ${sci(sol.fresh_mass_total_kg, 2)} kg = ${num(100 * sol.fresh_mass_mercury_frac, 1)} % of Mercury. Earth's sunlight lane ${sol.lane_clear ? 'kept clear' : '<b>NOT kept clear – Earth is in the belt shadow</b>'}.</div>`;
  }
  // belt
  h += '<h4>Do the collectors stay on their orbits?</h4>';
  const b = dec && dec.belt;
  if ((!b || !b.on) && dec && meta && meta.belt) h += '<div class="dim small">Belt simulation for this interval is still being computed. It runs in its own thread behind the mission physics (12+12 collector orbits in half-day steps take about 1 s of computer time per 100 simulated years), so at high playback speed it lags behind the display; until it arrives the collectors are drawn at their assigned slots.</div>';
  else if (!b || !b.on) h += '<div class="dim small">The belt simulation runs while a mission is played (12 collectors followed with and without station keeping).</div>';
  else {
    h += `<table class="small"><tr><th class="l"></th><th>WITH station keeping (solid)</th><th>WITHOUT (faint ghosts)</th></tr>`
      + `<tr><td class="l" title="Largest distance of any of the 12 simulated collectors from its assigned slot during this interval.">max distance from slot</td><td class="pass">${int(b.on.max_km)} km</td><td class="fail">${int(b.off.max_km)} km</td></tr>`
      + `<tr><td class="l" title="Same, measured in collector spacings (neighbouring collectors are this far apart).">… in collector spacings</td><td>${num(b.on.slots, 3)}</td><td>${num(b.off.slots, 1)}</td></tr>`
      + `<tr><td class="l" title="Change of the orbit's size and shape: |Δa| + a·e (largest change of the distance from the Sun).">orbit shape change</td><td>${int(b.on.swing_km)} km</td><td>${int(b.off.swing_km)} km</td></tr>`
      + `<tr><td class="l" title="Velocity change per year that the sail-tilt controller delivers (no propellant: it re-directs the sunlight push).">control effort</td><td>${num(b.on.dv_ms_per_yr, 0)} m/s per year</td><td>0</td></tr></table>`;
    h += `<div class="small">Each collector feels: the Sun (gravity reduced by its light push, ${num(100 * b.light_ratio, 2)} %), the beam recoil <b>${sci(b.recoil_ratio, 1)}</b> × the Sun's pull, Earth+Moon and Mars up to ${sci(b.planet_ratio, 1)} ×, its neighbours ${sci(b.neighbour_ratio, 0)} ×. The sail tilt can correct up to ${sci(b.control_ratio, 1)} ×. Each effect is tiny, but without corrections they add up: the collectors keep their orbit's shape but slide along it out of their slots.</div>`;
    h += `<div class="small dim">${o.frozen ? 'Belt motion is frozen on screen at this playback speed (each collector orbits every ~130 days); slow down to see it move. ' : ''}The deviations with station keeping (hundreds of km) are far too small to see at this scale – the cyan collectors sit exactly on their orbits; the red ghosts are drawn at their true simulated positions.</div>`;
  }
  // enablers + problems
  h += '<h4>Enablers (exactly three; assumed, not shown to be possible)</h4><table class="small"><tr><th class="l">id</th><th class="l">enabler</th><th class="l">gap to reality</th></tr>'
    + (o.enablers || []).map((e) => `<tr title="${esc('Provides: ' + e.provides + '\nParameters: ' + e.params + '\nModelled: ' + e.modelled + (e.instruments ? '\nInstruments: ' + e.instruments.map((i) => i.name).join('; ') : ''))}"><td class="l">${esc(e.id)}</td><td class="l">${esc(e.name)}</td><td class="l">${esc(e.gap)}</td></tr>`).join('') + '</table>';
  h += '<h4>Major problems</h4><table class="small"><tr><th class="l">id</th><th class="l">problem</th><th>status</th></tr>'
    + (o.problems || []).map((m) => `<tr title="${esc(m.detail + '\nRemaining: ' + m.remaining)}"><td class="l">${esc(m.id)}</td><td class="l">${esc(m.title)}</td><td class="${m.status === 'addressed' ? 'pass' : m.status === 'open' ? 'fail' : 'warn'}">${esc(m.status)}</td></tr>`).join('') + '</table>';
  return h;
}

export function drawClimateChart(cv, decs, meta, tNow) {
  if (!cv || !cv.offsetWidth) return;
  const dpr = window.devicePixelRatio || 1, W = cv.offsetWidth, H = 150;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + 'px';
  const g = cv.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const pts = []; for (const d of decs) if (d.clim_track) for (const p of d.clim_track) if (p[0] <= tNow + 1e-9) pts.push(p);
  const cl = meta && meta.climate; if (!pts.length || !cl) { g.fillStyle = '#889'; g.fillText('no climate data yet', 10, 20); return; }
  const t1 = Math.max(tNow, pts[pts.length - 1][0], 1), Tlo = Math.min(cl.target - 40, ...pts.map((p) => p[1])), Thi = Math.max(cl.target + 25, ...pts.map((p) => p[1]));
  const X = (t) => 40 + (W - 80) * t / t1, Y = (T) => H - 18 - (H - 30) * (T - Tlo) / (Thi - Tlo), Y2 = (r) => H - 18 - (H - 30) * r / 2.5;
  g.fillStyle = 'rgba(61,220,132,0.15)'; g.fillRect(40, Y(cl.target + cl.tol), W - 80, Y(cl.target - cl.tol) - Y(cl.target + cl.tol));
  g.strokeStyle = '#334'; g.strokeRect(40, 12, W - 80, H - 30);
  const line = (col, f) => { g.strokeStyle = col; g.beginPath(); pts.forEach((p, k) => { const x = X(p[0]), y = f(p); k ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); };
  g.lineWidth = 1.2; line('#ffd84d', (p) => Y2(p[2])); line('#f4f4ff', (p) => Y2(p[3] / cl.earth_sun_W)); g.lineWidth = 2; line('#ff9a40', (p) => Y(p[1]));
  g.fillStyle = '#aab'; g.font = '10px sans-serif';
  g.fillText(`${Math.round(Thi)} K`, 2, Y(Thi) + 8); g.fillText(`${Math.round(Tlo)} K`, 2, Y(Tlo)); g.fillText(`${cl.target} K`, 2, Y(cl.target) + 3);
  g.fillText('2.5×', W - 34, Y2(2.5) + 8); g.fillText('1×', W - 34, Y2(1) + 3); g.fillText('0', W - 34, Y2(0));
  g.fillText(`year ${Math.round(t1).toLocaleString('en-US')}`, W - 110, H - 4); g.fillText('0', 38, H - 4);
  g.fillStyle = '#ff9a40'; g.fillText('surface T', 46, 22); g.fillStyle = '#ffd84d'; g.fillText('sunlight', 110, 22); g.fillStyle = '#f4f4ff'; g.fillText('mirror light', 160, 22);
}
