// OCTO-EARTH shared arm rig (update p37).
//
// The launcher ring and its 8 arms, as ONE reusable component. It is the same mechanism that the solar-system
// overview draws on Earth (ovextras.js used to build these meshes itself; it now calls createArmRig), and the two
// cinematic views (hydrogenScene.js, dysonScene.js) use exactly the same meshes, materials and arm shape:
//
//   * ring of radius RING_R around Earth's centre, 4 struts down to the surface, 8 root hubs (the shoulders)
//   * every arm has the fixed length ARM_L = 2 Earth radii and is drawn as NSEG instanced segments that taper
//     towards the claw; its shape comes from armCurve() in launcher.js (it bows instead of stretching)
//   * 3 joints per arm (shoulder / elbow / wrist, orange) and a 3-finger claw at the tip (the end effector)
//
// Everything is in Earth radii in the parent's coordinates with Earth's centre at the origin; the caller scales the
// group. The ring plane is spanned by e1, e2 and its axis is ax (setFrame).
//
// Extra features used by the cinematic views (they do nothing in the overview):
//   * strain (0..1+): joints and segments are tinted towards hot red and the arm vibrates
//   * breakArm(): the arm snaps at a segment; the outer part flies off as one rigid piece (position + rotation are
//     analytic functions of time, so pause / replay are exact), the stump droops
//   * a pool of pre-created fragment shards (spawnFragments) - nothing is created or deleted while animating
import * as THREE from './vendor/three.module.js';
import { ARM_L, RING_R, armCurve } from './launcher.js';

const V3 = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const UPY = V3(0, 1, 0);
const _q = new THREE.Quaternion(), _q2 = new THREE.Quaternion(), _m = new THREE.Matrix4(), _sc = V3(), _u = V3(), _mid = V3(), _seg = V3(),
  _a = V3(), _b = V3(), _c = V3(), _side = V3(), _side2 = V3(), _col = new THREE.Color();
const COL_COOL = new THREE.Color(1, 1, 1), COL_HOT = new THREE.Color(1.0, 0.22, 0.12), COL_SEG_HOT = new THREE.Color(1.0, 0.55, 0.45);

/** Materials of the mechanism. 'overview' = the exact look of the solar-system overview (self-lit so it reads at
 *  small size); 'cinematic' = the same colours, less self-glow, more metal, for the lit close-up scenes. */
export function armMaterials(look = 'overview') {
  if (look === 'cinematic') {
    return {
      metal: new THREE.MeshStandardMaterial({ color: 0xc4ccd8, metalness: 0.75, roughness: 0.3, emissive: 0x1a2230, emissiveIntensity: 0.6 }),
      dark: new THREE.MeshStandardMaterial({ color: 0x66717f, metalness: 0.7, roughness: 0.38, emissive: 0x0c1118, emissiveIntensity: 0.6 }),
      joint: new THREE.MeshStandardMaterial({ color: 0xffb060, metalness: 0.3, roughness: 0.45, emissive: 0x6a3008, emissiveIntensity: 0.9 }),
    };
  }
  return {
    metal: new THREE.MeshStandardMaterial({ color: 0xc4ccd8, metalness: 0.55, roughness: 0.35, emissive: 0x3a4658, emissiveIntensity: 0.9 }),
    dark: new THREE.MeshStandardMaterial({ color: 0x66717f, metalness: 0.6, roughness: 0.4, emissive: 0x1c2430, emissiveIntensity: 0.9 }),
    joint: new THREE.MeshStandardMaterial({ color: 0xffb060, emissive: 0xff8a30, emissiveIntensity: 1.1 }),
  };
}

function placeInstance(mesh, idx, pos, dir, sx, sy, sz) {
  _q.setFromUnitVectors(UPY, _u.copy(dir).normalize()); _sc.set(sx, sy, sz); _m.compose(pos, _q, _sc); mesh.setMatrixAt(idx, _m);
}
function hideInstance(mesh, idx) { _sc.set(0, 0, 0); _m.compose(_mid.set(0, 0, 0), _q.identity(), _sc); mesh.setMatrixAt(idx, _m); }

/**
 * Build the ring + 8 arms.
 * opt: { nseg = 10, look = 'overview' | 'cinematic', fragments = 0 (size of the shard pool), armL = ARM_L, ringR = RING_R }
 */
export function createArmRig(opt = {}) {
  const o = { n: 8, nseg: 10, look: 'overview', fragments: 0, armL: ARM_L, ringR: RING_R, ...opt };
  const N = o.n, NSEG = o.nseg;
  const group = new THREE.Group();
  const mat = armMaterials(o.look);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(o.ringR, 0.07, 10, 128), mat.metal); group.add(ring);
  const unitCyl = new THREE.CylinderGeometry(1, 1, 1, 8);
  const struts = new THREE.InstancedMesh(unitCyl, mat.dark, 4);
  const segs = new THREE.InstancedMesh(unitCyl, mat.metal, N * NSEG);
  const joints = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 10, 8), mat.joint, N * 3);
  const claws = new THREE.InstancedMesh(new THREE.ConeGeometry(0.06, 0.42, 6), mat.dark, N * 3);
  const hubs = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.13, 0.15, 0.2, 10), mat.dark, N);
  for (const m of [struts, segs, joints, claws, hubs]) { m.frustumCulled = false; group.add(m); }
  for (let i = 0; i < N * NSEG; i++) segs.setColorAt(i, COL_COOL);
  for (let i = 0; i < N * 3; i++) joints.setColorAt(i, COL_COOL);
  // pre-created fragment shards (hidden until spawned; never created or removed while animating)
  const NF = o.fragments;
  const frags = NF ? new THREE.InstancedMesh(new THREE.TetrahedronGeometry(1, 0), mat.dark, NF) : null;
  const fragState = [...Array(NF)].map(() => ({ on: false, t0: 0, p0: V3(), v: V3(), axis: V3(1, 0, 0), w: 0, s: 0.05 }));
  if (frags) { frags.frustumCulled = false; group.add(frags); for (let i = 0; i < NF; i++) hideInstance(frags, i); }

  const F = { e1: V3(1, 0, 0), e2: V3(0, 1, 0), ax: V3(0, 0, 1), phase: 0 };
  const arms = [...Array(N)].map((_, k) => ({
    k, dir: V3(), base: V3(), pts: [...Array(NSEG + 1)].map(() => V3()), tip: V3(), open: 0.35, strain: 0, broken: null, side: V3(),
  }));

  /** Ring plane e1, e2 (unit, orthogonal), axis ax; phaseDeg rotates the 8 roots around the axis. */
  function setFrame(e1, e2, ax, phaseDeg = 0) {
    F.e1.copy(e1); F.e2.copy(e2); F.ax.copy(ax); F.phase = phaseDeg * Math.PI / 180;
    _m.makeBasis(F.e1, F.e2, F.ax); ring.quaternion.setFromRotationMatrix(_m);
    for (let s2 = 0; s2 < 4; s2++) {           // struts from the surface up to the ring
      const a = (s2 * 90 + 22.5) * Math.PI / 180 + F.phase; _seg.copy(F.e1).multiplyScalar(Math.cos(a)).addScaledVector(F.e2, Math.sin(a));
      _mid.copy(_seg).multiplyScalar((1 + o.ringR) / 2); placeInstance(struts, s2, _mid, _seg, 0.04, o.ringR - 0.98, 0.04);
    }
    struts.instanceMatrix.needsUpdate = true;
    for (const arm of arms) {
      const ang = arm.k * Math.PI / 4 + F.phase;
      arm.dir.copy(F.e1).multiplyScalar(Math.cos(ang)).addScaledVector(F.e2, Math.sin(ang));
      arm.base.copy(arm.dir).multiplyScalar(o.ringR + 0.05);
      arm.side.copy(F.ax).multiplyScalar(arm.k % 2 ? -1 : 1);
      _mid.copy(arm.dir).multiplyScalar(o.ringR + 0.03); placeInstance(hubs, arm.k, _mid, arm.dir, 1, 1, 1);
    }
    hubs.instanceMatrix.needsUpdate = true;
  }
  setFrame(F.e1, F.e2, F.ax, 0);

  /** Rest pose target of arm k (reaching outward, alternately above / below the ring, slow sway with time t). */
  function restTarget(k, t, out) {
    const arm = arms[k];
    return out.copy(arm.base).addScaledVector(arm.dir, 1.25 + 0.12 * Math.sin(0.9 * t + k))
      .addScaledVector(F.ax, (k % 2 ? -0.7 : 0.7) + 0.18 * Math.sin(0.7 * t + 1.3 * k));
  }

  /** Shape of an intact arm: armCurve (fixed length) + a strain vibration across the arm. */
  function shapeArm(arm, T, t) {
    armCurve(arm.base, T, arm.side, arm.pts, NSEG, o.armL);
    if (arm.strain > 0.02) {
      _a.subVectors(arm.pts[NSEG], arm.pts[0]).normalize(); _b.crossVectors(_a, arm.side).normalize();
      const amp = 0.05 * Math.min(2, arm.strain) ** 1.5, w = 23 + 3 * arm.k;
      for (let j = 1; j <= NSEG; j++) {
        const env = Math.sin(Math.PI * j / NSEG) + 0.3 * j / NSEG;
        arm.pts[j].addScaledVector(_b, amp * env * Math.sin(w * t + 1.7 * arm.k + 0.6 * j)).addScaledVector(arm.side, 0.6 * amp * env * Math.sin(0.7 * w * t + arm.k));
      }
    }
  }

  /** Points of a broken arm at time t: stump drooping from the root, outer piece flying as one rigid body. */
  function shapeBroken(arm, t) {
    const B = arm.broken, dt = Math.max(0, t - B.t0);
    const droop = B.droop * (1 - Math.exp(-3 * dt)) + 0.05 * Math.exp(-1.2 * dt) * Math.sin(9 * dt);
    _q.setFromAxisAngle(B.droopAxis, droop);
    for (let j = 0; j <= B.s; j++) arm.pts[j].copy(B.stump[j]).sub(arm.base).applyQuaternion(_q).add(arm.base);
    _q2.setFromAxisAngle(B.axis, B.w * dt);
    _c.copy(B.p0).addScaledVector(B.v, dt);
    for (let j = B.s; j <= NSEG; j++) {
      const p = B.outer[j - B.s];
      B.outPts[j - B.s].copy(p).applyQuaternion(_q2).add(_c);
    }
  }

  /**
   * Pose all arms. targets[k] = claw target (Vector3, may be pulled in to reach); opts:
   *   open:   claw opening (number or array, radians), strain: number or array (0 = relaxed, >= 1 = overloaded)
   *   t:      time (s) for vibration and broken-piece motion
   */
  function pose(targets, opts = {}) {
    const t = opts.t || 0;
    for (const arm of arms) {
      arm.open = Array.isArray(opts.open) ? opts.open[arm.k] : (opts.open ?? 0.35);
      arm.strain = Array.isArray(opts.strain) ? opts.strain[arm.k] : (opts.strain || 0);
      if (arm.broken) shapeBroken(arm, t);
      else { _a.copy(targets[arm.k]); shapeArm(arm, _a, t); }
      const B = arm.broken;
      const P = (j) => (B && j > B.s ? B.outPts[j - B.s] : arm.pts[j]);
      const Pseg0 = (j) => (B && j >= B.s ? B.outPts[j - B.s] : arm.pts[j]);   // segment j runs from point j to j+1
      for (let j = 0; j < NSEG; j++) {
        const a = Pseg0(j), b = P(j + 1); _mid.addVectors(a, b).multiplyScalar(0.5); _seg.subVectors(b, a);
        const r = 0.085 - 0.035 * j / NSEG; placeInstance(segs, arm.k * NSEG + j, _mid, _seg, r, _seg.length() * 1.05, r);
        segs.setColorAt(arm.k * NSEG + j, _col.copy(COL_COOL).lerp(COL_SEG_HOT, Math.min(1, Math.max(0, arm.strain - 0.4) / 0.8)));
      }
      for (let j = 0; j < 3; j++) {
        const p = P(Math.round((j + 1) * NSEG / 3.5)); _sc.setScalar(0.1 - 0.012 * j); _m.compose(p, _q.identity(), _sc); joints.setMatrixAt(arm.k * 3 + j, _m);
        joints.setColorAt(arm.k * 3 + j, _col.copy(COL_COOL).lerp(COL_HOT, Math.min(1, Math.max(0, arm.strain - 0.3) / 0.7)));
      }
      const tip = P(NSEG); arm.tip.copy(tip);
      _seg.subVectors(tip, P(NSEG - 1)).normalize();
      _side.crossVectors(_seg, F.ax); if (_side.lengthSq() < 1e-6) _side.set(1, 0, 0); _side.normalize(); _side2.crossVectors(_seg, _side);
      for (let c = 0; c < 3; c++) {
        const th = c * 2.094; _a.copy(_side).multiplyScalar(Math.cos(th)).addScaledVector(_side2, Math.sin(th));
        _b.copy(_seg).multiplyScalar(Math.cos(arm.open)).addScaledVector(_a, Math.sin(arm.open));
        placeInstance(claws, arm.k * 3 + c, _c.copy(tip).addScaledVector(_b, 0.2), _b, 1, 1, 1);
      }
    }
    for (const m of [segs, joints, claws]) m.instanceMatrix.needsUpdate = true;
    segs.instanceColor.needsUpdate = true; joints.instanceColor.needsUpdate = true;
    updateFragments(t);
  }

  /**
   * Snap arm k between point s-1 and s (1 <= s < NSEG) at time t0. The outer part flies off with velocity v
   * (Earth radii per second) and spins at w rad/s around a random axis; the stump droops by 'droop' radians.
   * Uses the arm's CURRENT shape, so call it after a pose().
   */
  function breakArm(k, s, t0, v, w = 1.5, droop = 0.45, axisSeed = 0) {
    const arm = arms[k]; if (arm.broken) return false;
    s = Math.max(1, Math.min(NSEG - 1, s));
    const p0 = arm.pts[s].clone();
    const axis = V3(Math.sin(1.7 + axisSeed * 2.3), Math.cos(0.9 + axisSeed * 1.1), Math.sin(0.4 + axisSeed)).normalize();
    arm.broken = {
      s, t0, p0, v: v.clone(), w, axis, droop,
      droopAxis: V3().crossVectors(arm.dir, F.ax).normalize().multiplyScalar(arm.k % 2 ? 1 : -1),
      stump: arm.pts.slice(0, s + 1).map((p) => p.clone()),
      outer: arm.pts.slice(s).map((p) => p.clone().sub(p0)),
      outPts: arm.pts.slice(s).map((p) => p.clone()),
    };
    return true;
  }

  /** Release n shards from the pool at position p (parent coords) at time t0, around velocity v. */
  let fragNext = 0;
  function spawnFragments(p, n, t0, v, spread = 0.6, size = 0.05, seed = 0) {
    for (let i = 0; i < n && NF; i++) {
      const f = fragState[fragNext]; fragNext = (fragNext + 1) % NF;
      const h1 = Math.sin(12.9898 * (i + 1) + 78.233 * (seed + 1)) * 43758.5453, h2 = Math.sin(39.346 * (i + 1) + 11.135 * (seed + 2)) * 24634.6345, h3 = Math.sin(73.156 * (i + 1) + 52.235 * (seed + 3)) * 12345.678;
      const r1 = h1 - Math.floor(h1), r2 = h2 - Math.floor(h2), r3 = h3 - Math.floor(h3);
      const th = 2 * Math.PI * r1, z = 2 * r2 - 1, rr = Math.sqrt(1 - z * z);
      f.on = true; f.t0 = t0; f.p0.copy(p); f.v.copy(v).add(_a.set(rr * Math.cos(th), rr * Math.sin(th), z).multiplyScalar(spread * (0.35 + 0.65 * r3)));
      f.axis.set(r2 - 0.5, r3 - 0.5, r1 - 0.5).normalize(); f.w = 2 + 6 * r3; f.s = size * (0.5 + r1);
    }
  }
  function updateFragments(t) {
    if (!NF) return;
    for (let i = 0; i < NF; i++) {
      const f = fragState[i];
      if (!f.on || t < f.t0) { hideInstance(frags, i); continue; }
      const dt = t - f.t0; _q.setFromAxisAngle(f.axis, f.w * dt); _sc.setScalar(f.s);
      _m.compose(_c.copy(f.p0).addScaledVector(f.v, dt), _q, _sc); frags.setMatrixAt(i, _m);
    }
    frags.instanceMatrix.needsUpdate = true;
  }

  /** Back to intact arms, no fragments. */
  function reset() {
    for (const arm of arms) { arm.broken = null; arm.strain = 0; }
    for (const f of fragState) f.on = false; fragNext = 0;
  }
  /** Position of joint j (0 shoulder .. 2 wrist) of arm k. */
  function jointPoint(k, j, out) { return out.copy(arms[k].pts[Math.round((j + 1) * NSEG / 3.5)]); }

  return {
    group, arms, frame: F, setFrame, restTarget, pose, breakArm, spawnFragments, reset, jointPoint,
    meshes: { ring, struts, segs, joints, claws, hubs, frags }, materials: mat, NSEG, N, armL: o.armL, ringR: o.ringR,
    brokenCount: () => arms.filter((a) => a.broken).length,
  };
}
