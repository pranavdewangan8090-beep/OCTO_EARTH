// OCTO-EARTH launcher geometry (update p36), shared by the Earth POV (pov.js) and the overview (ovextras.js).
// Same numbers as ARM_LENGTH_RE / RING_RADIUS_RE / GRAB_HEIGHT_RE in octoearth_rl.py (a test compares them).
//
//  * 8 arm roots on a ring of radius RING_R around Earth (in the plane perpendicular to Earth's orbit normal).
//  * The docked asteroid sits on the ring's AXIS: its Earth-facing side rests on the grab point at GRAB_Z, so all
//    8 roots are exactly the same distance from it: sqrt(RING_R^2 + GRAB_Z^2) = 1.77 R_E < ARM_L = 2 R_E.
//    The docking clamps lower the asteroid as it shrinks, so the grab point never moves.
//  * Every arm is drawn with the same length ARM_L: when its claw is closer than ARM_L the arm bends (bows away
//    from Earth) instead of stretching.
// All lengths in Earth radii, Earth's centre at the origin.
import * as THREE from './vendor/three.module.js';

export const ARM_L = 2.0, RING_R = 1.25, GRAB_Z = 1.25;

const _d = new THREE.Vector3(), _m = new THREE.Vector3(), _b = new THREE.Vector3(), _p1 = new THREE.Vector3(), _p2 = new THREE.Vector3(), _x = new THREE.Vector3();

/** Root positions of the 8 arms for ring axes e1, e2 (unit vectors in the ring plane). */
export function rootDir(k, e1, e2, out) { const a = k * Math.PI / 4; return out.copy(e1).multiplyScalar(Math.cos(a)).addScaledVector(e2, Math.sin(a)); }

/**
 * Stroke of the joint throw: how far from the grab point G the packet can be carried along unit vector dir while
 * EVERY root (roots: array of Vector3) stays within ARM_L of it.  Bisection; 0 if G itself is out of reach.
 */
export function strokeLength(G, dir, roots) {
  const ok = (s) => roots.every((r) => _x.copy(G).addScaledVector(dir, s).distanceTo(r) <= ARM_L + 1e-9);
  if (!ok(0)) return 0;
  let lo = 0, hi = ARM_L;
  for (let i = 0; i < 40; i++) { const mid = 0.5 * (lo + hi); if (ok(mid)) lo = mid; else hi = mid; }
  return lo;
}

function bez(p0, p1, p2, p3, t, out) {
  const u = 1 - t; return out.set(0, 0, 0).addScaledVector(p0, u * u * u).addScaledVector(p1, 3 * u * u * t).addScaledVector(p2, 3 * u * t * t).addScaledVector(p3, t * t * t);
}
function curveLen(p0, p1, p2, p3) {
  let L = 0; const a = new THREE.Vector3().copy(p0), b = new THREE.Vector3();
  for (let j = 1; j <= 24; j++) { bez(p0, p1, p2, p3, j / 24, b); L += a.distanceTo(b); a.copy(b); }
  return L;
}

/**
 * Shape of one arm of fixed length L (default ARM_L) from base to the claw target T (T is pulled in if it is out of reach).
 * The arm bows away from Earth's centre (and towards 'side' a little) until its length is L.
 * Writes nseg + 1 points into pts and returns the (possibly pulled-in) target.
 */
export function armCurve(base, T, side, pts, nseg, L = ARM_L) {
  _d.subVectors(T, base); let c = _d.length();
  if (c > L) { _d.multiplyScalar(L / c); T.copy(base).add(_d); c = L; }
  _m.addVectors(base, T).multiplyScalar(0.5);
  _b.copy(_m); if (_b.lengthSq() < 1e-6) _b.copy(side); _b.normalize().addScaledVector(side, 0.35);
  _b.addScaledVector(_d, -_b.dot(_d) / Math.max(1e-9, c * c)).normalize();      // bow perpendicular to the chord
  let lo = 0, hi = L;
  for (let i = 0; i < 22; i++) {
    const h = 0.5 * (lo + hi);
    _p1.copy(base).addScaledVector(_d, 1 / 3).addScaledVector(_b, h); _p2.copy(base).addScaledVector(_d, 2 / 3).addScaledVector(_b, h);
    if (curveLen(base, _p1, _p2, T) < L) lo = h; else hi = h;
  }
  const h = 0.5 * (lo + hi);
  _p1.copy(base).addScaledVector(_d, 1 / 3).addScaledVector(_b, h); _p2.copy(base).addScaledVector(_d, 2 / 3).addScaledVector(_b, h);
  for (let j = 0; j <= nseg; j++) {
    bez(base, _p1, _p2, T, j / nseg, pts[j]);
    const r = pts[j].length(); if (r < 1.04) pts[j].multiplyScalar(1.04 / Math.max(1e-6, r));   // never inside Earth
  }
  return T;
}
