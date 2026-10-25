// OCTO-EARTH realistic Earth, star field and Sun for the cinematic views (update p37).
//
// Textures are bundled locally in frontend/assets/earth/ (no network access at run time). They are derived from
// NASA public-domain imagery (see assets/earth/CREDITS.txt):
//   earth_day_4096.jpg      Blue Marble Next Generation, Dec 2004, topography + bathymetry (NASA Earth Observatory)
//   earth_night_4096.jpg    Black Marble 2016 city lights (NASA Earth Observatory / Suomi NPP VIIRS), lights only
//   earth_clouds_2048.jpg   Blue Marble cloud composite (NASA Visible Earth), used as cloud opacity
//   earth_normal_2048.jpg   approximate relief shading DERIVED from the day image (land brightness), not elevation data
//   earth_ocean_mask_2048.png  water mask derived from the day image (where the ocean glint is drawn)
// If a texture cannot be loaded the views fall back to the procedural Earth texture of the POV (pov.js) and say so in
// earthTextureStatus().
//
// Lighting is done in the shader: day side lit by the Sun direction, a soft twilight band, city lights on the night
// side only, a sun glint on water, clouds that are lit / dark with the same terminator, and a thin atmosphere rim.
import * as THREE from './vendor/three.module.js';
import { earthTexture } from './pov.js';
import { glowTex } from './collector.js';

const BASE = './assets/earth/';
export const EARTH_ASSETS = {
  day: 'earth_day_4096.jpg', night: 'earth_night_4096.jpg', clouds: 'earth_clouds_2048.jpg',
  normal: 'earth_normal_2048.jpg', ocean: 'earth_ocean_mask_2048.png',
};
const STATUS = { requested: false, loaded: [], failed: [] };
let TEX = null;
const blackTex = () => { const t = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1); t.needsUpdate = true; return t; };
const flatNormal = () => { const t = new THREE.DataTexture(new Uint8Array([128, 128, 255, 255]), 1, 1); t.needsUpdate = true; return t; };

/** Load (once) and share the Earth textures; textures appear when loaded, fallbacks are used until then. */
export function earthTextures(renderer) {
  if (TEX) return TEX;
  STATUS.requested = true;
  const loader = new THREE.TextureLoader();
  const aniso = renderer ? Math.min(8, renderer.capabilities.getMaxAnisotropy()) : 1;
  TEX = { day: earthTexture(), night: blackTex(), clouds: blackTex(), normal: flatNormal(), ocean: blackTex(), real: false, listeners: [] };
  for (const [key, file] of Object.entries(EARTH_ASSETS)) {
    loader.load(BASE + file, (t) => {
      if (key === 'day' || key === 'night') t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = aniso; t.wrapS = THREE.RepeatWrapping;
      TEX[key] = t; STATUS.loaded.push(file); if (key === 'day') TEX.real = true;
      for (const f of TEX.listeners) f(key, t);
    }, undefined, () => { STATUS.failed.push(file); console.warn('[OCTO-EARTH] texture not found, using fallback:', BASE + file); });
  }
  return TEX;
}
export function earthTextureStatus() {
  return { ...STATUS, real: !!(TEX && TEX.real), note: STATUS.failed.length ? `missing: ${STATUS.failed.join(', ')} (procedural fallback in use)` : '' };
}

const EARTH_VS = `
varying vec2 vUv; varying vec3 vN; varying vec3 vT; varying vec3 vW;
uniform vec3 axis;
void main(){
  vUv = uv;
  vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz;
  vN = normalize(mat3(modelMatrix) * normal);
  vT = normalize(cross(axis, vN) + vec3(1e-5));
  gl_Position = projectionMatrix * viewMatrix * w;
}`;
const EARTH_FS = `
uniform sampler2D dayMap, nightMap, normalMap, oceanMap, cloudMap;
uniform vec3 sunDir; uniform float cloudShift; uniform float nightGain; uniform float bumpK;
uniform vec3 heatDir; uniform float heatSpot; uniform float spotCos; uniform float heatGlobal; uniform float scorch;   // p38 laser heating
varying vec2 vUv; varying vec3 vN; varying vec3 vT; varying vec3 vW;
void main(){
  vec3 N = normalize(vN), T = normalize(vT), B = normalize(cross(N, T));
  vec3 nm = texture2D(normalMap, vUv).xyz * 2.0 - 1.0;
  vec3 Np = normalize(T * nm.x * bumpK + B * nm.y * bumpK + N * nm.z);
  vec3 L = normalize(sunDir), V = normalize(cameraPosition - vW);
  float ndl = dot(N, L), lit = max(dot(Np, L), 0.0);
  float day = smoothstep(-0.12, 0.18, ndl);
  vec3 dcol = texture2D(dayMap, vUv).rgb;
  float sh = texture2D(cloudMap, vUv + vec2(cloudShift - 0.0025, 0.0)).g;       // cloud shadow
  vec3 col = dcol * (0.025 + 1.25 * lit * (1.0 - 0.45 * sh)) * day;
  col += vec3(0.95, 0.42, 0.18) * 0.10 * exp(-pow(ndl / 0.10, 2.0)) * (1.0 - sh);   // twilight band
  float ocean = texture2D(oceanMap, vUv).r;
  vec3 H = normalize(L + V);
  col += vec3(1.0, 0.92, 0.78) * pow(max(dot(N, H), 0.0), 70.0) * ocean * day * 0.85 * (1.0 - sh);
  col += vec3(0.25, 0.45, 0.85) * pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.35 * day;   // haze towards the limb
  float night = smoothstep(0.08, -0.2, ndl);
  col += texture2D(nightMap, vUv).rgb * night * nightGain * (1.0 - 0.7 * sh);
  // p38: heating by an external beam (all zero by default = normal Earth)
  float hd = dot(N, normalize(heatDir));
  float spot = heatSpot * smoothstep(spotCos - 0.06, spotCos + 0.04, hd);
  float burnt = scorch * max(smoothstep(spotCos - 0.45, spotCos, hd), heatGlobal);
  float lum = dot(dcol, vec3(0.3, 0.55, 0.15));
  col = mix(col, vec3(0.06, 0.045, 0.035) * (0.4 + 2.0 * lum) * (0.15 + lit), burnt);          // charred ground, no seas
  float g = max(spot, heatGlobal);
  vec3 hot = mix(vec3(0.9, 0.12, 0.02), vec3(1.0, 0.55, 0.12), smoothstep(0.15, 0.55, g));
  hot = mix(hot, vec3(1.0, 0.93, 0.8), smoothstep(0.8, 1.0, g));
  float crack = 0.75 + 0.5 * texture2D(normalMap, vUv * 3.0).r;                                  // lava texture
  float crack2 = 0.55 + 0.9 * texture2D(normalMap, vUv * 7.0 + vec2(0.13, 0.31)).g * texture2D(dayMap, vUv * 2.0).r;
  col = mix(col, hot * crack * crack2 * (0.6 + 0.4 * g), clamp(g * 1.3, 0.0, 1.0));
  gl_FragColor = vec4(col, 1.0);
  #include <colorspace_fragment>
}`;
const CLOUD_FS = `
uniform sampler2D cloudMap; uniform vec3 sunDir; uniform float fade;
varying vec2 vUv; varying vec3 vN; varying vec3 vT; varying vec3 vW;
void main(){
  float a = texture2D(cloudMap, vUv).g; a = smoothstep(0.08, 0.85, a);
  vec3 N = normalize(vN); float ndl = dot(N, normalize(sunDir));
  float l = 0.03 + 1.05 * smoothstep(-0.15, 0.35, ndl) * max(ndl + 0.15, 0.0);
  vec3 c = vec3(1.0) * l + vec3(0.9, 0.45, 0.25) * 0.10 * exp(-pow(ndl / 0.1, 2.0));
  gl_FragColor = vec4(c, a * 0.92 * fade);
  #include <colorspace_fragment>
}`;
const ATM_FS = `
uniform vec3 sunDir; uniform float hot; varying vec3 vN; varying vec3 vW; varying vec2 vUv; varying vec3 vT;
void main(){
  vec3 N = normalize(vN), V = normalize(cameraPosition - vW);
  float rim = pow(1.0 - abs(dot(N, V)), 2.6);
  float lit = smoothstep(-0.35, 0.45, dot(N, normalize(sunDir)));
  vec3 c = mix(vec3(0.32, 0.58, 1.0) * (0.08 + 1.25 * lit), vec3(1.0, 0.45, 0.12) * (1.0 + 2.0 * hot), hot);
  float a = mix(0.08 + lit, 1.0, hot);
  gl_FragColor = vec4(c * rim, rim * a);
}`;

/**
 * A realistic Earth (radius 1 in its own units; scale the returned group). Axis = +z of the group (tilt it with the
 * group's rotation if wanted). Call setSun(worldDirection) whenever the Sun direction changes and update(dt) to spin.
 */
export function createRealEarth(renderer, opt = {}) {
  const o = { segments: 128, spinRate: 0.02, cloudRate: 0.0035, ...opt };
  const T = earthTextures(renderer);
  const group = new THREE.Group();
  const spin = new THREE.Group(); group.add(spin);
  const sunDir = new THREE.Vector3(1, 0, 0);
  const axisW = new THREE.Vector3(0, 0, 1);
  const uni = {
    dayMap: { value: T.day }, nightMap: { value: T.night }, normalMap: { value: T.normal }, oceanMap: { value: T.ocean }, cloudMap: { value: T.clouds },
    sunDir: { value: sunDir }, axis: { value: axisW }, cloudShift: { value: 0 }, nightGain: { value: 1.7 }, bumpK: { value: 0.9 },
    heatDir: { value: new THREE.Vector3(1, 0, 0) }, heatSpot: { value: 0 }, spotCos: { value: 0.99 }, heatGlobal: { value: 0 }, scorch: { value: 0 },
  };
  const surface = new THREE.Mesh(new THREE.SphereGeometry(1, o.segments, o.segments / 2),
    new THREE.ShaderMaterial({ uniforms: uni, vertexShader: EARTH_VS, fragmentShader: EARTH_FS }));
  surface.rotation.x = Math.PI / 2; spin.add(surface);                      // texture north pole on +z
  const cloudUni = { cloudMap: { value: T.clouds }, sunDir: { value: sunDir }, axis: { value: axisW }, fade: { value: 1 } };
  const clouds = new THREE.Mesh(new THREE.SphereGeometry(1.012, o.segments, o.segments / 2),
    new THREE.ShaderMaterial({ uniforms: cloudUni, vertexShader: EARTH_VS, fragmentShader: CLOUD_FS, transparent: true, depthWrite: false }));
  clouds.rotation.x = Math.PI / 2; spin.add(clouds);
  const atmUni = { sunDir: { value: sunDir }, axis: { value: axisW }, hot: { value: 0 } };
  const atmo = new THREE.Mesh(new THREE.SphereGeometry(1.045, 96, 48),
    new THREE.ShaderMaterial({ uniforms: atmUni, vertexShader: EARTH_VS, fragmentShader: ATM_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  group.add(atmo);
  T.listeners.push((key, t) => {
    if (key === 'day') uni.dayMap.value = t; else if (key === 'night') uni.nightMap.value = t; else if (key === 'normal') uni.normalMap.value = t;
    else if (key === 'ocean') uni.oceanMap.value = t; else if (key === 'clouds') { uni.cloudMap.value = t; cloudUni.cloudMap.value = t; }
  });
  let cloudRot = 0;
  function setSun(dirWorld) { sunDir.copy(dirWorld).normalize(); }
  function update(dt) {
    spin.rotation.z += o.spinRate * dt; cloudRot += o.cloudRate * dt;
    clouds.rotation.y = cloudRot;                                            // clouds drift relative to the ground
    uni.cloudShift.value = -cloudRot / (2 * Math.PI);
    group.updateMatrixWorld(); axisW.set(0, 0, 1).transformDirection(group.matrixWorld);
  }
  return { group, spin, surface, clouds, atmo, setSun, update, uniforms: uni, cloudUniforms: cloudUni, atmUniforms: atmUni };
}

/** Background stars: several thousand points with slightly different colours and brightness. */
export function createStarfield(radius = 4000, n = 5000, seed = 1) {
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  const h = (i, k) => { const x = Math.sin((i + 1) * 12.9898 + (k + seed) * 78.233) * 43758.5453; return x - Math.floor(x); };
  for (let i = 0; i < n; i++) {
    const z = 2 * h(i, 1) - 1, ph = 2 * Math.PI * h(i, 2), r = Math.sqrt(1 - z * z);
    pos.set([r * Math.cos(ph) * radius, r * Math.sin(ph) * radius, z * radius], 3 * i);
    const b = 0.25 + 0.75 * Math.pow(h(i, 3), 3), t = h(i, 4);
    col.set(t < 0.15 ? [b, b * 0.85, b * 0.7] : t > 0.85 ? [b * 0.8, b * 0.88, b] : [b, b, b], 3 * i);
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const pts = new THREE.Points(g, new THREE.PointsMaterial({ size: 1.6, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false }));
  pts.frustumCulled = false; return pts;
}

const SUN_VS = 'varying vec3 vP; varying vec3 vN; varying vec3 vW; void main(){ vP = position; vN = normalize(mat3(modelMatrix) * normal); vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }';
const SUN_FS = `
uniform float time; varying vec3 vP; varying vec3 vN; varying vec3 vW;
float hash(vec3 p){ p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float noise(vec3 x){ vec3 i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x), mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x), mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z); }
void main(){
  vec3 p = normalize(vP) * 7.0;
  float g = 0.5 * noise(p + time * 0.15) + 0.3 * noise(p * 2.3 - time * 0.21) + 0.2 * noise(p * 5.1 + time * 0.33);
  vec3 V = normalize(cameraPosition - vW); float mu = max(dot(normalize(vN), V), 0.0);
  float limb = 0.45 + 0.55 * pow(mu, 0.55);                                   // limb darkening
  vec3 c = mix(vec3(1.0, 0.55, 0.16), vec3(1.0, 0.95, 0.78), smoothstep(0.25, 0.8, g)) * limb * 1.35;
  gl_FragColor = vec4(c, 1.0);
}`;
/** Luminous Sun: granulated, limb-darkened surface + additive glow sprites. Radius in scene units. */
export function createSun(radius = 6) {
  const group = new THREE.Group();
  const uni = { time: { value: 0 } };
  group.add(new THREE.Mesh(new THREE.SphereGeometry(radius, 64, 32), new THREE.ShaderMaterial({ uniforms: uni, vertexShader: SUN_VS, fragmentShader: SUN_FS })));
  const glowA = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex('rgba(255,236,190,1)', 'rgba(255,170,60,0)'), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0.95 }));
  glowA.scale.setScalar(radius * 4.2); group.add(glowA);
  const glowB = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex('rgba(255,200,120,0.6)', 'rgba(255,120,40,0)'), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0.55 }));
  glowB.scale.setScalar(radius * 10); group.add(glowB);
  const light = new THREE.PointLight(0xfff2dd, 3.0, 0, 0); group.add(light);
  return { group, light, update(t) { uni.time.value = t; } };
}

/** Simple environment map (dark sky, bright Sun spot) so the metal of the reflector has something to reflect. */
export function makeSpaceEnv(renderer, sunDir = new THREE.Vector3(-1, 0, 0)) {
  const s = new THREE.Scene();
  s.add(new THREE.Mesh(new THREE.SphereGeometry(50, 32, 16), new THREE.ShaderMaterial({
    side: THREE.BackSide, uniforms: { sd: { value: sunDir.clone().normalize() } },
    vertexShader: 'varying vec3 vD; void main(){ vD = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform vec3 sd; varying vec3 vD; void main(){ float s = max(dot(normalize(vD), sd), 0.0); vec3 c = vec3(0.035, 0.045, 0.075) + vec3(0.18, 0.2, 0.26) * (0.5 + 0.5 * vD.z) + vec3(1.0, 0.85, 0.6) * (pow(s, 300.0) * 40.0 + pow(s, 8.0) * 0.6); gl_FragColor = vec4(c, 1.0); }',
  })));
  const pm = new THREE.PMREMGenerator(renderer);
  const rt = pm.fromScene(s, 0.02); pm.dispose();
  return rt.texture;
}
