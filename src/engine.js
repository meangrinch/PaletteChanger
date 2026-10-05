/**
 * PaletteEngine - Standalone Pure JavaScript Palette Swap Engine
 * Color science and quantizers are copied function-for-function from Img2Pixel's PixelatorEngine
 * (src/engine.js there); port fixes by pasting the whole function. Palette swap code is at the bottom.
 */

(function (root, factory) {
  if (typeof define === 'function' && define.amd) {
    define([], factory);
  } else if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.PaletteEngine = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';


  // --- Helper Math & Color Utilities ---

  function hexToRgba(hex) {
    if (!hex || typeof hex !== 'string') return [0, 0, 0, 255];
    let h = hex.trim();
    if (h.startsWith('#')) h = h.slice(1);
    if (h.length === 3) {
      h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2] + 'ff';
    } else if (h.length === 6) {
      h = h + 'ff';
    } else if (h.length === 8) {
      // standard #rrggbbaa
    } else {
      return [0, 0, 0, 255];
    }
    const r = parseInt(h.slice(0, 2), 16) || 0;
    const g = parseInt(h.slice(2, 4), 16) || 0;
    const b = parseInt(h.slice(4, 6), 16) || 0;
    const a = parseInt(h.slice(6, 8), 16);
    return [r, g, b, isNaN(a) ? 255 : a];
  }

  function rgbaToHex(r, g, b, a = 255) {
    const toHex = (n) => Math.max(0, Math.min(255, Math.round(n))).toString(16).padStart(2, '0');
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
  }

  // --- Color Science: Linear RGB & Oklab ---

  const sRGBtoLinear = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const c = i / 255;
    sRGBtoLinear[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }

  function linearToSRGB(v) {
    if (v <= 0.0) return 0;
    if (v >= 1.0) return 255;
    const c = v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1.0 / 2.4) - 0.055;
    return Math.max(0, Math.min(255, Math.round(c * 255)));
  }

  // Linear-light RGB -> LMS cone response (linear, so averages of colors can be taken here)
  function linearToLmsInto(rLin, gLin, bLin, out, offset = 0) {
    out[offset] = 0.4122214708 * rLin + 0.5363325363 * gLin + 0.0514459929 * bLin;
    out[offset + 1] = 0.2119034982 * rLin + 0.6806995451 * gLin + 0.1073969566 * bLin;
    out[offset + 2] = 0.0883024619 * rLin + 0.2817188376 * gLin + 0.6299787005 * bLin;
    return out;
  }

  // LMS -> Oklab, written into `out` (no allocation; used in hot loops)
  function lmsToOklabInto(l, m, s, out) {
    const l_ = Math.cbrt(l);
    const m_ = Math.cbrt(m);
    const s_ = Math.cbrt(s);
    out[0] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    out[1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    out[2] = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;
    return out;
  }

  function rgbToOklab(r, g, b) {
    const lms = linearToLmsInto(sRGBtoLinear[r], sRGBtoLinear[g], sRGBtoLinear[b], [0, 0, 0]);
    return lmsToOklabInto(lms[0], lms[1], lms[2], lms);
  }

  function oklabToLinear(L, a, b) {
    const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
    const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
    const s_ = L - 0.0894841775 * a - 1.2914855480 * b;

    const l = l_ * l_ * l_;
    const m = m_ * m_ * m_;
    const s = s_ * s_ * s_;

    return [
      +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
      -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    ];
  }

  function oklabToRgb(L, a, b) {
    const [rLin, gLin, bLin] = oklabToLinear(L, a, b);
    return [linearToSRGB(rLin), linearToSRGB(gLin), linearToSRGB(bLin)];
  }

  function colorDistOklabSq(L1, a1, b1, L2, a2, b2) {
    const dL = L1 - L2;
    const da = a1 - a2;
    const db = b1 - b2;
    return dL * dL + da * da + db * db;
  }

  function precomputePaletteOklab(palette) {
    if (!palette) return [];
    return palette.map((col) => {
      let r, g, b;
      if (typeof col === 'string') {
        const rgba = hexToRgba(col);
        r = rgba[0]; g = rgba[1]; b = rgba[2];
      } else if (Array.isArray(col)) {
        r = col[0]; g = col[1]; b = col[2];
      } else if (col && typeof col === 'object') {
        r = col.r; g = col.g; b = col.b;
      } else {
        r = 0; g = 0; b = 0;
      }
      const [L, a, b_] = rgbToOklab(r, g, b);
      return { r, g, b, L, a, b: b_, rgb: [r, g, b] };
    });
  }

  // Median Cut color quantization helper
  function buildMedianCutPalette(buf, maxColors) {
    maxColors = Math.max(2, Math.min(256, Math.round(maxColors) || 8));
    const pixels = [];
    const data = buf.data;
    const step = Math.max(1, Math.floor(data.length / (4 * 5000))); // sample up to 5000 pixels

    for (let i = 0; i < data.length; i += 4 * step) {
      if (data[i + 3] > 64) {
        pixels.push([data[i], data[i + 1], data[i + 2]]);
      }
    }

    if (pixels.length === 0) return [[0, 0, 0]];
    if (pixels.length <= maxColors) return pixels;

    let boxes = [pixels];
    while (boxes.length < maxColors) {
      // Find box with greatest range along any channel
      let maxRange = -1;
      let splitIdx = -1;
      let splitChannel = 0;

      for (let b = 0; b < boxes.length; b++) {
        const box = boxes[b];
        if (box.length <= 1) continue;
        let minR = 255, maxR = 0, minG = 255, maxG = 0, minB = 255, maxB = 0;
        for (let p = 0; p < box.length; p++) {
          const col = box[p];
          if (col[0] < minR) minR = col[0];
          if (col[0] > maxR) maxR = col[0];
          if (col[1] < minG) minG = col[1];
          if (col[1] > maxG) maxG = col[1];
          if (col[2] < minB) minB = col[2];
          if (col[2] > maxB) maxB = col[2];
        }
        const rangeR = maxR - minR;
        const rangeG = maxG - minG;
        const rangeB = maxB - minB;
        const boxMax = Math.max(rangeR, rangeG, rangeB);
        if (boxMax > maxRange) {
          maxRange = boxMax;
          splitIdx = b;
          splitChannel = boxMax === rangeR ? 0 : (boxMax === rangeG ? 1 : 2);
        }
      }

      if (splitIdx === -1 || maxRange === 0) break;

      const targetBox = boxes.splice(splitIdx, 1)[0];
      targetBox.sort((a, b) => a[splitChannel] - b[splitChannel]);
      const median = Math.floor(targetBox.length / 2);
      boxes.push(targetBox.slice(0, median));
      boxes.push(targetBox.slice(median));
    }

    return boxes.map((box) => {
      let sumR = 0, sumG = 0, sumB = 0;
      for (let i = 0; i < box.length; i++) {
        sumR += box[i][0];
        sumG += box[i][1];
        sumB += box[i][2];
      }
      return [
        Math.round(sumR / box.length),
        Math.round(sumG / box.length),
        Math.round(sumB / box.length)
      ];
    });
  }

  // Wu's Fast Optimal 3D Variance Quantizer (Xiaolin Wu, Graphics Gems II)
  function buildWuPalette(buf, maxColors) {
    return buildWuPaletteWeighted(buf, maxColors).map((c) => [c[0], c[1], c[2]]);
  }

  // Same as buildWuPalette, but each entry is [r, g, b, pixelCount]
  function buildWuPaletteWeighted(buf, maxColors) {
    maxColors = Math.max(2, Math.min(1024, Math.round(maxColors) || 8));

    const N = 33;
    const TABLE_SIZE = 35937; // 33 * 33 * 33
    const vwt = new Float64Array(TABLE_SIZE);
    const vmr = new Float64Array(TABLE_SIZE);
    const vmg = new Float64Array(TABLE_SIZE);
    const vmb = new Float64Array(TABLE_SIZE);
    const vmq = new Float64Array(TABLE_SIZE);

    const data = buf.data;
    const len = data.length;
    let pixelCount = 0;

    for (let i = 0; i < len; i += 4) {
      if (data[i + 3] > 64) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const ir = (r >> 3) + 1;
        const ig = (g >> 3) + 1;
        const ib = (b >> 3) + 1;
        const idx = (ir * 33 + ig) * 33 + ib;

        vwt[idx] += 1;
        vmr[idx] += r;
        vmg[idx] += g;
        vmb[idx] += b;
        vmq[idx] += (r * r + g * g + b * b);
        pixelCount++;
      }
    }

    if (pixelCount === 0) return [[0, 0, 0, 1]];

    const areaW = new Float64Array(N);
    const areaMR = new Float64Array(N);
    const areaMG = new Float64Array(N);
    const areaMB = new Float64Array(N);
    const areaMQ = new Float64Array(N);

    // 3D cumulative volume moments via dynamic programming in 33^3 steps
    for (let r = 1; r < N; r++) {
      areaW.fill(0);
      areaMR.fill(0);
      areaMG.fill(0);
      areaMB.fill(0);
      areaMQ.fill(0);
      for (let g = 1; g < N; g++) {
        let lineW = 0, lineMR = 0, lineMG = 0, lineMB = 0, lineMQ = 0;
        for (let b = 1; b < N; b++) {
          const idx = (r * 33 + g) * 33 + b;
          const prev = ((r - 1) * 33 + g) * 33 + b;

          lineW += vwt[idx];
          lineMR += vmr[idx];
          lineMG += vmg[idx];
          lineMB += vmb[idx];
          lineMQ += vmq[idx];

          areaW[b] += lineW;
          areaMR[b] += lineMR;
          areaMG[b] += lineMG;
          areaMB[b] += lineMB;
          areaMQ[b] += lineMQ;

          vwt[idx] = vwt[prev] + areaW[b];
          vmr[idx] = vmr[prev] + areaMR[b];
          vmg[idx] = vmg[prev] + areaMG[b];
          vmb[idx] = vmb[prev] + areaMB[b];
          vmq[idx] = vmq[prev] + areaMQ[b];
        }
      }
    }

    function volume(table, r0, r1, g0, g1, b0, b1) {
      return (
        table[(r1 * 33 + g1) * 33 + b1] -
        table[(r1 * 33 + g1) * 33 + b0] -
        table[(r1 * 33 + g0) * 33 + b1] +
        table[(r1 * 33 + g0) * 33 + b0] -
        table[(r0 * 33 + g1) * 33 + b1] +
        table[(r0 * 33 + g1) * 33 + b0] +
        table[(r0 * 33 + g0) * 33 + b1] -
        table[(r0 * 33 + g0) * 33 + b0]
      );
    }

    function findBestCut(box) {
      if (box.weight <= 1 || box.variance <= 0) return { axis: null, cut: -1 };
      let minVarSum = Infinity;
      let bestAxis = null;
      let bestCut = -1;

      // Axis R
      for (let c = box.r0 + 1; c < box.r1; c++) {
        const w1 = volume(vwt, box.r0, c, box.g0, box.g1, box.b0, box.b1);
        const w2 = box.weight - w1;
        if (w1 <= 0 || w2 <= 0) continue;

        const mr1 = volume(vmr, box.r0, c, box.g0, box.g1, box.b0, box.b1);
        const mg1 = volume(vmg, box.r0, c, box.g0, box.g1, box.b0, box.b1);
        const mb1 = volume(vmb, box.r0, c, box.g0, box.g1, box.b0, box.b1);
        const mq1 = volume(vmq, box.r0, c, box.g0, box.g1, box.b0, box.b1);

        const mr2 = box.mr - mr1;
        const mg2 = box.mg - mg1;
        const mb2 = box.mb - mb1;
        const mq2 = box.mq - mq1;

        const v1 = mq1 - (mr1 * mr1 + mg1 * mg1 + mb1 * mb1) / w1;
        const v2 = mq2 - (mr2 * mr2 + mg2 * mg2 + mb2 * mb2) / w2;
        const sum = v1 + v2;
        if (sum < minVarSum) {
          minVarSum = sum;
          bestAxis = 'r';
          bestCut = c;
        }
      }

      // Axis G
      for (let c = box.g0 + 1; c < box.g1; c++) {
        const w1 = volume(vwt, box.r0, box.r1, box.g0, c, box.b0, box.b1);
        const w2 = box.weight - w1;
        if (w1 <= 0 || w2 <= 0) continue;

        const mr1 = volume(vmr, box.r0, box.r1, box.g0, c, box.b0, box.b1);
        const mg1 = volume(vmg, box.r0, box.r1, box.g0, c, box.b0, box.b1);
        const mb1 = volume(vmb, box.r0, box.r1, box.g0, c, box.b0, box.b1);
        const mq1 = volume(vmq, box.r0, box.r1, box.g0, c, box.b0, box.b1);

        const mr2 = box.mr - mr1;
        const mg2 = box.mg - mg1;
        const mb2 = box.mb - mb1;
        const mq2 = box.mq - mq1;

        const v1 = mq1 - (mr1 * mr1 + mg1 * mg1 + mb1 * mb1) / w1;
        const v2 = mq2 - (mr2 * mr2 + mg2 * mg2 + mb2 * mb2) / w2;
        const sum = v1 + v2;
        if (sum < minVarSum) {
          minVarSum = sum;
          bestAxis = 'g';
          bestCut = c;
        }
      }

      // Axis B
      for (let c = box.b0 + 1; c < box.b1; c++) {
        const w1 = volume(vwt, box.r0, box.r1, box.g0, box.g1, box.b0, c);
        const w2 = box.weight - w1;
        if (w1 <= 0 || w2 <= 0) continue;

        const mr1 = volume(vmr, box.r0, box.r1, box.g0, box.g1, box.b0, c);
        const mg1 = volume(vmg, box.r0, box.r1, box.g0, box.g1, box.b0, c);
        const mb1 = volume(vmb, box.r0, box.r1, box.g0, box.g1, box.b0, c);
        const mq1 = volume(vmq, box.r0, box.r1, box.g0, box.g1, box.b0, c);

        const mr2 = box.mr - mr1;
        const mg2 = box.mg - mg1;
        const mb2 = box.mb - mb1;
        const mq2 = box.mq - mq1;

        const v1 = mq1 - (mr1 * mr1 + mg1 * mg1 + mb1 * mb1) / w1;
        const v2 = mq2 - (mr2 * mr2 + mg2 * mg2 + mb2 * mb2) / w2;
        const sum = v1 + v2;
        if (sum < minVarSum) {
          minVarSum = sum;
          bestAxis = 'b';
          bestCut = c;
        }
      }

      return { axis: bestAxis, cut: bestCut };
    }

    function createBox(r0, r1, g0, g1, b0, b1) {
      const weight = volume(vwt, r0, r1, g0, g1, b0, b1);
      let mr = 0, mg = 0, mb = 0, mq = 0, variance = 0;
      if (weight > 0) {
        mr = volume(vmr, r0, r1, g0, g1, b0, b1);
        mg = volume(vmg, r0, r1, g0, g1, b0, b1);
        mb = volume(vmb, r0, r1, g0, g1, b0, b1);
        mq = volume(vmq, r0, r1, g0, g1, b0, b1);
        variance = mq - (mr * mr + mg * mg + mb * mb) / weight;
        if (variance < 0) variance = 0;
      }
      const vol = (r1 - r0) * (g1 - g0) * (b1 - b0);
      const box = { r0, r1, g0, g1, b0, b1, vol, weight, mr, mg, mb, mq, variance };
      box.cut = findBestCut(box);
      return box;
    }

    const initialBox = createBox(0, 32, 0, 32, 0, 32);
    if (initialBox.weight <= 0) return [[0, 0, 0, 1]];

    const boxes = [initialBox];

    while (boxes.length < maxColors) {
      let maxVar = -1;
      let splitIdx = -1;

      for (let i = 0; i < boxes.length; i++) {
        const box = boxes[i];
        if (box.cut.axis !== null && box.variance > maxVar) {
          maxVar = box.variance;
          splitIdx = i;
        }
      }

      if (splitIdx === -1) break;

      const parentBox = boxes.splice(splitIdx, 1)[0];
      let b1, b2;
      if (parentBox.cut.axis === 'r') {
        b1 = createBox(parentBox.r0, parentBox.cut.cut, parentBox.g0, parentBox.g1, parentBox.b0, parentBox.b1);
        b2 = createBox(parentBox.cut.cut, parentBox.r1, parentBox.g0, parentBox.g1, parentBox.b0, parentBox.b1);
      } else if (parentBox.cut.axis === 'g') {
        b1 = createBox(parentBox.r0, parentBox.r1, parentBox.g0, parentBox.cut.cut, parentBox.b0, parentBox.b1);
        b2 = createBox(parentBox.r0, parentBox.r1, parentBox.cut.cut, parentBox.g1, parentBox.b0, parentBox.b1);
      } else {
        b1 = createBox(parentBox.r0, parentBox.r1, parentBox.g0, parentBox.g1, parentBox.b0, parentBox.cut.cut);
        b2 = createBox(parentBox.r0, parentBox.r1, parentBox.g0, parentBox.g1, parentBox.cut.cut, parentBox.b1);
      }

      boxes.push(b1, b2);
    }

    const palette = [];
    for (let i = 0; i < boxes.length; i++) {
      const box = boxes[i];
      if (box.weight > 0) {
        palette.push([
          Math.round(box.mr / box.weight),
          Math.round(box.mg / box.weight),
          Math.round(box.mb / box.weight),
          box.weight
        ]);
      }
    }

    return palette.length > 0 ? palette : [[0, 0, 0, 1]];
  }

  // Normalize input pixels or image buffer to flat array of [r, g, b] samples
  function parsePixels(pixels, quality = 1) {
    if (!pixels) return [];
    if (pixels.data) pixels = pixels.data;
    const samples = [];
    const q = Math.max(1, Math.floor(quality || 1));

    if (Array.isArray(pixels) || (pixels.buffer && typeof pixels.length === 'number')) {
      if (pixels.length === 0) return [];

      if (typeof pixels[0] === 'object' && pixels[0] !== null) {
        for (let i = 0; i < pixels.length; i += q) {
          const p = pixels[i];
          if (Array.isArray(p)) {
            samples.push([p[0], p[1], p[2]]);
          } else if (p.r !== undefined) {
            samples.push([p.r, p.g, p.b]);
          }
        }
        return samples;
      }

      const isRgba = pixels.length % 4 === 0 && pixels.length >= 4;
      const step = isRgba ? 4 * q : 3 * q;

      if (isRgba) {
        for (let i = 0; i < pixels.length; i += step) {
          const a = pixels[i + 3];
          if (a <= 64) continue;
          samples.push([pixels[i], pixels[i + 1], pixels[i + 2]]);
        }
        if (samples.length === 0) {
          for (let i = 0; i < pixels.length; i += step) {
            samples.push([pixels[i], pixels[i + 1], pixels[i + 2]]);
          }
        }
      } else {
        for (let i = 0; i < pixels.length; i += step) {
          samples.push([pixels[i], pixels[i + 1], pixels[i + 2]]);
        }
      }
    }

    return samples;
  }

  // Direct K-Means++ Color Quantizer: OKLab clustering with chroma-weighted accent preservation, deterministic Maximin seeding, and centroid deduplication (ported from Img2Palette)
  function buildKMeansPalette(buf, maxColors, lockedColors = [], quality = 1, maxIterations = 25, mode = 'balanced') {
    const targetK = Math.max(1, Math.min(256, Math.round(maxColors) || 8));

    // Cap sampling to MAX_KMEANS_SAMPLES (8192) to guarantee responsive execution and prevent UI freezes
    const MAX_KMEANS_SAMPLES = 8192;
    let totalPixels = 0;
    if (buf) {
      const raw = buf.data || buf;
      if (raw && typeof raw.length === 'number') {
        const isRgba = raw.length % 4 === 0 && raw.length >= 4 && !Array.isArray(raw[0]);
        totalPixels = isRgba ? Math.floor(raw.length / 4) : raw.length;
      }
    }

    const effectiveQuality = totalPixels > MAX_KMEANS_SAMPLES
      ? Math.max(Math.floor(quality || 1), Math.ceil(totalPixels / MAX_KMEANS_SAMPLES))
      : Math.max(1, Math.floor(quality || 1));

    const samples = parsePixels(buf, effectiveQuality);

    const validLocked = Array.isArray(lockedColors) ? lockedColors : [];
    if (samples.length === 0) {
      if (validLocked.length > 0) {
        return validLocked.slice(0, targetK).map((c) => [c.r, c.g, c.b]);
      }
      return [[0, 0, 0]];
    }

    // Check unique colors in samples
    const colorMap = new Map();
    for (let i = 0; i < samples.length; i++) {
      const s = samples[i];
      const key = (s[0] << 16) | (s[1] << 8) | s[2];
      colorMap.set(key, (colorMap.get(key) || 0) + 1);
    }

    // If unique colors <= targetK and no locked colors need preservation
    if (colorMap.size <= targetK && validLocked.length === 0) {
      const res = [];
      for (const key of colorMap.keys()) {
        res.push([(key >> 16) & 255, (key >> 8) & 255, key & 255]);
      }
      return res;
    }

    // Convert all samples to OKLab coordinates and compute sample weights
    const nSamples = samples.length;
    const sampleL = new Float64Array(nSamples);
    const sampleA = new Float64Array(nSamples);
    const sampleB = new Float64Array(nSamples);
    const sampleWeight = new Float64Array(nSamples);
    const sampleScoreWeight = new Float64Array(nSamples);

    const chromaWeightFactor = mode === 'vibrant' ? 5.0 : mode === 'dominant' ? 0.0 : 2.0;

    for (let i = 0; i < nSamples; i++) {
      const s = samples[i];
      const [L, a, b] = rgbToOklab(s[0], s[1], s[2]);
      sampleL[i] = L;
      sampleA[i] = a;
      sampleB[i] = b;
      const chroma = Math.hypot(a, b);
      const sw = 1.0 + chromaWeightFactor * chroma;
      sampleWeight[i] = sw;
      const key = (s[0] << 16) | (s[1] << 8) | s[2];
      const count = colorMap.get(key) || 1;
      sampleScoreWeight[i] = sw * Math.log(1 + count);
    }

    // Initialize centroids and flat coordinate buffers
    const centroids = [];
    const centL = new Float64Array(targetK);
    const centA = new Float64Array(targetK);
    const centB = new Float64Array(targetK);

    // 1. Seed locked colors first
    for (let i = 0; i < validLocked.length && centroids.length < targetK; i++) {
      const lc = validLocked[i];
      const r = Math.max(0, Math.min(255, Math.round(Array.isArray(lc) ? lc[0] : lc.r)));
      const g = Math.max(0, Math.min(255, Math.round(Array.isArray(lc) ? lc[1] : lc.g)));
      const b = Math.max(0, Math.min(255, Math.round(Array.isArray(lc) ? lc[2] : lc.b)));
      const [L, a, b_] = rgbToOklab(r, g, b);
      const idx = centroids.length;
      centL[idx] = L;
      centA[idx] = a;
      centB[idx] = b_;
      centroids.push({
        okL: L,
        oka: a,
        okb: b_,
        r,
        g,
        b,
        locked: true,
        count: 0
      });
    }

    // 2. Deterministic density-weighted furthest-point (Maximin) seeding
    if (centroids.length === 0) {
      let maxCount = -1;
      let modeKey = 0;
      for (const [key, count] of colorMap.entries()) {
        if (count > maxCount) {
          maxCount = count;
          modeKey = key;
        }
      }
      const r = (modeKey >> 16) & 255;
      const g = (modeKey >> 8) & 255;
      const b = modeKey & 255;
      const [L, a, b_] = rgbToOklab(r, g, b);
      centL[0] = L;
      centA[0] = a;
      centB[0] = b_;
      centroids.push({
        okL: L,
        oka: a,
        okb: b_,
        r,
        g,
        b,
        locked: false,
        count: 0
      });
    }

    const minDistSq = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let minD = Infinity;
      const sL = sampleL[i], sa = sampleA[i], sb = sampleB[i];
      for (let c = 0; c < centroids.length; c++) {
        const dL = sL - centL[c];
        const da = sa - centA[c];
        const db = sb - centB[c];
        const d = dL * dL + da * da + db * db;
        if (d < minD) minD = d;
      }
      minDistSq[i] = minD;
    }

    while (centroids.length < targetK) {
      let maxScore = -1;
      let nextIdx = 0;
      for (let i = 0; i < nSamples; i++) {
        const score = minDistSq[i] * sampleScoreWeight[i];
        if (score > maxScore) {
          maxScore = score;
          nextIdx = i;
        }
      }

      if (maxScore <= 0) break;

      const s = samples[nextIdx];
      const nL = sampleL[nextIdx];
      const na = sampleA[nextIdx];
      const nb = sampleB[nextIdx];
      const cIdx = centroids.length;
      centL[cIdx] = nL;
      centA[cIdx] = na;
      centB[cIdx] = nb;
      const newCentroid = {
        okL: nL,
        oka: na,
        okb: nb,
        r: s[0],
        g: s[1],
        b: s[2],
        locked: false,
        count: 0
      };
      centroids.push(newCentroid);

      for (let i = 0; i < nSamples; i++) {
        const dL = sampleL[i] - nL;
        const da = sampleA[i] - na;
        const db = sampleB[i] - nb;
        const d = dL * dL + da * da + db * db;
        if (d < minDistSq[i]) minDistSq[i] = d;
      }
    }

    const actualK = centroids.length;

    // 3. Lloyd's iterative refinement loop in OKLab space
    const sumL = new Float64Array(actualK);
    const sumA = new Float64Array(actualK);
    const sumB = new Float64Array(actualK);
    const sumW = new Float64Array(actualK);
    const counts = new Uint32Array(actualK);

    for (let iter = 0; iter < maxIterations; iter++) {
      sumL.fill(0);
      sumA.fill(0);
      sumB.fill(0);
      sumW.fill(0);
      counts.fill(0);

      // Assignment step
      for (let i = 0; i < nSamples; i++) {
        const sL = sampleL[i], sa = sampleA[i], sb = sampleB[i];
        let bestDist = Infinity;
        let bestIdx = 0;
        for (let c = 0; c < actualK; c++) {
          const dL = sL - centL[c];
          const da = sa - centA[c];
          const db = sb - centB[c];
          const d = dL * dL + da * da + db * db;
          if (d < bestDist) {
            bestDist = d;
            bestIdx = c;
          }
        }
        const w = sampleWeight[i];
        sumL[bestIdx] += sL * w;
        sumA[bestIdx] += sa * w;
        sumB[bestIdx] += sb * w;
        sumW[bestIdx] += w;
        counts[bestIdx]++;
      }

      // Update step
      let maxShift = 0;
      for (let c = 0; c < actualK; c++) {
        centroids[c].count = counts[c];
        if (centroids[c].locked) continue;

        if (sumW[c] > 0) {
          const newL = sumL[c] / sumW[c];
          const newA = sumA[c] / sumW[c];
          const newB = sumB[c] / sumW[c];

          const dL = newL - centL[c];
          const da = newA - centA[c];
          const db = newB - centB[c];
          const shift = Math.sqrt(dL * dL + da * da + db * db);
          if (shift > maxShift) maxShift = shift;

          centL[c] = centroids[c].okL = newL;
          centA[c] = centroids[c].oka = newA;
          centB[c] = centroids[c].okb = newB;

          const [r, g, b] = oklabToRgb(newL, newA, newB);
          centroids[c].r = r;
          centroids[c].g = g;
          centroids[c].b = b;
        } else {
          // Reseed empty cluster to sample with highest residual error
          let maxErr = -1;
          let reseedIdx = 0;
          for (let i = 0; i < nSamples; i++) {
            let minD = Infinity;
            const sL = sampleL[i], sa = sampleA[i], sb = sampleB[i];
            for (let k = 0; k < actualK; k++) {
              if (counts[k] > 0) {
                const dL = sL - centL[k];
                const da = sa - centA[k];
                const db = sb - centB[k];
                const d = dL * dL + da * da + db * db;
                if (d < minD) minD = d;
              }
            }
            const err = minD * sampleWeight[i];
            if (err > maxErr) {
              maxErr = err;
              reseedIdx = i;
            }
          }
          centL[c] = centroids[c].okL = sampleL[reseedIdx];
          centA[c] = centroids[c].oka = sampleA[reseedIdx];
          centB[c] = centroids[c].okb = sampleB[reseedIdx];
          centroids[c].r = samples[reseedIdx][0];
          centroids[c].g = samples[reseedIdx][1];
          centroids[c].b = samples[reseedIdx][2];
        }
      }

      if (maxShift < 0.001) break;
    }

    // 4. Centroid Deduplication (Detect near-duplicates with DeltaE_OK < 0.035)
    if (actualK > 1 && nSamples > actualK) {
      let reseededAny = false;
      for (let i = 0; i < actualK; i++) {
        if (centroids[i].locked) continue;
        for (let j = 0; j < i; j++) {
          const dL = centL[i] - centL[j];
          const da = centA[i] - centA[j];
          const db = centB[i] - centB[j];
          const dist = Math.sqrt(dL * dL + da * da + db * db);

          if (dist < 0.035) {
            let maxD = -1;
            let bestIdx = 0;
            for (let s = 0; s < nSamples; s++) {
              let localMin = Infinity;
              const sL = sampleL[s], sa = sampleA[s], sb = sampleB[s];
              for (let c = 0; c < actualK; c++) {
                if (c === i) continue;
                const dL = sL - centL[c];
                const da = sa - centA[c];
                const db = sb - centB[c];
                const d = dL * dL + da * da + db * db;
                if (d < localMin) localMin = d;
              }
              if (localMin * sampleWeight[s] > maxD) {
                maxD = localMin * sampleWeight[s];
                bestIdx = s;
              }
            }

            centL[i] = centroids[i].okL = sampleL[bestIdx];
            centA[i] = centroids[i].oka = sampleA[bestIdx];
            centB[i] = centroids[i].okb = sampleB[bestIdx];
            centroids[i].r = samples[bestIdx][0];
            centroids[i].g = samples[bestIdx][1];
            centroids[i].b = samples[bestIdx][2];
            reseededAny = true;
            break;
          }
        }
      }

      if (reseededAny) {
        for (let iter = 0; iter < 3; iter++) {
          sumL.fill(0);
          sumA.fill(0);
          sumB.fill(0);
          sumW.fill(0);
          counts.fill(0);

          for (let i = 0; i < nSamples; i++) {
            const sL = sampleL[i], sa = sampleA[i], sb = sampleB[i];
            let bestDist = Infinity;
            let bestIdx = 0;
            for (let c = 0; c < actualK; c++) {
              const dL = sL - centL[c];
              const da = sa - centA[c];
              const db = sb - centB[c];
              const d = dL * dL + da * da + db * db;
              if (d < bestDist) {
                bestDist = d;
                bestIdx = c;
              }
            }
            const w = sampleWeight[i];
            sumL[bestIdx] += sL * w;
            sumA[bestIdx] += sa * w;
            sumB[bestIdx] += sb * w;
            sumW[bestIdx] += w;
            counts[bestIdx]++;
          }

          for (let c = 0; c < actualK; c++) {
            centroids[c].count = counts[c];
            if (centroids[c].locked) continue;
            if (sumW[c] > 0) {
              centL[c] = centroids[c].okL = sumL[c] / sumW[c];
              centA[c] = centroids[c].oka = sumA[c] / sumW[c];
              centB[c] = centroids[c].okb = sumB[c] / sumW[c];
              const [r, g, b] = oklabToRgb(centL[c], centA[c], centB[c]);
              centroids[c].r = r;
              centroids[c].g = g;
              centroids[c].b = b;
            }
          }
        }
      }
    }

    return centroids.map((c) => [c.r, c.g, c.b]);
  }

  // Fast Two-Stage K-Means++ Color Quantizer: Wu 3D variance pre-quantization refined by perceptual OKLab K-Means
  function buildKMeansFastPalette(buf, maxColors) {
    maxColors = Math.max(2, Math.min(256, Math.round(maxColors) || 8));

    const colorCounts = new Map();
    const data = buf && buf.data ? buf.data : buf;
    const len = data ? data.length : 0;

    for (let i = 0; i < len; i += 4) {
      if (data[i + 3] > 64) {
        const key = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
        colorCounts.set(key, (colorCounts.get(key) || 0) + 1);
      }
    }

    if (colorCounts.size === 0) return [[0, 0, 0]];
    if (colorCounts.size <= maxColors) {
      const res = [];
      for (const key of colorCounts.keys()) {
        res.push([(key >> 16) & 255, (key >> 8) & 255, key & 255]);
      }
      return res;
    }

    // Stage 1: Candidate pre-quantization
    // If unique colors <= 512, use exact unique colors. Otherwise, extract candidate pool via Wu.
    let candidates = [];
    const candidateCount = Math.min(colorCounts.size, Math.max(2 * maxColors, 256));

    if (colorCounts.size <= 512) {
      for (const [key, count] of colorCounts.entries()) {
        candidates.push({
          r: (key >> 16) & 255,
          g: (key >> 8) & 255,
          b: key & 255,
          count
        });
      }
    } else {
      const wuColors = buildWuPaletteWeighted(buf, candidateCount);
      candidates = wuColors.map((c) => ({ r: c[0], g: c[1], b: c[2], count: c[3] }));
    }

    if (candidates.length <= maxColors) {
      return candidates.map((c) => [c.r, c.g, c.b]);
    }

    // Stage 2: OKLab K-Means refinement with chroma-aware weighting
    const n = candidates.length;
    const cL = new Float64Array(n);
    const ca = new Float64Array(n);
    const cb = new Float64Array(n);
    const weights = new Float64Array(n);

    for (let i = 0; i < n; i++) {
      const c = candidates[i];
      const [L, a, b] = rgbToOklab(c.r, c.g, c.b);
      cL[i] = L;
      ca[i] = a;
      cb[i] = b;
      const chroma = Math.hypot(a, b);
      weights[i] = c.count * (1.0 + 2.0 * chroma);
    }

    const centroidsL = new Float64Array(maxColors);
    const centroidsa = new Float64Array(maxColors);
    const centroidsb = new Float64Array(maxColors);

    // Deterministic Maximin seeding
    let maxW = -1;
    let firstIdx = 0;
    for (let i = 0; i < n; i++) {
      if (weights[i] > maxW) {
        maxW = weights[i];
        firstIdx = i;
      }
    }
    centroidsL[0] = cL[firstIdx];
    centroidsa[0] = ca[firstIdx];
    centroidsb[0] = cb[firstIdx];

    const minDistSq = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const dL = cL[i] - centroidsL[0];
      const da = ca[i] - centroidsa[0];
      const db = cb[i] - centroidsb[0];
      minDistSq[i] = dL * dL + da * da + db * db;
    }

    for (let k = 1; k < maxColors; k++) {
      let maxScore = -1;
      let nextIdx = 0;
      for (let i = 0; i < n; i++) {
        const score = minDistSq[i] * weights[i];
        if (score > maxScore) {
          maxScore = score;
          nextIdx = i;
        }
      }
      centroidsL[k] = cL[nextIdx];
      centroidsa[k] = ca[nextIdx];
      centroidsb[k] = cb[nextIdx];

      for (let i = 0; i < n; i++) {
        const dL = cL[i] - centroidsL[k];
        const da = ca[i] - centroidsa[k];
        const db = cb[i] - centroidsb[k];
        const d = dL * dL + da * da + db * db;
        if (d < minDistSq[i]) minDistSq[i] = d;
      }
    }

    // Lloyd's iterative refinement in OKLab space
    const sumL = new Float64Array(maxColors);
    const sumA = new Float64Array(maxColors);
    const sumB = new Float64Array(maxColors);
    const sumW = new Float64Array(maxColors);

    for (let iter = 0; iter < 8; iter++) {
      sumL.fill(0);
      sumA.fill(0);
      sumB.fill(0);
      sumW.fill(0);

      for (let i = 0; i < n; i++) {
        let bestDist = Infinity;
        let bestK = 0;
        for (let k = 0; k < maxColors; k++) {
          const dL = cL[i] - centroidsL[k];
          const da = ca[i] - centroidsa[k];
          const db = cb[i] - centroidsb[k];
          const d = dL * dL + da * da + db * db;
          if (d < bestDist) {
            bestDist = d;
            bestK = k;
          }
        }
        const w = weights[i];
        sumL[bestK] += cL[i] * w;
        sumA[bestK] += ca[i] * w;
        sumB[bestK] += cb[i] * w;
        sumW[bestK] += w;
      }

      let maxShift = 0;
      for (let k = 0; k < maxColors; k++) {
        if (sumW[k] > 0) {
          const newL = sumL[k] / sumW[k];
          const newA = sumA[k] / sumW[k];
          const newB = sumB[k] / sumW[k];
          const dL = newL - centroidsL[k];
          const da = newA - centroidsa[k];
          const db = newB - centroidsb[k];
          const shift = Math.sqrt(dL * dL + da * da + db * db);
          if (shift > maxShift) maxShift = shift;
          centroidsL[k] = newL;
          centroidsa[k] = newA;
          centroidsb[k] = newB;
        }
      }

      if (maxShift < 0.001) break;
    }

    const palette = [];
    for (let k = 0; k < maxColors; k++) {
      palette.push(oklabToRgb(centroidsL[k], centroidsa[k], centroidsb[k]));
    }
    return palette;
  }


  // Extract adaptive palette using Wu's variance quantizer (default), median cut, K-Means++, or K-Means++ (Fast)
  function buildAdaptivePalette(buf, maxColors, quantizerMethod = 'wu') {
    const qMethod = (quantizerMethod || 'wu').toLowerCase();
    if (qMethod === 'median' || qMethod === 'median_cut' || qMethod === 'median-cut') {
      return buildMedianCutPalette(buf, maxColors);
    }
    if (qMethod === 'kmeans_fast' || qMethod === 'kmeans-fast' || qMethod === 'fast' || qMethod === 'two_stage' || qMethod === 'two-stage') {
      return buildKMeansFastPalette(buf, maxColors);
    }
    if (qMethod === 'kmeans' || qMethod === 'hybrid' || qMethod === 'kmeans_direct' || qMethod === 'kmeans-direct' || qMethod === 'direct') {
      return buildKMeansPalette(buf, maxColors);
    }
    return buildWuPalette(buf, maxColors);
  }

  // Extract a clean palette from an image buffer:
  // - Swatch strips (<= maxColors unique colors) keep exact colors losslessly
  // - Continuous-tone images (> maxColors) are adaptively quantized via selected quantizer
  // Returns array of hex strings sorted monotonically by Oklab perceptual lightness
  function extractPalette(buf, maxColors = 256, quantizerMethod = 'wu') {
    if (!buf || !buf.data) return [];
    maxColors = Math.max(2, Math.min(256, Math.round(maxColors) || 256));

    const colorCounts = new Map();
    const data = buf.data;
    const len = data.length;

    for (let i = 0; i < len; i += 4) {
      if (data[i + 3] > 64) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const key = (r << 16) | (g << 8) | b;
        colorCounts.set(key, (colorCounts.get(key) || 0) + 1);
        if (colorCounts.size > maxColors) {
          // Exceeds maxColors: cannot be a discrete swatch strip, stop collecting
          break;
        }
      }
    }

    if (colorCounts.size === 0) return ['#000000'];

    let paletteRgb = [];

    if (colorCounts.size <= maxColors) {
      for (const key of colorCounts.keys()) {
        paletteRgb.push([
          (key >> 16) & 255,
          (key >> 8) & 255,
          key & 255
        ]);
      }
    } else {
      paletteRgb = buildAdaptivePalette(buf, maxColors, quantizerMethod);
    }

    paletteRgb.sort((c1, c2) => {
      const l1 = rgbToOklab(c1[0], c1[1], c1[2])[0];
      const l2 = rgbToOklab(c2[0], c2[1], c2[2])[0];
      return l1 - l2;
    });

    const seen = new Set();
    const hexList = [];
    for (let i = 0; i < paletteRgb.length; i++) {
      const hex = rgbaToHex(paletteRgb[i][0], paletteRgb[i][1], paletteRgb[i][2]);
      if (!seen.has(hex)) {
        seen.add(hex);
        hexList.push(hex);
      }
    }

    return hexList.length > 0 ? hexList : ['#000000'];
  }


  // --- Palette swap ---
  // Smooth mode weights each anchor by 1 / distance^SMOOTH_POWER: higher keeps regions closer to their own
  // anchor's shift, lower blends shifts across the image
  const SMOOTH_POWER = 2;

  // Unique colors among pixels extractPalette would sample (alpha > 64), stopping once past limit
  function countColors(buf, limit) {
    const seen = new Set();
    const data = buf.data;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] <= 64) continue;
      seen.add((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]);
      if (seen.size > limit) break;
    }
    return seen.size;
  }

  // Exact colors when the image has <= maxAnchors of them, quantized otherwise; sorted dark to light
  function findAnchors(buf, maxAnchors, quantizerMethod = 'wu') {
    return extractPalette(buf, maxAnchors, quantizerMethod);
  }

  // One target hex per anchor. 'rank' pairs anchors and palette by Oklab lightness order, so tonal
  // structure survives any hue change; 'nearest' takes the closest palette color.
  function autoMap(anchors, palette, method = 'rank') {
    if (!palette || palette.length === 0) return anchors.slice();
    const pal = precomputePaletteOklab(palette).map((p, i) => ({ ...p, hex: palette[i] }));
    const anc = precomputePaletteOklab(anchors);
    if (method === 'nearest') {
      return anc.map((a) => {
        let best = pal[0];
        let bestDist = Infinity;
        for (const p of pal) {
          const d = colorDistOklabSq(a.L, a.a, a.b, p.L, p.a, p.b);
          if (d < bestDist) { bestDist = d; best = p; }
        }
        return best.hex;
      });
    }
    const palSorted = pal.slice().sort((p, q) => p.L - q.L);
    const order = anc.map((a, i) => i).sort((i, j) => anc[i].L - anc[j].L);
    const P = palSorted.length;
    const A = order.length;
    const out = new Array(A);
    order.forEach((anchorIdx, rank) => {
      const idx = A === 1 ? Math.round((P - 1) / 2) : Math.round(rank * (P - 1) / (A - 1));
      out[anchorIdx] = palSorted[idx].hex;
    });
    return out;
  }

  // Recolors buf by mapping anchors[i] -> targets[i]. 'exact' snaps each pixel to its nearest anchor's
  // target; 'smooth' shifts each pixel in Oklab by the distance-weighted blend of anchor shifts, keeping
  // all shading. Returns a new buffer; alpha is copied and fully transparent pixels are left as-is.
  function swap(buf, anchors, targets, mode = 'smooth') {
    const src = buf.data;
    const data = new Uint8ClampedArray(src);
    const out = { width: buf.width, height: buf.height, data };
    const anc = precomputePaletteOklab(anchors);
    const tgt = precomputePaletteOklab(targets);
    const n = anc.length;
    if (n === 0) return out;
    const dL = anc.map((a, i) => tgt[i].L - a.L);
    const dA = anc.map((a, i) => tgt[i].a - a.a);
    const dB = anc.map((a, i) => tgt[i].b - a.b);

    const recolor = (r, g, b) => {
      const [L, a, b_] = rgbToOklab(r, g, b);
      if (mode === 'exact') {
        let best = 0;
        let bestDist = Infinity;
        for (let i = 0; i < n; i++) {
          const d = colorDistOklabSq(L, a, b_, anc[i].L, anc[i].a, anc[i].b);
          if (d < bestDist) { bestDist = d; best = i; }
        }
        return tgt[best].rgb;
      }
      let wSum = 0, sL = 0, sA = 0, sB = 0;
      for (let i = 0; i < n; i++) {
        const d = colorDistOklabSq(L, a, b_, anc[i].L, anc[i].a, anc[i].b);
        if (d < 1e-12) return oklabToRgb(L + dL[i], a + dA[i], b_ + dB[i]);
        const w = 1 / Math.pow(d, SMOOTH_POWER / 2);
        wSum += w; sL += w * dL[i]; sA += w * dA[i]; sB += w * dB[i];
      }
      return oklabToRgb(L + sL / wSum, a + sA / wSum, b_ + sB / wSum);
    };

    // Packed-int values: photos reach ~1M unique colors, and an array per entry costs hundreds of MB
    const cache = new Map();
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] === 0) continue;
      const key = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
      let packed = cache.get(key);
      if (packed === undefined) {
        const rgb = recolor(data[i], data[i + 1], data[i + 2]);
        packed = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2];
        cache.set(key, packed);
      }
      data[i] = packed >> 16;
      data[i + 1] = (packed >> 8) & 255;
      data[i + 2] = packed & 255;
    }
    return out;
  }

  return {
    hexToRgba,
    rgbaToHex,
    sRGBtoLinear,
    linearToSRGB,
    rgbToOklab,
    oklabToRgb,
    colorDistOklabSq,
    precomputePaletteOklab,
    buildAdaptivePalette,
    buildWuPalette,
    buildMedianCutPalette,
    buildKMeansPalette,
    buildKMeansFastPalette,
    extractPalette,
    countColors,
    findAnchors,
    autoMap,
    swap
  };
});
