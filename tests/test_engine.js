const assert = require('assert');
const PaletteEngine = require('../src/engine.js');

console.log('--- Running PaletteEngine Unit Tests ---');

// Helper to create simple ImageData-like buffer
function createBuffer(w, h, fill = [0, 0, 0, 0]) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    data[i * 4] = fill[0];
    data[i * 4 + 1] = fill[1];
    data[i * 4 + 2] = fill[2];
    data[i * 4 + 3] = fill[3];
  }
  return { width: w, height: h, data };
}

function getPixel(buf, x, y) {
  if (x < 0 || x >= buf.width || y < 0 || y >= buf.height) return [0, 0, 0, 0];
  const idx = (y * buf.width + x) * 4;
  return [buf.data[idx], buf.data[idx + 1], buf.data[idx + 2], buf.data[idx + 3]];
}

function setPixel(buf, x, y, col) {
  const idx = (y * buf.width + x) * 4;
  buf.data[idx] = col[0];
  buf.data[idx + 1] = col[1];
  buf.data[idx + 2] = col[2];
  buf.data[idx + 3] = col[3];
}

// --- Shared with Img2Pixel (tests copied from Img2Pixel/tests/test_engine.js) ---

{
  console.log('Test 6: Color Science & Oklab');
  assert.ok(PaletteEngine.sRGBtoLinear, 'sRGBtoLinear LUT must exist');
  assert.strictEqual(PaletteEngine.sRGBtoLinear.length, 256, 'sRGBtoLinear must have 256 entries');
  assert.strictEqual(PaletteEngine.sRGBtoLinear[0], 0, 'sRGBtoLinear[0] must be 0');
  assert.ok(Math.abs(PaletteEngine.sRGBtoLinear[255] - 1.0) < 1e-5, 'sRGBtoLinear[255] must be ~1.0');

  assert.strictEqual(PaletteEngine.linearToSRGB(0.0), 0, 'linearToSRGB(0) must be 0');
  assert.strictEqual(PaletteEngine.linearToSRGB(1.0), 255, 'linearToSRGB(1) must be 255');
  assert.strictEqual(PaletteEngine.linearToSRGB(PaletteEngine.sRGBtoLinear[128]), 128, 'linearToSRGB roundtrip 128');

  const blackLab = PaletteEngine.rgbToOklab(0, 0, 0);
  assert.ok(Math.abs(blackLab[0] - 0.0) < 1e-4, `Black L should be ~0, got ${blackLab[0]}`);

  const whiteLab = PaletteEngine.rgbToOklab(255, 255, 255);
  assert.ok(Math.abs(whiteLab[0] - 1.0) < 1e-4, `White L should be ~1.0, got ${whiteLab[0]}`);

  // Perceptual test: verify peach [245, 185, 155] is closer to skin [240, 190, 160] than olive [190, 190, 130] in Oklab space
  const peachLab = PaletteEngine.rgbToOklab(245, 185, 155);
  const skinLab = PaletteEngine.rgbToOklab(240, 190, 160);
  const oliveLab = PaletteEngine.rgbToOklab(190, 190, 130);
  const distPeachSkin = PaletteEngine.colorDistOklabSq(peachLab[0], peachLab[1], peachLab[2], skinLab[0], skinLab[1], skinLab[2]);
  const distPeachOlive = PaletteEngine.colorDistOklabSq(peachLab[0], peachLab[1], peachLab[2], oliveLab[0], oliveLab[1], oliveLab[2]);
  assert.ok(distPeachSkin < distPeachOlive, `Peach should be closer to skin (${distPeachSkin}) than olive (${distPeachOlive})`);

  // Test precomputePaletteOklab
  const precomp = PaletteEngine.precomputePaletteOklab(['#ffffff', [0, 0, 0]]);
  assert.strictEqual(precomp.length, 2);
  assert.strictEqual(precomp[0].r, 255);
  assert.strictEqual(precomp[0].g, 255);
  assert.deepStrictEqual(precomp[0].rgb, [255, 255, 255]);
  assert.ok(Math.abs(precomp[0].L - 1.0) < 0.01);
  assert.ok(Math.abs(precomp[0].a - 0.0) < 0.01);
  assert.ok(Math.abs(precomp[0].b - 0.0) < 0.01);

  assert.strictEqual(precomp[1].r, 0);
  assert.strictEqual(precomp[1].g, 0);
  assert.deepStrictEqual(precomp[1].rgb, [0, 0, 0]);
  assert.ok(Math.abs(precomp[1].L - 0.0) < 0.01);
  assert.ok(Math.abs(precomp[1].a - 0.0) < 0.01);
  assert.ok(Math.abs(precomp[1].b - 0.0) < 0.01);

  console.log('  Passed!');
}

// 7. Test Pipeline Reordering & Edge Defringing (Priority 2)

{
  console.log('Test 9: Wu Variance Quantizer & Adaptive Palette');
  // Subtest 9A: Accent Preservation Test
  // Create buffer with 95 dark navy pixels [10, 10, 50, 255] and 5 bright yellow pixels [255, 240, 0, 255]
  const buf = createBuffer(100, 1);
  for (let i = 0; i < 95; i++) {
    setPixel(buf, i, 0, [10, 10, 50, 255]);
  }
  for (let i = 95; i < 100; i++) {
    setPixel(buf, i, 0, [255, 240, 0, 255]);
  }

  // Run buildWuPalette(buf, 2)
  const wuPalette = PaletteEngine.buildWuPalette(buf, 2);
  assert.strictEqual(wuPalette.length, 2, 'Wu palette should extract exactly 2 colors');

  // Assert one color is dark navy and the other color is bright yellow (r > 200, g > 200, b < 50)
  const yellowColor = wuPalette.find((c) => c[0] > 200 && c[1] > 200 && c[2] < 50);
  const navyColor = wuPalette.find((c) => c[0] < 50 && c[1] < 50 && c[2] > 20);
  assert.ok(yellowColor, `Wu quantizer must preserve bright yellow accent color (got ${JSON.stringify(wuPalette)})`);
  assert.ok(navyColor, `Wu quantizer must preserve dark navy color (got ${JSON.stringify(wuPalette)})`);

  // Subtest 9B: Test buildAdaptivePalette with 'wu' (and default)
  const adaptiveDef = PaletteEngine.buildAdaptivePalette(buf, 2);
  assert.strictEqual(adaptiveDef.length, 2);
  const defYellow = adaptiveDef.find((c) => c[0] > 200 && c[1] > 200 && c[2] < 50);
  assert.ok(defYellow, 'buildAdaptivePalette default should use Wu and preserve yellow accent');

  const adaptiveExplicitWu = PaletteEngine.buildAdaptivePalette(buf, 2, 'wu');
  assert.deepStrictEqual(adaptiveExplicitWu, adaptiveDef, 'buildAdaptivePalette("wu") should match default');

  // Subtest 9C: Test buildAdaptivePalette with 'median'
  const adaptiveMedian = PaletteEngine.buildAdaptivePalette(buf, 2, 'median');
  assert.strictEqual(adaptiveMedian.length, 2, 'Median cut should produce 2 colors');

  // Subtest 9E: Edge cases - transparent buffer and single color buffer
  const transBuf = createBuffer(10, 10, [0, 0, 0, 0]);
  const transPal = PaletteEngine.buildWuPalette(transBuf, 4);
  assert.deepStrictEqual(transPal, [[0, 0, 0]], 'Transparent buffer should return fallback [0, 0, 0]');

  const singleColorBuf = createBuffer(10, 10, [120, 80, 200, 255]);
  const singlePal = PaletteEngine.buildWuPalette(singleColorBuf, 8);
  assert.strictEqual(singlePal.length, 1, 'Single color buffer should return 1 color');
  assert.deepStrictEqual(singlePal[0], [120, 80, 200], 'Single color should match source color');

  // Subtest 9F: Test buildKMeansPalette (direct K-Means++) and buildKMeansFastPalette (two-stage)
  const kMeansPal = PaletteEngine.buildKMeansPalette(buf, 2);
  assert.strictEqual(kMeansPal.length, 2, 'K-Means++ palette should produce 2 colors');
  const kMeansYellow = kMeansPal.find((c) => c[0] > 200 && c[1] > 200 && c[2] < 50);
  assert.ok(kMeansYellow, 'K-Means++ quantizer must preserve yellow accent');

  const kMeansFastPal = PaletteEngine.buildKMeansFastPalette(buf, 2);
  assert.strictEqual(kMeansFastPal.length, 2, 'K-Means++ Fast palette should produce 2 colors');
  const kMeansFastYellow = kMeansFastPal.find((c) => c[0] > 200 && c[1] > 200 && c[2] < 50);
  assert.ok(kMeansFastYellow, 'K-Means++ Fast quantizer must preserve yellow accent');

  // Determinism check: multiple runs on identical buffer must produce identical results
  const kMeansPal2 = PaletteEngine.buildKMeansPalette(buf, 2);
  assert.deepStrictEqual(kMeansPal, kMeansPal2, 'K-Means++ quantizer must be 100% deterministic');
  const kMeansFastPal2 = PaletteEngine.buildKMeansFastPalette(buf, 2);
  assert.deepStrictEqual(kMeansFastPal, kMeansFastPal2, 'K-Means++ Fast quantizer must be 100% deterministic');

  // Direct K-Means++ locked colors preservation
  const lockedGreen = [{ r: 0, g: 255, b: 0 }];
  const kMeansLocked = PaletteEngine.buildKMeansPalette(buf, 3, lockedGreen);
  assert.strictEqual(kMeansLocked.length, 3, 'K-Means++ with locked color should return 3 colors');
  const hasLockedGreen = kMeansLocked.some((c) => c[0] === 0 && c[1] === 255 && c[2] === 0);
  assert.ok(hasLockedGreen, 'Direct K-Means++ must preserve locked color');

  // Direct K-Means++ deduplication test
  const dedupBuf = createBuffer(4, 1);
  setPixel(dedupBuf, 0, 0, [20, 20, 20, 255]);
  setPixel(dedupBuf, 1, 0, [200, 200, 200, 255]);
  setPixel(dedupBuf, 2, 0, [255, 0, 0, 255]);
  setPixel(dedupBuf, 3, 0, [0, 0, 255, 255]);
  const dedupPal = PaletteEngine.buildKMeansPalette(dedupBuf, 4);
  assert.strictEqual(dedupPal.length, 4, 'Deduplication must maintain 4 distinct clusters');
  for (let i = 0; i < dedupPal.length; i++) {
    for (let j = i + 1; j < dedupPal.length; j++) {
      const lab1 = PaletteEngine.rgbToOklab(dedupPal[i][0], dedupPal[i][1], dedupPal[i][2]);
      const lab2 = PaletteEngine.rgbToOklab(dedupPal[j][0], dedupPal[j][1], dedupPal[j][2]);
      const d = PaletteEngine.colorDistOklabSq(lab1[0], lab1[1], lab1[2], lab2[0], lab2[1], lab2[2]);
      assert.ok(d > 0.001, 'Centroids should not collapse into identical points');
    }
  }

  // Subtest 9G: Test buildAdaptivePalette with 'kmeans', 'kmeans_fast' (and aliases)
  const adaptiveKMeans = PaletteEngine.buildAdaptivePalette(buf, 2, 'kmeans');
  assert.deepStrictEqual(adaptiveKMeans, kMeansPal, 'buildAdaptivePalette("kmeans") should match buildKMeansPalette');
  const adaptiveKMeansFast = PaletteEngine.buildAdaptivePalette(buf, 2, 'kmeans_fast');
  assert.deepStrictEqual(adaptiveKMeansFast, kMeansFastPal, 'buildAdaptivePalette("kmeans_fast") should match buildKMeansFastPalette');
  const adaptiveFastAlias = PaletteEngine.buildAdaptivePalette(buf, 2, 'fast');
  assert.deepStrictEqual(adaptiveFastAlias, kMeansFastPal, 'buildAdaptivePalette("fast") should match buildKMeansFastPalette');
  const adaptiveTwoStageAlias = PaletteEngine.buildAdaptivePalette(buf, 2, 'two_stage');
  assert.deepStrictEqual(adaptiveTwoStageAlias, kMeansFastPal, 'buildAdaptivePalette("two_stage") should match buildKMeansFastPalette');

  // Subtest 9I: Test extractPalette with quantizerMethod="kmeans" and "kmeans_fast"
  const extractedKMeans = PaletteEngine.extractPalette(buf, 2, 'kmeans');
  assert.strictEqual(extractedKMeans.length, 2, 'extractPalette with "kmeans" should return 2 hex colors');
  const extractedKMeansFast = PaletteEngine.extractPalette(buf, 2, 'kmeans_fast');
  assert.strictEqual(extractedKMeansFast.length, 2, 'extractPalette with "kmeans_fast" should return 2 hex colors');

  // Subtest 9J: Performance & correctness check for direct K-Means++ on large buffer (512x512, 256 colors)
  const largeBuf = createBuffer(512, 512);
  for (let y = 0; y < 512; y++) {
    for (let x = 0; x < 512; x++) {
      setPixel(largeBuf, x, y, [
        Math.round((x / 512) * 255),
        Math.round((y / 512) * 255),
        Math.round(((x + y) / 1024) * 255),
        255
      ]);
    }
  }
  const tStartKMeans = Date.now();
  const largeKMeansPal = PaletteEngine.extractPalette(largeBuf, 256, 'kmeans');
  const elapsedKMeans = Date.now() - tStartKMeans;
  assert.strictEqual(largeKMeansPal.length, 256, 'Large buffer K-Means++ should produce 256 colors');
  assert.ok(elapsedKMeans < 1500, `Large buffer K-Means++ must complete responsively (took ${elapsedKMeans}ms, expected < 1500ms)`);
  for (const hex of largeKMeansPal) {
    assert.match(hex, /^#[0-9a-f]{6}$/i, `Color ${hex} must be a valid hex string`);
  }

  console.log('  Passed!');
}

// 10. Test Retro Dithering Suite (Bayer, Atkinson, Serpentine Floyd-Steinberg, ditherAmount, colorMetric)

// --- Test 15: Dual-Mode Palette Extraction (Lossless <= 256 & Quantized > 256) ---
console.log('Test 15: Dual-Mode Palette Extraction');
{
  // 15A: Lossless extraction for swatch strips (<= 256 colors)
  const swatchBuf = createBuffer(4, 1);
  setPixel(swatchBuf, 0, 0, [255, 255, 255, 255]); // White
  setPixel(swatchBuf, 1, 0, [0, 0, 0, 255]);       // Black
  setPixel(swatchBuf, 2, 0, [255, 0, 0, 255]);     // Red
  setPixel(swatchBuf, 3, 0, [0, 255, 0, 255]);     // Green

  const extracted = PaletteEngine.extractPalette(swatchBuf, 256);
  assert.strictEqual(extracted.length, 4, 'Should losslessly preserve exact 4 colors');
  assert.ok(extracted.includes('#000000'), 'Should contain black');
  assert.ok(extracted.includes('#ffffff'), 'Should contain white');
  assert.ok(extracted.includes('#ff0000'), 'Should contain red');
  assert.ok(extracted.includes('#00ff00'), 'Should contain green');

  // Verify sorted monotonically by Oklab perceptual lightness (black first, white last)
  assert.strictEqual(extracted[0], '#000000', 'Black must be first due to lowest Oklab lightness');
  assert.strictEqual(extracted[extracted.length - 1], '#ffffff', 'White must be last due to highest Oklab lightness');

  // 15B: Transparent pixels are ignored
  const transBuf = createBuffer(3, 1);
  setPixel(transBuf, 0, 0, [255, 0, 0, 255]);
  setPixel(transBuf, 1, 0, [0, 255, 0, 30]);  // alpha <= 64 -> ignored
  setPixel(transBuf, 2, 0, [0, 0, 255, 0]);   // transparent -> ignored
  const transExtracted = PaletteEngine.extractPalette(transBuf, 256);
  assert.strictEqual(transExtracted.length, 1, 'Should only extract opaque colors');
  assert.strictEqual(transExtracted[0], '#ff0000');

  // 15C: High-color continuous tone buffer (> 256 colors) is automatically quantized to <= 256 colors
  const highColorBuf = createBuffer(64, 64); // 4096 pixels
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) {
      setPixel(highColorBuf, x, y, [(x * 4) % 256, (y * 4) % 256, ((x + y) * 2) % 256, 255]);
    }
  }

  const quantExtracted = PaletteEngine.extractPalette(highColorBuf, 256);
  assert.ok(quantExtracted.length > 0 && quantExtracted.length <= 256, `Extracted count (${quantExtracted.length}) must be <= 256`);
  for (const hex of quantExtracted) {
    assert.match(hex, /^#[0-9a-f]{6}$/i, `Color ${hex} must be a valid hex string`);
  }

  // Verify Oklab lightness sorting on quantized result
  for (let i = 0; i < quantExtracted.length - 1; i++) {
    const c1 = PaletteEngine.hexToRgba(quantExtracted[i]);
    const c2 = PaletteEngine.hexToRgba(quantExtracted[i + 1]);
    const l1 = PaletteEngine.rgbToOklab(c1[0], c1[1], c1[2])[0];
    const l2 = PaletteEngine.rgbToOklab(c2[0], c2[1], c2[2])[0];
    assert.ok(l1 <= l2 + 1e-5, `Palette should be sorted by lightness: ${l1} <= ${l2}`);
  }

  console.log('  Passed!');
}

const make = (w, h, f) => {
  const b = createBuffer(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) setPixel(b, x, y, f(x, y));
  return b;
};
const okDist = (c1, c2) => {
  const l1 = PaletteEngine.rgbToOklab(c1[0], c1[1], c1[2]);
  const l2 = PaletteEngine.rgbToOklab(c2[0], c2[1], c2[2]);
  return Math.sqrt(PaletteEngine.colorDistOklabSq(l1[0], l1[1], l1[2], l2[0], l2[1], l2[2]));
};

{
  console.log('Test 19: K-Means fast keeps dominant color');
  // 95% dark grey with sparse vivid noise: the grey must survive palette reduction
  const greyScene = () => make(200, 200, (x, y) => ((x * 7 + y * 13) % 20 === 0
    ? [(x * 37) % 256, (y * 91) % 256, (x * y) % 256, 255] : [40 + (x % 3), 40, 40, 255]));
  const hasGrey = (pal) => pal.some((c) => okDist(c, [41, 40, 40]) < 0.03);
  assert.ok(hasGrey(PaletteEngine.buildKMeansFastPalette(greyScene(), 4)), 'K-Means fast keeps the dominant grey');
  console.log('  Passed!');
}

// --- Palette swap ---
const colorsOf = (b) => new Set(Array.from({ length: b.width * b.height }, (_, i) => b.data.slice(i * 4, i * 4 + 3).join(',')));
const GREYS4 = ['#000000', '#555555', '#aaaaaa', '#ffffff'];
const GB = ['#9bbc0f', '#0f380f', '#8bac0f', '#306230']; // deliberately not lightness-ordered

{
  console.log('Test S1: countColors & findAnchors');
  const sprite = make(3, 2, (x) => [[10, 20, 30, 255], [200, 0, 0, 255], [0, 200, 0, 255]][x]);
  assert.strictEqual(PaletteEngine.countColors(sprite, 64), 3, 'sprite has 3 colors');
  const grad = make(256, 1, (x) => [x, x, x, 255]);
  assert.strictEqual(PaletteEngine.countColors(grad, 64), 65, 'counting stops at limit + 1');
  const faint = make(2, 1, (x) => [x * 255, 0, 0, x ? 255 : 30]);
  assert.strictEqual(PaletteEngine.countColors(faint, 64), 1, 'near-transparent pixels are not counted (matches extractPalette)');

  assert.deepStrictEqual(PaletteEngine.findAnchors(sprite, 8, 'wu').slice().sort(), ['#00c800', '#0a141e', '#c80000'].sort(), 'sprite anchors are its exact colors');
  const gradAnchors = PaletteEngine.findAnchors(grad, 8, 'wu');
  assert.ok(gradAnchors.length > 1 && gradAnchors.length <= 8, `gradient quantized to <= 8 anchors (${gradAnchors.length})`);
  console.log('  Passed!');
}

{
  console.log('Test S2: autoMap');
  const sortedGB = ['#0f380f', '#306230', '#8bac0f', '#9bbc0f'];
  assert.deepStrictEqual(PaletteEngine.autoMap(GREYS4, GB, 'rank'), sortedGB, 'rank keeps lightness order');
  assert.deepStrictEqual(PaletteEngine.autoMap(['#ffffff', '#000000'], GB, 'rank'), ['#9bbc0f', '#0f380f'], 'targets align with unsorted anchors');
  assert.deepStrictEqual(PaletteEngine.autoMap(GREYS4, ['#ffffff', '#000000'], 'rank'), ['#000000', '#000000', '#ffffff', '#ffffff'], 'more anchors than colors spread by rank');
  assert.deepStrictEqual(PaletteEngine.autoMap(['#000000', '#ffffff'], GB, 'rank'), ['#0f380f', '#9bbc0f'], 'fewer anchors take the ends');
  assert.deepStrictEqual(PaletteEngine.autoMap(['#808080'], GB, 'rank'), ['#8bac0f'], 'single anchor takes the middle entry');
  assert.deepStrictEqual(PaletteEngine.autoMap(GREYS4, ['#123456'], 'rank'), Array(4).fill('#123456'), 'single-color palette');
  assert.deepStrictEqual(PaletteEngine.autoMap(['#fe0000', '#00fe00'], ['#00ff00', '#ff0000', '#0000ff'], 'nearest'), ['#ff0000', '#00ff00'], 'nearest picks closest');
  console.log('  Passed!');
}

{
  console.log('Test S3: swap exact');
  const src = make(4, 1, (x) => [[0, 0, 0, 255], [255, 255, 255, 128], [250, 250, 250, 255], [0, 0, 0, 0]][x]);
  const before = Array.from(src.data);
  const out = PaletteEngine.swap(src, ['#000000', '#ffffff'], ['#0f380f', '#9bbc0f'], 'exact');
  assert.deepStrictEqual(Array.from(src.data), before, 'input not mutated');
  assert.deepStrictEqual(getPixel(out, 0, 0), [0x0f, 0x38, 0x0f, 255], 'anchor maps to its target');
  assert.deepStrictEqual(getPixel(out, 1, 0), [0x9b, 0xbc, 0x0f, 128], 'semi-transparent keeps its alpha');
  assert.deepStrictEqual(getPixel(out, 2, 0), [0x9b, 0xbc, 0x0f, 255], 'non-anchor color takes nearest anchor target');
  assert.deepStrictEqual(getPixel(out, 3, 0), [0, 0, 0, 0], 'transparent pixel untouched');
  assert.strictEqual(out.width, 4);
  assert.strictEqual(out.height, 1);
  console.log('  Passed!');
}

{
  console.log('Test S4: swap smooth');
  const grad = make(256, 1, (x) => [x, x, x, 255]);
  const out = PaletteEngine.swap(grad, ['#000000', '#ffffff'], ['#0f380f', '#9bbc0f'], 'smooth');
  const near = (a, b) => a.every((v, i) => Math.abs(v - b[i]) <= 1);
  assert.ok(near(getPixel(out, 0, 0).slice(0, 3), [0x0f, 0x38, 0x0f]), `anchor black lands on target (${getPixel(out, 0, 0)})`);
  assert.ok(near(getPixel(out, 255, 0).slice(0, 3), [0x9b, 0xbc, 0x0f]), `anchor white lands on target (${getPixel(out, 255, 0)})`);
  assert.notDeepStrictEqual(getPixel(out, 100, 0), getPixel(out, 104, 0), 'near colors stay distinct');
  assert.ok(colorsOf(out).size > 2, `output is not capped to the palette (${colorsOf(out).size} colors)`);
  const mid = PaletteEngine.rgbToOklab(...getPixel(out, 128, 0).slice(0, 3))[0];
  const lo = PaletteEngine.rgbToOklab(0x0f, 0x38, 0x0f)[0];
  const hi = PaletteEngine.rgbToOklab(0x9b, 0xbc, 0x0f)[0];
  assert.ok(mid > lo && mid < hi, 'mid-grey lands between the targets');
  console.log('  Passed!');
}

{
  console.log('Test S5: degenerate inputs');
  const clear = createBuffer(3, 3, [12, 34, 56, 0]);
  const anchors = PaletteEngine.findAnchors(clear, 8, 'wu');
  assert.deepStrictEqual(anchors, ['#000000'], 'transparent image falls back to one anchor');
  for (const mode of ['exact', 'smooth']) {
    const out = PaletteEngine.swap(clear, anchors, PaletteEngine.autoMap(anchors, GB, 'rank'), mode);
    assert.deepStrictEqual(Array.from(out.data), Array.from(clear.data), `${mode}: fully transparent image unchanged`);
  }
  const flat = createBuffer(2, 2, [90, 30, 160, 255]);
  const one = PaletteEngine.findAnchors(flat, 8, 'wu');
  const out = PaletteEngine.swap(flat, one, PaletteEngine.autoMap(one, GB, 'rank'), 'smooth');
  assert.ok(okDist(getPixel(out, 0, 0), PaletteEngine.hexToRgba('#8bac0f')) < 0.01, 'single-color image takes the middle target');
  console.log('  Passed!');
}

console.log('All unit tests passed successfully!');
