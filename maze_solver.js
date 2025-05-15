
// --- Configuration ---
const MODEL_PATH = "assets/onnx/atm_maze_small3.onnx";
const INPUT_NAME = "x";
const MAZE_DIM = 39;
const MAX_MAZE_INDEX = 19; // Example: if you have maze_0 to maze_19
let currentMazeIndex = -1;
const do_render_dynamics = false;

// --- MODEL PARAMETERS ---
let MODEL_SEQUENCE_LENGTH = 75;
let MODEL_ROUTE_LENGTH = 100;
let NUM_HEADS = 16;
let MODEL_NUM_NEURONS = 2048;
let ATTENTION_GRID_DIM = 10;
let ATTENTION_GRID_SIZE = ATTENTION_GRID_DIM * ATTENTION_GRID_DIM;
let NORMALIZE_INPUT_NEG1_POS1 = false;

if (MODEL_PATH == "assets/onnx/atm_maze_small3.onnx") {
    MODEL_SEQUENCE_LENGTH = 75;
    MODEL_ROUTE_LENGTH = 150;
    NUM_HEADS = 8;
    MODEL_NUM_NEURONS = 512;
    NORMALIZE_INPUT_NEG1_POS1 = true;
}

const NUM_NEURONS_TO_PLOT = 12;
const MAX_HISTORY_RUNS = 5;
console.log(`Configuration Check: NUM_NEURONS_TO_PLOT = ${NUM_NEURONS_TO_PLOT}`);
console.log(`Configuration Check: MAX_HISTORY_RUNS = ${MAX_HISTORY_RUNS}`);
console.log(`Model Params Check: SeqLength=${MODEL_SEQUENCE_LENGTH}, RouteLength=${MODEL_ROUTE_LENGTH}, Neurons=${MODEL_NUM_NEURONS}, AttnGrid=${ATTENTION_GRID_DIM}x${ATTENTION_GRID_DIM}`);

let moveMode = 'start';
let activationHistory = [];
let selectedNeuronIndices = [];
let selectedNeuronColors = [];

let latestRawPredictionData = null;
let latestPostActsData = null;
let latestAttentionData = null;

let currentAnimationValidPosition = null;
let currentAnimationStep = -1;

let initialAutoSolveDone = false;
let currentAnimationFrameId = null;
let animationStartTime = 0;
let currentAnimationFPS = 60;
const TOTAL_STEPS = MODEL_SEQUENCE_LENGTH;

const HUSL_PALETTE_8 = ['#f7718a', '#D18f18', '#97a431', '#33b167', '#36ada4', '#3aa8d0', '#a48cf4', '#f561dd'];

// --- Global State ---
let ortSession = null;
let originalImageData = null;
let preprocessedImageDataForPath = null;
let isModelReady = false;
let isImageReady = false;
let finalValidPosition = null;
let hasInitialLoadStarted = false; // Flag to prevent multiple load starts

// --- DOM Elements (get them once ready) ---
let statusDiv, canvas, ctx, solveButton, teleportButton, validOnlyCheckbox, autoSolveCheckbox, loadNewMazeButton, showPathCheckbox, showOverlayCheckbox, fpsSlider, fpsValueDisplay, toggleModeButton, mazeDemoContainer, loadingIndicator, attentionHeadsContainer, canvasHint;

// --- Helper Functions --- (Keep updateStatus, loadImage, preprocessImage, displayPreprocessedData, processOutput, drawPathOnCanvas, etc.)
// (Your existing helper functions: hexToRgbComponents, loadImage, upsampleBilinear, getNormalizedAggregatedAttention, drawAttentionOverlay, findAllValidSquares, randomizeStartEndPositions, hslToRgb, getRainbowColorForStep, getRouteForStep, preprocessImage, displayPreprocessedData, processOutput, drawPathOnCanvas, findClosestValidSquare, isTeleportTargetValid, teleportStart, teleportEnd, getViridisColor, normalizeData, drawAttentionGrid, clearAttentionGrid will go here. I'm omitting them for brevity in this diff, but they should be present in your final file.)

function updateStatus(message) {
    console.log(message);
    if (statusDiv) {
        statusDiv.textContent = message;
    }
}

// Helper to load image (no changes needed from your provided version)
function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => resolve(img);
        img.onerror = (err) => reject(`Failed to load image: ${src}. Error: ${err}`);
        img.src = src;
    });
}

// Upsample Bilinear (ensure this is present from your code)
function upsampleBilinear(sourceData, sourceW, sourceH, targetW, targetH) {
    const targetData = new Float32Array(targetW * targetH);
    const xRatio = sourceW / targetW;
    const yRatio = sourceH / targetH;
    for (let ty = 0; ty < targetH; ty++) {
        const syFloat = ty * yRatio;
        const y1 = Math.floor(syFloat);
        const y2 = Math.min(y1 + 1, sourceH - 1);
        const yWeight = syFloat - y1;
        for (let tx = 0; tx < targetW; tx++) {
            const sxFloat = tx * xRatio;
            const x1 = Math.floor(sxFloat);
            const x2 = Math.min(x1 + 1, sourceW - 1);
            const xWeight = sxFloat - x1;
            const p11_idx = y1 * sourceW + x1;
            const p12_idx = y2 * sourceW + x1;
            const p21_idx = y1 * sourceW + x2;
            const p22_idx = y2 * sourceW + x2;
            if (p11_idx < 0 || p11_idx >= sourceData.length || p12_idx < 0 || p12_idx >= sourceData.length || p21_idx < 0 || p21_idx >= sourceData.length || p22_idx < 0 || p22_idx >= sourceData.length) {
                console.warn(`Upsample index out of bounds at tx:${tx}, ty:${ty}`);
                continue;
            }
            const p11 = sourceData[p11_idx];
            const p12 = sourceData[p12_idx];
            const p21 = sourceData[p21_idx];
            const p22 = sourceData[p22_idx];
            const interpTop = p11 * (1 - xWeight) + p21 * xWeight;
            const interpBottom = p12 * (1 - xWeight) + p22 * xWeight;
            const finalValue = interpTop * (1 - yWeight) + interpBottom * yWeight;
            const targetIndex = ty * targetW + tx;
            targetData[targetIndex] = finalValue;
        }
    }
    return targetData;
}

function getNormalizedAggregatedAttention(timeStep) {
    if (!latestAttentionData) return null;
    const T = MODEL_SEQUENCE_LENGTH;
    const H = NUM_HEADS;
    const W_SRC = ATTENTION_GRID_SIZE;
    const SRC_DIM = ATTENTION_GRID_DIM;
    const TARGET_DIM = MAZE_DIM;
    const TARGET_SIZE = TARGET_DIM * TARGET_DIM;
    if (timeStep < 0 || timeStep >= T) return null;
    const aggregatedMap = new Float32Array(TARGET_SIZE).fill(0);
    const stepOffset = timeStep * H * W_SRC;
    for (let headIndex = 0; headIndex < H; headIndex++) {
        const headOffset = stepOffset + headIndex * W_SRC;
        let headData10x10;
        try { headData10x10 = latestAttentionData.subarray(headOffset, headOffset + W_SRC); }
        catch (e) { headData10x10 = new Float32Array(latestAttentionData.slice(headOffset, headOffset + W_SRC)); console.warn("Used slice fallback for head data extraction."); }
        const upsampledHeadData = upsampleBilinear(headData10x10, SRC_DIM, SRC_DIM, TARGET_DIM, TARGET_DIM);
        for (let i = 0; i < TARGET_SIZE; i++) { aggregatedMap[i] += upsampledHeadData[i]; }
    }
    for (let i = 0; i < TARGET_SIZE; i++) { aggregatedMap[i] /= H; }
    return normalizeData(aggregatedMap);
}

function hexToRgbComponents(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? { r: parseInt(result[1], 16), g: parseInt(result[2], 16), b: parseInt(result[3], 16) } : null;
}

function getViridisColor(value) {
    const viridisColors = [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]];
    const t = Math.max(0, Math.min(1, value));
    const steps = viridisColors.length - 1;
    const scaledT = t * steps;
    const index1 = Math.floor(scaledT);
    const index2 = Math.min(index1 + 1, steps);
    const interpFactor = scaledT - index1;
    const color1 = viridisColors[index1];
    const color2 = viridisColors[index2];
    const r = Math.round(color1[0] + (color2[0] - color1[0]) * interpFactor);
    const g = Math.round(color1[1] + (color2[1] - color1[1]) * interpFactor);
    const b = Math.round(color1[2] + (color2[2] - color1[2]) * interpFactor);
    return [r, g, b];
}

function drawAttentionOverlay(normalizedAggregatedMap) {
    if (!normalizedAggregatedMap || !ctx) return;
    const targetW = MAZE_DIM; const targetH = MAZE_DIM;
    const currentImageData = ctx.getImageData(0, 0, targetW, targetH);
    const currentPixels = currentImageData.data;
    const overlayImageData = ctx.createImageData(targetW, targetH);
    const overlayPixels = overlayImageData.data;
    for (let i = 0; i < normalizedAggregatedMap.length; i++) {
        const attentionValue = normalizedAggregatedMap[i];
        const [vr, vg, vb] = getViridisColor(attentionValue);
        const pixelIndex = i * 4;
        const originalR = currentPixels[pixelIndex]; const originalG = currentPixels[pixelIndex + 1]; const originalB = currentPixels[pixelIndex + 2];
        let blendR = originalR * (1 - attentionValue * 0.6) + vr * attentionValue * 1.1;
        let blendG = originalG * (1 - attentionValue * 0.6) + vg * attentionValue * 1.1;
        let blendB = originalB * (1 - attentionValue * 0.6) + vb * attentionValue * 1.1;
        blendR = Math.max(0, Math.min(255, Math.round(blendR)));
        blendG = Math.max(0, Math.min(255, Math.round(blendG)));
        blendB = Math.max(0, Math.min(255, Math.round(blendB)));
        overlayPixels[pixelIndex] = blendR; overlayPixels[pixelIndex + 1] = blendG; overlayPixels[pixelIndex + 2] = blendB; overlayPixels[pixelIndex + 3] = 255;
    }
    ctx.putImageData(overlayImageData, 0, 0);
}

function findAllValidSquares(imageData) {
    console.log("[findAllValidSquares] Searching for white pixels...");
    const { width, height, data } = imageData;
    const validSquares = []; let whitePixelCount = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4;
            if (data[i] === 255 && data[i+1] === 255 && data[i+2] === 255) {
                validSquares.push({ x, y }); whitePixelCount++;
            }
        }
    }
    if (whitePixelCount < 2) { console.error(`[findAllValidSquares] Insufficient white squares found (${whitePixelCount}). Need at least 2 for start/end.`); }
    else { console.log(`[findAllValidSquares] Found ${whitePixelCount} potential start/end squares.`); }
    return validSquares;
}

function randomizeStartEndPositions(imageData) {
    console.log("[randomizeStartEndPositions] Attempting to randomize start/end...");
    const { width, height, data } = imageData;
    const validSquares = findAllValidSquares(imageData);
    if (validSquares.length < 2) {
        console.error("[randomizeStartEndPositions] Cannot randomize: Less than 2 valid (white) squares found.");
        updateStatus("⚠️ Warning: Could not randomize start/end. Using default."); return false;
    }
    let originalStartPos = null; let originalEndPos = null;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4;
            if (!originalStartPos && data[i] === 255 && data[i + 1] === 0 && data[i + 2] === 0) { originalStartPos = { x, y }; }
            else if (!originalEndPos && data[i] === 0 && data[i + 1] === 255 && data[i + 2] === 0) { originalEndPos = { x, y }; }
            if (originalStartPos && originalEndPos) break;
        }
        if (originalStartPos && originalEndPos) break;
    }
    console.log("[randomizeStartEndPositions] Original Start:", originalStartPos, "Original End:", originalEndPos);
    if (originalStartPos) { const idx = (originalStartPos.y * width + originalStartPos.x) * 4; data[idx] = 255; data[idx + 1] = 255; data[idx + 2] = 255; data[idx + 3] = 255; }
    if (originalEndPos) { const idx = (originalEndPos.y * width + originalEndPos.x) * 4; data[idx] = 255; data[idx + 1] = 255; data[idx + 2] = 255; data[idx + 3] = 255; }
    let startIndex = Math.floor(Math.random() * validSquares.length);
    let endIndex = Math.floor(Math.random() * validSquares.length);
    while (endIndex === startIndex) { endIndex = Math.floor(Math.random() * validSquares.length); }
    const newStartPos = validSquares[startIndex]; const newEndPos = validSquares[endIndex];
    console.log(`[randomizeStartEndPositions] New Start chosen: (${newStartPos.x}, ${newStartPos.y})`);
    console.log(`[randomizeStartEndPositions] New End chosen: (${newEndPos.x}, ${newEndPos.y})`);
    const newStartIndex = (newStartPos.y * width + newStartPos.x) * 4; data[newStartIndex] = 255; data[newStartIndex + 1] = 0; data[newStartIndex + 2] = 0; data[newStartIndex + 3] = 255;
    const newEndIndex = (newEndPos.y * width + newEndPos.x) * 4; data[newEndIndex] = 0; data[newEndIndex + 1] = 255; data[newEndIndex + 2] = 0; data[newEndIndex + 3] = 255;
    console.log("[randomizeStartEndPositions] Successfully randomized start/end positions in ImageData."); return true;
}

function hslToRgb(h, s, l){
    var r, g, b;
    if(s == 0){ r = g = b = l; }
    else {
        var hue2rgb = function hue2rgb(p, q, t){
            if(t < 0) t += 1; if(t > 1) t -= 1;
            if(t < 1/6) return p + (q - p) * 6 * t;
            if(t < 1/2) return q;
            if(t < 2/3) return p + (q - p) * (2/3 - t) * 6;
            return p;
        }
        var q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        var p = 2 * l - q;
        r = hue2rgb(p, q, h + 1/3); g = hue2rgb(p, q, h); b = hue2rgb(p, q, h - 1/3);
    }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function getRainbowColorForStep(stepIndex, totalSteps) {
    if (totalSteps <= 1) return [0, 0, 255];
    const hue = (stepIndex / (totalSteps - 1)) * 0.9;
    const saturation = 0.9; const lightness = 0.5;
    return hslToRgb(hue, saturation, lightness);
}

function getRouteForStep(timeStep, rawPredictionData) {
    const numClasses = 5; const numCells = MODEL_ROUTE_LENGTH; const sequenceLength = TOTAL_STEPS;
    const elementsPerSequence = numCells * numClasses;
    if (!rawPredictionData || rawPredictionData.length !== elementsPerSequence * sequenceLength) {
        console.error(`getRouteForStep: Invalid rawPredictionData provided. Length: ${rawPredictionData?.length}`);
        return new Array(numCells).fill(4);
    }
    if (timeStep < 0 || timeStep >= sequenceLength) { console.error(`getRouteForStep: Invalid timeStep ${timeStep}.`); return new Array(numCells).fill(4); }
    const route = [];
    for (let i = 0; i < numCells; i++) {
        let maxLogit = -Infinity; let predictedClass = -1;
        for (let k = 0; k < numClasses; k++) {
            const flatIndex = (i * numClasses + k) * sequenceLength + timeStep;
            if (flatIndex >= rawPredictionData.length) { console.error(`getRouteForStep: Calculated flatIndex ${flatIndex} OOB`); predictedClass = 4; break; }
            const logit = rawPredictionData[flatIndex];
            if (logit > maxLogit) { maxLogit = logit; predictedClass = k; }
        }
        route.push(predictedClass === -1 ? 4 : predictedClass);
    }
    return route;
}

function preprocessImage(imageData) {
    const { data, width, height } = imageData;
    console.log(`Preprocessing image. Target range: ${NORMALIZE_INPUT_NEG1_POS1 ? '[-1, 1]' : '[0, 1]'}`);
    if (width !== MAZE_DIM || height !== MAZE_DIM) { console.error(`ImageData dimensions (${width}x${height}) mismatch MAZE_DIM (${MAZE_DIM})`); }
    const inputData = new Float32Array(1 * 3 * MAZE_DIM * MAZE_DIM);
    let bluePixelFound = false; let bluePixelsProcessed = 0;
    for (let y = 0; y < MAZE_DIM; y++) {
        for (let x = 0; x < MAZE_DIM; x++) {
            const index = (y * MAZE_DIM + x) * 4;
            const oR_u8 = data[index]; const oG_u8 = data[index + 1]; const oB_u8 = data[index + 2];
            let r_0_1 = oR_u8 / 255.0; let g_0_1 = oG_u8 / 255.0; let b_0_1 = oB_u8 / 255.0;
            if (oR_u8 === 0 && oG_u8 === 0 && oB_u8 === 255) {
                if (!bluePixelFound) { console.log(`Found blue pixel. Overwriting with white.`); bluePixelFound = true; }
                bluePixelsProcessed++; r_0_1 = 1.0; g_0_1 = 1.0; b_0_1 = 1.0;
            }
            let store_r, store_g, store_b;
            if (NORMALIZE_INPUT_NEG1_POS1) {
                store_r = (r_0_1 * 2.0) - 1.0; store_g = (g_0_1 * 2.0) - 1.0; store_b = (b_0_1 * 2.0) - 1.0;
            } else { store_r = r_0_1; store_g = g_0_1; store_b = b_0_1; }
            inputData[0 * (MAZE_DIM * MAZE_DIM) + y * MAZE_DIM + x] = store_r;
            inputData[1 * (MAZE_DIM * MAZE_DIM) + y * MAZE_DIM + x] = store_g;
            inputData[2 * (MAZE_DIM * MAZE_DIM) + y * MAZE_DIM + x] = store_b;
        }
    }
    if (!bluePixelFound) { console.warn("Preprocessing Warning: No RGB(0,0,255) pixels found."); }
    else { console.log(`Preprocessing Info: Processed ${bluePixelsProcessed} blue pixel(s).`); }
    return inputData;
}

function displayPreprocessedData(inputData) {
    if (!ctx) return null;
    console.log(`Displaying preprocessed data (converting from ${NORMALIZE_INPUT_NEG1_POS1 ? '[-1, 1]' : '[0, 1]'} range)...`);
    const C = 3; const H = MAZE_DIM; const W = MAZE_DIM; const channelSize = H * W;
    const displayImageData = ctx.createImageData(W, H); const displayData = displayImageData.data;
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const idx = y * W + x;
            const r_f = inputData[0*channelSize+idx]; const g_f = inputData[1*channelSize+idx]; const b_f = inputData[2*channelSize+idx];
            let r_0_1, g_0_1, b_0_1;
            if (NORMALIZE_INPUT_NEG1_POS1) { r_0_1 = (r_f+1)/2; g_0_1 = (g_f+1)/2; b_0_1 = (b_f+1)/2; }
            else { r_0_1 = r_f; g_0_1 = g_f; b_0_1 = b_f; }
            const r_u8 = Math.max(0,Math.min(255,Math.round(r_0_1*255))); const g_u8 = Math.max(0,Math.min(255,Math.round(g_0_1*255))); const b_u8 = Math.max(0,Math.min(255,Math.round(b_0_1*255)));
            const displayIdx = (y * W + x) * 4;
            displayData[displayIdx] = r_u8; displayData[displayIdx + 1] = g_u8; displayData[displayIdx + 2] = b_u8; displayData[displayIdx + 3] = 255;
        }
    }
    ctx.putImageData(displayImageData, 0, 0);
    console.log("Displayed preprocessed image on canvas.");
    return displayImageData;
}

function processOutput(outputTensor) { /* Your existing function, ensure it uses MODEL_ROUTE_LENGTH */
    const expectedShape = [1, MODEL_ROUTE_LENGTH * 5, MODEL_SEQUENCE_LENGTH];
    const outputData = outputTensor.data;
    const numClasses = 5; const numCells = MODEL_ROUTE_LENGTH; const sequenceLength = expectedShape[2];
    if (outputTensor.dims.length !== expectedShape.length || !outputTensor.dims.every((dim, i) => dim === expectedShape[i])) {
        console.warn(`processOutput: Unexpected output tensor shape: ${outputTensor.dims}. Expected ${expectedShape}.`);
    }
    const predictions = []; const lastStepIndex = sequenceLength - 1;
    for (let i = 0; i < numCells; i++) {
        let maxLogit = -Infinity; let predictedClass = -1;
        for (let k = 0; k < numClasses; k++) {
            const flatIndex = (i * numClasses + k) * sequenceLength + lastStepIndex;
            if (flatIndex >= outputData.length) { console.error(`Index ${flatIndex} out of bounds.`); continue; }
            const logit = outputData[flatIndex];
            if (logit > maxLogit) { maxLogit = logit; predictedClass = k; }
        }
        predictions.push(predictedClass === -1 ? 4 : predictedClass);
    }
    return predictions;
}

function drawPathOnCanvas(baseImageData, route, drawValidOnly, drawPath) { /* Your existing function */
    if (!ctx) return;
    const { width, height } = baseImageData;
    const pathImageData = new ImageData(new Uint8ClampedArray(baseImageData.data), width, height);
    const data = pathImageData.data;
    let startPos = null;
    if (drawPath) {
        for (let y = 0; y < height; y++) { for (let x = 0; x < width; x++) { const index = (y * width + x) * 4; if (data[index] === 255 && data[index + 1] === 0 && data[index + 2] === 0) { startPos = { x, y }; break; } } if (startPos) break; }
        if (!startPos) { console.error("Could not find start pixel for path drawing."); drawPath = false; }
    }
    let currentPos = startPos ? { ...startPos } : null;
    let stepColorIndex = 0; let ignoredOrWarnedMoves = 0;
    if (drawPath && currentPos) {
        console.log("Drawing path...");
        for (const [stepIndex, step] of route.entries()) {
            let potentialX = currentPos.x; let potentialY = currentPos.y; let moveType = "InvalidValue";
            if (step === 0) { potentialY -= 1; moveType="Up"; } else if (step === 1) { potentialY += 1; moveType="Down"; } else if (step === 2) { potentialX -= 1; moveType="Left"; } else if (step === 3) { potentialX += 1; moveType="Right"; } else if (step === 4) { moveType="Stay"; }
            if (step === 4 || moveType === "InvalidValue") { continue; }
            let isInBounds = (potentialX >= 0 && potentialX < width && potentialY >= 0 && potentialY < height);
            let isWall = false; if (isInBounds) { const pixelIndex = (potentialY * width + potentialX) * 4; isWall = (data[pixelIndex] === 0 && data[pixelIndex + 1] === 0 && data[pixelIndex + 2] === 0); }
            let isValidStrictMove = isInBounds && !isWall;
            if (drawValidOnly) {
                if (isValidStrictMove) { currentPos = { x: potentialX, y: potentialY }; const pixelIndex = (currentPos.y * width + currentPos.x) * 4; const isStart = (data[pixelIndex] === 255 && data[pixelIndex+1] === 0 && data[pixelIndex+2] === 0); const isEnd = (data[pixelIndex] === 0 && data[pixelIndex+1] === 255 && data[pixelIndex+2] === 0); if (!isStart && !isEnd) { const [stepR, stepG, stepB] = getRainbowColorForStep(stepColorIndex, route.length); data[pixelIndex] = Math.round(data[pixelIndex]*0.5+stepR*0.5); data[pixelIndex+1]=Math.round(data[pixelIndex+1]*0.5+stepG*0.5); data[pixelIndex+2]=Math.round(data[pixelIndex+2]*0.5+stepB*0.5); data[pixelIndex+3]=255; stepColorIndex++; } }
                else { ignoredOrWarnedMoves++; }
            } else {
                if (isInBounds) { currentPos = { x: potentialX, y: potentialY }; const pixelIndex = (currentPos.y * width + currentPos.x) * 4; const isStart = (data[pixelIndex] === 255 && data[pixelIndex+1] === 0 && data[pixelIndex+2] === 0); const isEnd = (data[pixelIndex] === 0 && data[pixelIndex+1] === 255 && data[pixelIndex+2] === 0); if (!isStart && !isEnd) { const [stepR, stepG, stepB] = getRainbowColorForStep(stepColorIndex, route.length); data[pixelIndex]=Math.round(data[pixelIndex]*0.5+stepR*0.5); data[pixelIndex+1]=Math.round(data[pixelIndex+1]*0.5+stepG*0.5); data[pixelIndex+2]=Math.round(data[pixelIndex+2]*0.5+stepB*0.5); data[pixelIndex+3]=255; stepColorIndex++; } }
                else { currentPos = { x: potentialX, y: potentialY }; ignoredOrWarnedMoves++; }
            }
        }
    } else { console.log("Skipping path drawing (drawPath is false or startPos not found)."); }
    ctx.putImageData(pathImageData, 0, 0);
}


function findClosestValidSquare(startX, startY, imageData) { /* Your existing function */
    const { width, height } = imageData;
    if (isTeleportTargetValid(startX, startY, imageData)) { return { x: startX, y: startY }; }
    const queue = [{ x: startX, y: startY, dist: 0 }]; const visited = new Set([`${startX},${startY}`]);
    let head = 0; let foundAtDistance = -1; let candidates = [];
    while (head < queue.length) {
        const { x, y, dist } = queue[head++];
        if (foundAtDistance !== -1 && dist >= foundAtDistance) { continue; }
        for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
                if (dx === 0 && dy === 0) continue;
                const checkX = x + dx; const checkY = y + dy; const coordKey = `${checkX},${checkY}`;
                if (checkX < 0 || checkX >= width || checkY < 0 || checkY >= height || visited.has(coordKey)) { continue; }
                if (isTeleportTargetValid(checkX, checkY, imageData)) {
                    const candidatePos = { x: checkX, y: checkY }; const currentCandidateDistance = dist + 1;
                    if (foundAtDistance === -1 || currentCandidateDistance === foundAtDistance) { if (foundAtDistance === -1) { console.log(`findClosestValidSquare: Found first valid candidate(s) at distance ${currentCandidateDistance}`); foundAtDistance = currentCandidateDistance; } if (!visited.has(coordKey)){ candidates.push(candidatePos); visited.add(coordKey); } }
                    else if (currentCandidateDistance < foundAtDistance) { foundAtDistance = currentCandidateDistance; candidates = [candidatePos]; visited.add(coordKey); }
                } else { if ((foundAtDistance === -1 || dist + 1 < foundAtDistance) && !visited.has(coordKey)) { visited.add(coordKey); queue.push({ x: checkX, y: checkY, dist: dist + 1 }); } }
            }
        }
    }
    if (candidates.length === 0) { console.warn(`findClosestValidSquare: No valid (white) square found reachable from (${startX},${startY}).`); return null; }
    if (candidates.length === 1) { console.log(`findClosestValidSquare: Clicked (${startX},${startY}), found single closest valid at (${candidates[0].x},${candidates[0].y})`); return candidates[0]; }
    console.log(`findClosestValidSquare: Found ${candidates.length} candidates at distance ${foundAtDistance}. Euclidean tie-breaker...`);
    let bestCandidate = null; let minDistanceSq = Infinity;
    for (const candidate of candidates) { const dx_ = candidate.x - startX; const dy_ = candidate.y - startY; const distSq = dx_*dx_ + dy_*dy_; if (distSq < minDistanceSq) { minDistanceSq = distSq; bestCandidate = candidate; } }
    console.log(`findClosestValidSquare: Clicked (${startX},${startY}), returning best candidate (${bestCandidate.x},${bestCandidate.y})`);
    return bestCandidate;
}

function isTeleportTargetValid(x, y, imageData) { /* Your existing function */
    const { width, height, data } = imageData;
    if (x < 0 || x >= width || y < 0 || y >= height) { return false; }
    const i = (y * width + x) * 4;
    return data[i] === 255 && data[i+1] === 255 && data[i+2] === 255;
}

function teleportStart(newStartX, newStartY) { /* Your existing function */
    console.log(`Attempting teleport start to (${newStartX}, ${newStartY})`);
    if (!originalImageData || !preprocessedImageDataForPath) { console.error("teleportStart Error: Image data not ready."); updateStatus("❌ Error: Image data missing for teleport."); return false; }
    if (newStartX < 0 || newStartX >= MAZE_DIM || newStartY < 0 || newStartY >= MAZE_DIM) { console.error(`teleportStart Error: Target OOB.`); updateStatus("❌ Error: Teleport target out of bounds."); return false; }
    try {
        const { width, height } = originalImageData; const ppData = preprocessedImageDataForPath.data;
        const targetPixelIndex = (newStartY * width + newStartX) * 4;
        if (ppData[targetPixelIndex] === 0 && ppData[targetPixelIndex + 1] === 255 && ppData[targetPixelIndex + 2] === 0) { console.warn(`teleportStart: Target is goal.`); updateStatus("⚠️ Cannot place start on goal."); return false; }
        let currentStartPos = null;
        for(let y=0; y<height; y++) { for(let x=0; x<width; x++) { const i = (y*width+x)*4; if(ppData[i]===255 && ppData[i+1]===0 && ppData[i+2]===0){ currentStartPos={x,y}; break; } } if(currentStartPos)break; }
        if (!currentStartPos) { console.error("teleportStart Error: Could not find current start."); updateStatus("❌ Error: Cannot find current start."); return false; }
        if (currentStartPos.x === newStartX && currentStartPos.y === newStartY) { console.log("teleportStart: Target same as current."); return true; }
        let modOriginalData = new Uint8ClampedArray(originalImageData.data); let modPreprocessedData = new Uint8ClampedArray(preprocessedImageDataForPath.data);
        const oldStartIndex = (currentStartPos.y * width + currentStartPos.x) * 4;
        modOriginalData[oldStartIndex]=255; modOriginalData[oldStartIndex+1]=255; modOriginalData[oldStartIndex+2]=255; modPreprocessedData[oldStartIndex]=255; modPreprocessedData[oldStartIndex+1]=255; modPreprocessedData[oldStartIndex+2]=255;
        const newStartIndex_ = (newStartY * width + newStartX) * 4;
        modOriginalData[newStartIndex_]=255; modOriginalData[newStartIndex_+1]=0; modOriginalData[newStartIndex_+2]=0; modPreprocessedData[newStartIndex_]=255; modPreprocessedData[newStartIndex_+1]=0; modPreprocessedData[newStartIndex_+2]=0;
        modOriginalData[oldStartIndex+3]=255; modPreprocessedData[oldStartIndex+3]=255; modOriginalData[newStartIndex_+3]=255; modPreprocessedData[newStartIndex_+3]=255;
        originalImageData = new ImageData(modOriginalData, width, height); preprocessedImageDataForPath = new ImageData(modPreprocessedData, width, height);
        if (ctx) ctx.putImageData(preprocessedImageDataForPath, 0, 0);
        console.log(`teleportStart: Success to (${newStartX}, ${newStartY})`); return true;
    } catch(error) { console.error("Error during teleportStart:", error); updateStatus("❌ Teleport failed (internal error)."); return false; }
}

function teleportEnd(newEndX, newEndY) { /* Your existing function */
    console.log(`Attempting to move end point to (${newEndX}, ${newEndY})`);
    if (!originalImageData || !preprocessedImageDataForPath) { console.error("teleportEnd Error: Image data not ready."); updateStatus("❌ Error: Image data missing."); return false; }
    if (newEndX < 0 || newEndX >= MAZE_DIM || newEndY < 0 || newEndY >= MAZE_DIM) { console.error("teleportEnd Error: Target OOB."); updateStatus("❌ Error: Move goal target OOB."); return false; }
    try {
        const { width, height } = originalImageData; const ppData = preprocessedImageDataForPath.data;
        const targetPixelIndex = (newEndY * width + newEndX) * 4;
        if (ppData[targetPixelIndex] === 255 && ppData[targetPixelIndex + 1] === 0 && ppData[targetPixelIndex + 2] === 0) { console.warn("teleportEnd: Target is start."); updateStatus("⚠️ Cannot place goal on start."); return false; }
        let currentEndPos = null;
        for(let y=0; y<height; y++) { for(let x=0; x<width; x++) { const i=(y*width+x)*4; if(ppData[i]===0&&ppData[i+1]===255&&ppData[i+2]===0){ currentEndPos={x,y}; break; } } if(currentEndPos)break; }
        if (!currentEndPos) { console.error("teleportEnd Error: Could not find current end."); updateStatus("❌ Error: Cannot find current goal."); return false; }
        if (currentEndPos.x === newEndX && currentEndPos.y === newEndY) { console.log("teleportEnd: Target same as current."); return true; }
        let modOriginalData = new Uint8ClampedArray(originalImageData.data); let modPreprocessedData = new Uint8ClampedArray(preprocessedImageDataForPath.data);
        const oldEndIndex = (currentEndPos.y * width + currentEndPos.x) * 4;
        modOriginalData[oldEndIndex]=255; modOriginalData[oldEndIndex+1]=255; modOriginalData[oldEndIndex+2]=255; modPreprocessedData[oldEndIndex]=255; modPreprocessedData[oldEndIndex+1]=255; modPreprocessedData[oldEndIndex+2]=255;
        const newEndIndex_ = (newEndY * width + newEndX) * 4;
        modOriginalData[newEndIndex_]=0; modOriginalData[newEndIndex_+1]=255; modOriginalData[newEndIndex_+2]=0; modPreprocessedData[newEndIndex_]=0; modPreprocessedData[newEndIndex_+1]=255; modPreprocessedData[newEndIndex_+2]=0;
        modOriginalData[oldEndIndex+3]=255; modPreprocessedData[oldEndIndex+3]=255; modOriginalData[newEndIndex_+3]=255; modPreprocessedData[newEndIndex_+3]=255;
        originalImageData = new ImageData(modOriginalData, width, height); preprocessedImageDataForPath = new ImageData(modPreprocessedData, width, height);
        if (ctx) ctx.putImageData(preprocessedImageDataForPath, 0, 0);
        console.log(`teleportEnd: Success to (${newEndX}, ${newEndY})`); return true;
    } catch(error) { console.error("Error during teleportEnd:", error); updateStatus("❌ Move goal failed (internal error)."); return false; }
}

function normalizeData(data) { /* Your existing function */
    if (!data || data.length === 0) { return new Float32Array(0); }
    let minVal = data[0]; let maxVal = data[0];
    for (let i = 1; i < data.length; i++) { if (data[i] < minVal) minVal = data[i]; if (data[i] > maxVal) maxVal = data[i]; }
    const range = maxVal - minVal; const normalized = new Float32Array(data.length);
    if (range === 0) { normalized.fill(0); } else { for (let i = 0; i < data.length; i++) { normalized[i] = (data[i] - minVal) / range; } }
    return normalized;
}

function drawAttentionGrid(timeStep) { /* Your existing function, ensure attentionHeadsContainer items are used */
    if (!latestAttentionData || !attentionHeadsContainer) return;
    const T = MODEL_SEQUENCE_LENGTH; const H = NUM_HEADS; const W_GRID = ATTENTION_GRID_SIZE;
    if (timeStep < 0 || timeStep >= T) { console.warn(`drawAttentionGrid: Invalid timeStep ${timeStep}.`); clearAttentionGrid(); return; }
    const stepOffset = timeStep * H * W_GRID;
    for (let headIndex = 0; headIndex < H; headIndex++) {
        const headOffset = stepOffset + headIndex * W_GRID;
        const headData = latestAttentionData.slice(headOffset, headOffset + W_GRID);
        const normalizedHeadData = normalizeData(headData);
        const headCanvas = document.getElementById(`attentionHead_${headIndex}`);
        if (!headCanvas) continue;
        const headCtx = headCanvas.getContext('2d'); if (!headCtx) continue;
        const headImageData = headCtx.createImageData(ATTENTION_GRID_DIM, ATTENTION_GRID_DIM); const pixels = headImageData.data;
        for (let i = 0; i < W_GRID; i++) {
            const normalizedValue = normalizedHeadData[i]; const [r, g, b] = getViridisColor(normalizedValue);
            const pixelIndex = i * 4; pixels[pixelIndex]=r; pixels[pixelIndex+1]=g; pixels[pixelIndex+2]=b; pixels[pixelIndex+3]=255;
        }
        headCtx.putImageData(headImageData, 0, 0);
    }
}

function clearAttentionGrid() { /* Your existing function */
    if (!attentionHeadsContainer) return;
    const [r, g, b] = getViridisColor(0); const zeroColorStyle = `rgb(${r}, ${g}, ${b})`;
    for (let headIndex = 0; headIndex < NUM_HEADS; headIndex++) {
        const headCanvas = document.getElementById(`attentionHead_${headIndex}`); if (!headCanvas) continue;
        const headCtx = headCanvas.getContext('2d'); if (!headCtx) continue;
        headCtx.fillStyle = zeroColorStyle; headCtx.fillRect(0, 0, headCanvas.width, headCanvas.height);
    }
    console.log("Cleared attention grid.");
}

// --- Initialization and Solving Logic ---

async function loadModel() {
    updateStatus("Loading AI Model..."); // Status update
    try {
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
        ortSession = await ort.InferenceSession.create(MODEL_PATH, { executionProviders: ['wasm'] });
        console.log("ONNX session created successfully.");

        if (selectedNeuronIndices.length === 0) {
            const totalNeurons = MODEL_NUM_NEURONS;
            const allIndices = Array.from(Array(totalNeurons).keys());
            for (let i = allIndices.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1));[allIndices[i], allIndices[j]] = [allIndices[j], allIndices[i]]; }
            let numToPlot = NUM_NEURONS_TO_PLOT;
            if (numToPlot > totalNeurons) { console.warn(`NUM_NEURONS_TO_PLOT too high, clamping.`); numToPlot = totalNeurons; }
            selectedNeuronIndices = allIndices.slice(0, numToPlot).sort((a, b) => a - b);
            console.log(`Selected ${numToPlot} neuron indices:`, selectedNeuronIndices);
            selectedNeuronColors = selectedNeuronIndices.map(() => HUSL_PALETTE_8[Math.floor(Math.random() * HUSL_PALETTE_8.length)]);
            console.log(`Generated ${selectedNeuronColors.length} colors.`);
        }
        updateStatus("AI Model ready."); // Status update
        isModelReady = true;
    } catch (error) {
        console.error("Error loading ONNX model:", error);
        updateStatus(`❌ Error loading model: ${error.message}`);
        isModelReady = false;
    }
    checkResourcesReady();
}

async function loadImageAndDisplay(imagePath) {
    updateStatus(`Loading maze: ${imagePath.split('/').pop()}...`); // Status update
    try {
        const img = await loadImage(imagePath);
        if (img.width !== MAZE_DIM || img.height !== MAZE_DIM) {
            throw new Error(`Image dimensions (${img.width}x${img.height}) incorrect. Expected ${MAZE_DIM}x${MAZE_DIM}.`);
        }
        if (ctx) ctx.drawImage(img, 0, 0, MAZE_DIM, MAZE_DIM); else { throw new Error("Canvas context not available.");}
        originalImageData = ctx.getImageData(0, 0, MAZE_DIM, MAZE_DIM);
        console.log("Original image data captured.");
        randomizeStartEndPositions(originalImageData);
        const inputData = preprocessImage(originalImageData);
        console.log("Image data preprocessed (after randomization).");
        preprocessedImageDataForPath = displayPreprocessedData(inputData);
        updateStatus(`Maze ${currentMazeIndex} loaded.`); // Status update (already good)
        isImageReady = true;
    } catch (error) {
        console.error(`Failed to load/process image: ${imagePath}`, error);
        updateStatus(`❌ Error loading image: ${imagePath.split('/').pop()}. ${error.message}.`);
        isImageReady = false;
    }
    checkResourcesReady();
}

async function loadRandomMaze() {
    console.log("loadRandomMaze called.");
    if (solveButton) solveButton.disabled = true;
    if (teleportButton) teleportButton.disabled = true;
    if (loadNewMazeButton) loadNewMazeButton.disabled = true;
    updateStatus("Selecting and loading new maze...");

    if (currentAnimationFrameId) { cancelAnimationFrame(currentAnimationFrameId); currentAnimationFrameId = null; console.log("Cancelled previous animation."); }
    isImageReady = false; originalImageData = null; preprocessedImageDataForPath = null;
    latestRawPredictionData = null; latestPostActsData = null; latestAttentionData = null;
    finalValidPosition = null; currentAnimationValidPosition = null; animationStartTime = 0; currentAnimationStep = -1;
    initialAutoSolveDone = false; activationHistory = [];
    if (do_render_dynamics) renderNeuralDynamics(-1); // Assuming this function exists
    clearAttentionGrid();

    let chosenIndex;
    if (MAX_MAZE_INDEX <= 0) { chosenIndex = 0; }
    else {
        chosenIndex = Math.floor(Math.random() * (MAX_MAZE_INDEX + 1));
        if (chosenIndex === currentMazeIndex) { chosenIndex = (currentMazeIndex + 1) % (MAX_MAZE_INDEX + 1); }
    }
    currentMazeIndex = chosenIndex;
    const imagePath = `assets/mazes/maze_${currentMazeIndex}.png`;
    console.log(`Selected maze index: ${currentMazeIndex}, Path: ${imagePath}`);
    await loadImageAndDisplay(imagePath); // This will call checkResourcesReady
    if (isModelReady && loadNewMazeButton) { loadNewMazeButton.disabled = false; }
}


function checkResourcesReady() {
    if (!mazeDemoContainer || !loadingIndicator || !statusDiv || !solveButton || !teleportButton || !loadNewMazeButton || !fpsSlider) {
        // Elements not ready yet, defer UI update or log error
        console.warn("checkResourcesReady: Some DOM elements not available yet.");
        return;
    }

    if (isImageReady && isModelReady) {
        mazeDemoContainer.classList.remove('loading-active');
        mazeDemoContainer.classList.add('loading-complete');
        loadingIndicator.style.display = 'none';
        statusDiv.style.display = 'flex'; // Ensure visible for "Ready"
        fpsSlider.disabled = false;

        solveButton.disabled = false;
        solveButton.textContent = 'Run';
        teleportButton.disabled = true; // Until first solve

        // Explicitly show the content with their correct display types
        if (canvas) canvas.style.display = 'block'; // Or its original if different
        if (canvasHint) canvasHint.style.display = 'block'; // Or its original
        if (attentionHeadsContainer) attentionHeadsContainer.style.display = 'grid';
        const controlsElement = document.getElementById('controls'); // Get it if not global
        if (controlsElement) controlsElement.style.display = 'flex';

        if (autoSolveCheckbox.checked && !initialAutoSolveDone) {
            updateStatus(`Maze ${currentMazeIndex} ready. Auto-solving...`);
            initialAutoSolveDone = true;
            setTimeout(handleSolveClick, 100);
        } else {
            updateStatus(`Maze ${currentMazeIndex} ready.`);
        }
        console.log(`Initialization/Load complete for Maze ${currentMazeIndex}. Ready to solve.`);

    } else if (!isModelReady && !isImageReady && hasInitialLoadStarted) {
        // Both failed after an attempt
        mazeDemoContainer.classList.remove('loading-active');
        loadingIndicator.style.display = 'none';
        if (!statusDiv.textContent.includes("❌ Error")) { updateStatus("❌ Failed to load resources."); }
        statusDiv.style.display = 'flex';
        solveButton.disabled = true; solveButton.textContent = 'Load Error';
        teleportButton.disabled = true; loadNewMazeButton.disabled = true; fpsSlider.disabled = false; // Allow FPS change even on error

    } else if (hasInitialLoadStarted) { // Still loading one or the other, or one failed
        if (!mazeDemoContainer.classList.contains('loading-active')) {
            mazeDemoContainer.classList.add('loading-active');
        }
        if (loadingIndicator.style.display === 'none') loadingIndicator.style.display = 'flex';
        statusDiv.style.display = 'flex';
        if (!statusDiv.textContent.includes("❌ Error")) {
            if (isModelReady && !isImageReady) { updateStatus("Model ready. Loading maze..."); }
            else if (!isModelReady && isImageReady) { updateStatus(`Loading model... Please wait.`); }
            else { updateStatus("Loading resources...");} // Generic if both still pending
        }
        solveButton.disabled = true; teleportButton.disabled = true; fpsSlider.disabled = true;
    } else {
         // Initial state before hasInitialLoadStarted is true (handled by initializeApp)
    }

    // Load New Maze button state
    if (isModelReady && MAX_MAZE_INDEX >= 0) { // Allow even if only one maze
        // Disable if solving or animating
        const isSolvingOrAnimating = (solveButton.disabled && (solveButton.textContent.includes('Solving') || solveButton.textContent.includes('Processing'))) || (statusDiv.style.display !== 'none' && statusDiv.textContent.includes('Animating'));
        loadNewMazeButton.disabled = isSolvingOrAnimating;
    } else {
        loadNewMazeButton.disabled = true;
    }
}


async function handleSolveClick() {
    console.log("handleSolveClick started...");
    if (!isModelReady || !isImageReady || !ortSession || !originalImageData || !solveButton || !teleportButton || !fpsSlider || !loadNewMazeButton) {
        updateStatus("❌ Error: Resources not ready. Please refresh."); console.error("Solve clicked but resources not ready.");
        if(solveButton) solveButton.disabled = !isModelReady || !isImageReady; if(teleportButton) teleportButton.disabled = true; return;
    }
    if (currentAnimationFrameId) { cancelAnimationFrame(currentAnimationFrameId); currentAnimationFrameId = null; console.log("Cancelled previous animation."); }
    animationStartTime = 0; finalValidPosition = null; currentAnimationValidPosition = null; currentAnimationStep = -1;
    solveButton.disabled = true; teleportButton.disabled = true; fpsSlider.disabled = true; loadNewMazeButton.disabled = true;
    updateStatus("Processing...");
    try {
        updateStatus("Preprocessing image for model...");
        const inputData = preprocessImage(originalImageData); // Use current originalImageData
        preprocessedImageDataForPath = displayPreprocessedData(inputData); // Display it
        updateStatus("Preparing tensor...");
        const inputTensor = new ort.Tensor('float32', inputData, [1, 3, MAZE_DIM, MAZE_DIM]);
        updateStatus(`Running inference...`);
        const startTime = performance.now();
        const feeds = { [INPUT_NAME]: inputTensor };
        const outputNamesToFetch = [ortSession.outputNames[0], 'post-acts', 'attn'];
        const availableOutputNames = outputNamesToFetch.filter(name => ortSession.outputNames.includes(name));
        console.log("Requesting outputs:", availableOutputNames);
        if (!availableOutputNames.includes(ortSession.outputNames[0])) { throw new Error(`Primary prediction output missing.`); }

        const results = await ortSession.run(feeds, availableOutputNames);
        const endTime = performance.now();
        console.log(`Inference took: ${(endTime - startTime).toFixed(2)} ms`);
        updateStatus(`✅ Inference complete. Processing results...`);

        const predictionTensor = results[ortSession.outputNames[0]];
        const postActsTensor = results['post-acts'];
        const attentionTensor = results['attn'];

        if (!predictionTensor || !predictionTensor.data) { throw new Error(`Prediction output invalid.`); }
        latestRawPredictionData = new Float32Array(predictionTensor.data);
        if (postActsTensor && postActsTensor.data) { latestPostActsData = new Float32Array(postActsTensor.data); } else { latestPostActsData = null; }
        if (attentionTensor && attentionTensor.data) { latestAttentionData = new Float32Array(attentionTensor.data); } else { latestAttentionData = null; }

        if (latestPostActsData && selectedNeuronIndices.length > 0 && do_render_dynamics) { /* ... your dynamics processing ... */ }

        const finalRoute = getRouteForStep(TOTAL_STEPS - 1, latestRawPredictionData);
        if (!finalRoute || finalRoute.length !== MODEL_ROUTE_LENGTH) { throw new Error(`getRouteForStep invalid for final pos calc.`); }
        finalValidPosition = calculateFinalValidPosition(preprocessedImageDataForPath, finalRoute);
        console.log("Final valid position for Teleport (after solve):", finalValidPosition);

        updateStatus("Starting path animation...");
        if (fpsSlider) fpsSlider.disabled = false; // <<< --- ADD THIS LINE TO ENABLE THE SLIDER
        if (solveButton) { // Ensure solveButton is checked for existence before accessing properties
            solveButton.disabled = false; // Re-enable solve button for re-runs or skip
            solveButton.textContent = 'Run';
        }
        // skipAnimationButton.disabled = false; // If you add it back
        animationStartTime = 0;
        currentAnimationFrameId = requestAnimationFrame(animatePath);
        // Teleport button state handled by animatePath/skip
    } catch (error) {
        updateStatus(`❌ Error during solve: ${error.message}`);
        console.error("--- ERROR CAUGHT in handleSolveClick ---", error.message, error.stack, error);
        solveButton.disabled = false; solveButton.textContent = 'Error - Retry?';
        teleportButton.disabled = true; fpsSlider.disabled = false;
        if (isModelReady && MAX_MAZE_INDEX >=0) loadNewMazeButton.disabled = false; else loadNewMazeButton.disabled = true;
        latestRawPredictionData = null; latestPostActsData = null; latestAttentionData = null;
        if (currentAnimationFrameId) { cancelAnimationFrame(currentAnimationFrameId); currentAnimationFrameId = null;}
    }
}


function handleSkipAnimation() { /* Your existing function */
    console.log("Skip animation clicked.");
    if (!currentAnimationFrameId || !latestRawPredictionData || !preprocessedImageDataForPath) { return; }
    cancelAnimationFrame(currentAnimationFrameId); currentAnimationFrameId = null;
    animationStartTime = 0; currentAnimationStep = -1;
    updateStatus('✅ Animation skipped. Final state shown.');
    if (latestAttentionData) { drawAttentionGrid(TOTAL_STEPS - 1); }
    const finalRoute = getRouteForStep(TOTAL_STEPS - 1, latestRawPredictionData);
    const drawValidOnly = validOnlyCheckbox.checked; const shouldDrawPath = showPathCheckbox.checked;
    drawPathOnCanvas(preprocessedImageDataForPath, finalRoute, drawValidOnly, shouldDrawPath);
    solveButton.disabled = false; solveButton.textContent = 'Run';
    teleportButton.disabled = !finalValidPosition;
    if (isModelReady && MAX_MAZE_INDEX >=0) loadNewMazeButton.disabled = false; else loadNewMazeButton.disabled = true;
    fpsSlider.disabled = false;
}


function animatePath(timestamp) { /* Your existing function */
    if (!latestRawPredictionData || !preprocessedImageDataForPath || !solveButton || !teleportButton || !fpsSlider || !loadNewMazeButton) {
        console.error("animatePath: Missing critical data or DOM elements.");
        updateStatus("❌ Animation error."); if(currentAnimationFrameId) cancelAnimationFrame(currentAnimationFrameId);
        currentAnimationFrameId=null; animationStartTime=0; currentAnimationStep=-1;
        if(solveButton) {solveButton.disabled = false; solveButton.textContent = 'Run';} if(teleportButton) teleportButton.disabled = true; return;
    }
    const msPerFrame = 1000 / currentAnimationFPS;
    if (animationStartTime === 0) { animationStartTime = timestamp; }
    const elapsedTime = timestamp - animationStartTime;
    let step = Math.floor(elapsedTime / msPerFrame);
    step = Math.min(step, TOTAL_STEPS - 1);

    if (step !== currentAnimationStep) {
        currentAnimationStep = step;
        updateStatus(`Animating step ${step + 1}/${TOTAL_STEPS}...`); // User-friendly step count
        const routeForStep = getRouteForStep(step, latestRawPredictionData);
        currentAnimationValidPosition = calculateFinalValidPosition(preprocessedImageDataForPath, routeForStep);
        teleportButton.disabled = !currentAnimationValidPosition;
        const drawValidOnly = validOnlyCheckbox.checked; const shouldDrawPath = showPathCheckbox.checked; const shouldShowOverlay = showOverlayCheckbox.checked;
        drawPathOnCanvas(preprocessedImageDataForPath, routeForStep, drawValidOnly, shouldDrawPath); // Ensure 4th arg
        if (latestPostActsData && do_render_dynamics) { renderNeuralDynamics(step); } // Assuming renderNeuralDynamics exists
        if (latestAttentionData) {
            drawAttentionGrid(step);
            if (shouldShowOverlay) { const normAggMap = getNormalizedAggregatedAttention(step); drawAttentionOverlay(normAggMap); }
        }
    }
    if (step < TOTAL_STEPS - 1 && currentAnimationFrameId) { currentAnimationFrameId = requestAnimationFrame(animatePath); }
    else if (currentAnimationFrameId) {
        currentAnimationFrameId = null; console.log("Path animation complete.");
        updateStatus(`✅ Animation finished. Maze ${currentMazeIndex}`);
        animationStartTime = 0;
        if (latestAttentionData) { drawAttentionGrid(TOTAL_STEPS - 1); }
        const finalRoute = getRouteForStep(TOTAL_STEPS - 1, latestRawPredictionData);
        const drawValidOnly = validOnlyCheckbox.checked; const shouldDrawPath = showPathCheckbox.checked; const shouldShowOverlay = showOverlayCheckbox.checked;
        drawPathOnCanvas(preprocessedImageDataForPath, finalRoute, drawValidOnly, shouldDrawPath); // Ensure 4th arg
        if (latestAttentionData && shouldShowOverlay) { const normAggMap = getNormalizedAggregatedAttention(TOTAL_STEPS -1); drawAttentionOverlay(normAggMap); }


        solveButton.disabled = false; solveButton.textContent = 'Run';
        teleportButton.disabled = !finalValidPosition; // Use the overall finalValidPosition
        if (isModelReady && MAX_MAZE_INDEX >=0) loadNewMazeButton.disabled = false; else loadNewMazeButton.disabled = true;
        fpsSlider.disabled = false;
        console.log(`Animation end. Final Teleport Target:`, finalValidPosition);
    } else { console.log("Animation loop ending (likely cancelled)."); }
}


function calculateFinalValidPosition(baseImageData, route) { /* Your existing function */
    const { width, height, data } = baseImageData;
    let startPos = null;
    for (let y = 0; y < height; y++) { for (let x = 0; x < width; x++) { const i = (y*width+x)*4; if (data[i]===255&&data[i+1]===0&&data[i+2]===0) { startPos={x,y}; break; } } if (startPos) break; }
    if (!startPos) { console.error("calculateFinalValidPosition: Start pixel not found."); return null; }
    let currentPos = { ...startPos };
    for (const [stepIndex, step] of route.entries()) {
        let potentialX = currentPos.x; let potentialY = currentPos.y; let moveType = "InvalidValue";
        if (step===0) {potentialY-=1; moveType="Up";} else if (step===1) {potentialY+=1; moveType="Down";} else if (step===2) {potentialX-=1; moveType="Left";} else if (step===3) {potentialX+=1; moveType="Right";} else if (step===4) {moveType="Stay";}
        if (step===4 || moveType === "InvalidValue") continue;
        let isValidMove = false; let isPotentialGoal = false;
        if (potentialX>=0 && potentialX<width && potentialY>=0 && potentialY<height) {
            const pixelIndex = (potentialY*width+potentialX)*4;
            const targetR=data[pixelIndex]; const targetG=data[pixelIndex+1]; const targetB=data[pixelIndex+2];
            const isWall = (targetR===0&&targetG===0&&targetB===0);
            isPotentialGoal = (targetR===0&&targetG===255&&targetB===0);
            if (!isWall) isValidMove = true;
        }
        if (isValidMove) { if (isPotentialGoal) { console.log(`[calcFinalPos] Step ${stepIndex} would reach GOAL. Target is previous pos.`); return currentPos; } else { currentPos = {x: potentialX, y: potentialY}; } }
    }
    console.log("[calcFinalPos] Route finished. Returning final valid position:", currentPos);
    return currentPos;
}


function handleTeleportClick() { /* Your existing function, ensure DOM elements are checked if not global */
    if(!solveButton || !teleportButton) { console.error("TeleportClick: Buttons not ready"); return;}
    let targetPosition = null; let wasAnimating = (currentAnimationFrameId !== null);
    if (wasAnimating && currentAnimationValidPosition) { targetPosition = currentAnimationValidPosition; console.log("Teleport during animation. Target:", targetPosition); }
    else if (!wasAnimating && finalValidPosition) { targetPosition = finalValidPosition; console.log("Teleport after animation. Target:", targetPosition); }
    else { console.error("Teleport clicked, but no valid target."); updateStatus("⚠️ No valid position for teleport."); return; }
    if (wasAnimating) { cancelAnimationFrame(currentAnimationFrameId); currentAnimationFrameId = null; animationStartTime=0; currentAnimationStep=-1; console.log("Stopped animation for teleport."); solveButton.textContent = 'Run'; solveButton.disabled = false; }
    teleportButton.disabled = true; updateStatus(`Teleporting start to (${targetPosition.x}, ${targetPosition.y})...`);
    if (teleportStart(targetPosition.x, targetPosition.y)) {
        finalValidPosition = null; currentAnimationValidPosition = null;
        if (autoSolveCheckbox.checked) { updateStatus(`Teleported. Auto-solving...`); setTimeout(handleSolveClick, 50); }
        else { updateStatus(`Teleported. Ready to solve.`); solveButton.disabled = false; solveButton.textContent = 'Run'; }
    } else { updateStatus("❌ Teleport start failed."); solveButton.disabled = false; teleportButton.disabled = true; if(wasAnimating) solveButton.textContent = 'Run'; else solveButton.textContent = 'Run'; }
}


function handleCanvasClick(event) { /* Your existing function, ensure DOM elements are checked */
    if (!isImageReady || !isModelReady || !preprocessedImageDataForPath || !solveButton || solveButton.disabled) return;
    let wasAnimating = (currentAnimationFrameId !== null);
    if (wasAnimating) { cancelAnimationFrame(currentAnimationFrameId); currentAnimationFrameId=null; animationStartTime=0; currentAnimationStep=-1; console.log(`Stopped animation for canvas tap (${moveMode}).`); solveButton.textContent='Run'; teleportButton.disabled=true; solveButton.disabled=false; }
    const rect = canvas.getBoundingClientRect(); const scaleX = canvas.width/rect.width; const scaleY = canvas.height/rect.height;
    const clickXrel = event.clientX-rect.left; const clickYrel = event.clientY-rect.top;
    const canvasX = clickXrel*scaleX; const canvasY = clickYrel*scaleY;
    const mazeX = Math.floor(canvasX); const mazeY = Math.floor(canvasY);
    if (mazeX<0||mazeX>=MAZE_DIM||mazeY<0||mazeY>=MAZE_DIM) return;
    const ppData = preprocessedImageDataForPath.data; const pixelIndex=(mazeY*MAZE_DIM+mazeX)*4;
    const r=ppData[pixelIndex]; const g=ppData[pixelIndex+1]; const b=ppData[pixelIndex+2];
    let targetX = -1, targetY = -1;
    if (r===255&&g===255&&b===255) { targetX=mazeX; targetY=mazeY; }
    else { updateStatus(`Searching for valid square for ${moveMode}...`); const closestPos=findClosestValidSquare(mazeX,mazeY,preprocessedImageDataForPath); if(closestPos){targetX=closestPos.x;targetY=closestPos.y;}else{updateStatus(`Cannot move ${moveMode}: No valid square.`);if(wasAnimating)solveButton.disabled=false;return;}}
    if (targetX !== -1) {
        let success = false; const targetCoords = `(${targetX}, ${targetY})`;
        if (moveMode==='start') { updateStatus(`Teleporting start to ${targetCoords}...`); success=teleportStart(targetX,targetY); }
        else { updateStatus(`Moving goal to ${targetCoords}...`); success=teleportEnd(targetX,targetY); }
        if (success) { finalValidPosition=null; currentAnimationValidPosition=null; teleportButton.disabled=true; if(autoSolveCheckbox.checked){updateStatus(`Moved ${moveMode}. Auto-solving...`);setTimeout(handleSolveClick,50);}else{updateStatus(`Moved ${moveMode}. Ready to solve.`);solveButton.disabled=false;} }
        else { if(wasAnimating){solveButton.disabled=false;solveButton.textContent='Run';}else{solveButton.disabled=false;} }
    }
}

function handleCanvasRightClick(event) { /* Your existing function, ensure DOM elements are checked */
    event.preventDefault();
    if (!isImageReady || !isModelReady || !preprocessedImageDataForPath || !solveButton || solveButton.disabled) return;
    let wasAnimating = (currentAnimationFrameId !== null);
    if (wasAnimating) { cancelAnimationFrame(currentAnimationFrameId); currentAnimationFrameId=null; animationStartTime=0; currentAnimationStep=-1; console.log(`Stopped animation for right-click.`); solveButton.textContent='Run'; teleportButton.disabled=true; solveButton.disabled=false; }
    const rect = canvas.getBoundingClientRect(); const scaleX = canvas.width/rect.width; const scaleY = canvas.height/rect.height;
    const clickXrel = event.clientX-rect.left; const clickYrel = event.clientY-rect.top;
    const canvasX = clickXrel*scaleX; const canvasY = clickYrel*scaleY;
    const mazeX = Math.floor(canvasX); const mazeY = Math.floor(canvasY);
    if (mazeX<0||mazeX>=MAZE_DIM||mazeY<0||mazeY>=MAZE_DIM) return;
    const ppData = preprocessedImageDataForPath.data; const pixelIndex=(mazeY*MAZE_DIM+mazeX)*4;
    const r=ppData[pixelIndex]; const g=ppData[pixelIndex+1]; const b=ppData[pixelIndex+2];
    let targetX = -1, targetY = -1;
    if (r===255&&g===255&&b===255) { targetX=mazeX; targetY=mazeY; }
    else { const intendedTarget=(moveMode==='start')?'goal':'start'; updateStatus(`Searching valid for ${intendedTarget}...`); const closestPos=findClosestValidSquare(mazeX,mazeY,preprocessedImageDataForPath); if(closestPos){targetX=closestPos.x;targetY=closestPos.y;}else{updateStatus(`Cannot move ${intendedTarget}: No valid square.`);if(wasAnimating)solveButton.disabled=false;return;}}
    if (targetX !== -1) {
        let success = false; const targetCoords = `(${targetX}, ${targetY})`; let actionTaken = '';
        if (moveMode==='start') { actionTaken='goal'; updateStatus(`Moving goal to ${targetCoords}...`); success=teleportEnd(targetX,targetY); }
        else { actionTaken='start'; updateStatus(`Teleporting start to ${targetCoords}...`); success=teleportStart(targetX,targetY); }
        if (success) { finalValidPosition=null; currentAnimationValidPosition=null; teleportButton.disabled=true; if(autoSolveCheckbox.checked){updateStatus(`Moved ${actionTaken}. Auto-solving...`);setTimeout(handleSolveClick,50);}else{updateStatus(`Moved ${actionTaken}. Ready to solve.`);solveButton.disabled=false;} }
        else { if(wasAnimating){solveButton.disabled=false;solveButton.textContent='Run';}else{solveButton.disabled=false;} }
    }
}

function renderNeuralDynamics(currentTimeStep) { /* Your existing function, ensure it's guarded by do_render_dynamics */
    if (!do_render_dynamics) return;
    // ... your implementation ...
    console.log(`renderNeuralDynamics called for step: ${currentTimeStep} (if enabled)`);
}


// --- App Initialization ---
function initializeApp() {
    console.log("initializeApp: Setting up UI elements and base listeners.");

    // Assign DOM elements to global vars
    statusDiv = document.getElementById('status');
    canvas = document.getElementById('mazeCanvas');
    if (canvas) ctx = canvas.getContext('2d', { willReadFrequently: true });
    solveButton = document.getElementById('solveButton');
    teleportButton = document.getElementById('teleportButton');
    validOnlyCheckbox = document.getElementById('validOnlyCheckbox');
    autoSolveCheckbox = document.getElementById('autoSolveCheckbox');
    loadNewMazeButton = document.getElementById('loadNewMazeButton');
    showPathCheckbox = document.getElementById('showPathCheckbox');
    showOverlayCheckbox = document.getElementById('showOverlayCheckbox');
    fpsSlider = document.getElementById('fpsSlider');
    fpsValueDisplay = document.getElementById('fpsValueDisplay');
    toggleModeButton = document.getElementById('toggleModeButton');
    mazeDemoContainer = document.getElementById('maze-demo');
    loadingIndicator = document.getElementById('loadingIndicator');
    attentionHeadsContainer = document.getElementById('attentionHeadsContainer');
    canvasHint = document.getElementById('canvasHint');


    if (mazeDemoContainer) mazeDemoContainer.classList.add('loading-active');
    if (statusDiv) {
        statusDiv.textContent = "Loading CTM model, please wait."; // MODIFIED TEXT
        statusDiv.style.display = 'flex';
    }
    if (loadingIndicator) loadingIndicator.style.display = 'flex';

    if (solveButton) solveButton.disabled = true;
    if (teleportButton) teleportButton.disabled = true;
    if (loadNewMazeButton) loadNewMazeButton.disabled = true;
    if (fpsSlider) fpsSlider.disabled = true; // Disabled until loading attempt finishes
    if (solveButton) solveButton.textContent = 'Run';

    if (canvas) canvas.style.display = 'none';
    if (canvasHint) canvasHint.style.display = 'none'; // Make sure canvasHint is defined
    if (attentionHeadsContainer) attentionHeadsContainer.style.display = 'none'; // Make sure attentionHeadsContainer is defined
    const controlsElement = document.getElementById('controls'); // Get it if not global
    if (controlsElement) controlsElement.style.display = 'none';

    if (attentionHeadsContainer) {
        attentionHeadsContainer.innerHTML = '';
        if (typeof NUM_HEADS === 'number' && NUM_HEADS > 0 && typeof ATTENTION_GRID_DIM === 'number') {
            for (let i = 0; i < NUM_HEADS; i++) {
                const attnCanvas = document.createElement('canvas');
                attnCanvas.id = `attentionHead_${i}`;
                attnCanvas.width = ATTENTION_GRID_DIM; attnCanvas.height = ATTENTION_GRID_DIM;
                attentionHeadsContainer.appendChild(attnCanvas);
            }
            console.log("Created attention canvases.");
        } else { console.warn("Cannot create attention canvases: Params invalid."); }
    } else { console.warn("Attention container not found."); }

    if (fpsSlider && fpsValueDisplay) {
        fpsValueDisplay.textContent = fpsSlider.value;
        currentAnimationFPS = parseInt(fpsSlider.value, 10);
        console.log(`Initial FPS set from slider: ${currentAnimationFPS}`); // For debugging
    }

    // Event Listeners
    if (solveButton) solveButton.addEventListener('click', handleSolveClick);
    if (teleportButton) teleportButton.addEventListener('click', handleTeleportClick);
    if (canvas) {
        canvas.addEventListener('click', handleCanvasClick);
        canvas.addEventListener('contextmenu', handleCanvasRightClick);
    }
    if (loadNewMazeButton) loadNewMazeButton.addEventListener('click', loadRandomMaze);
    if (fpsSlider) fpsSlider.addEventListener('input', (event) => {
                const newFPS = parseInt(event.target.value, 10);
                if (fpsValueDisplay) fpsValueDisplay.textContent = newFPS;
                currentAnimationFPS = newFPS; // This updates the FPS for the *next* animation run
        
                // ---- NEW LOGIC TO STOP ANIMATION ----
                if (currentAnimationFrameId) {
                    console.log("FPS slider changed during animation. Stopping animation.");
                    cancelAnimationFrame(currentAnimationFrameId);
                    currentAnimationFrameId = null;
        
                    animationStartTime = 0; // Reset animation timer
                    // currentAnimationStep is not reset here, so the visuals remain at the interrupted step.
        
                    updateStatus('Animation stopped. Adjust FPS and Run again.');
        
                    // Restore button states
                    if (solveButton) {
                        solveButton.disabled = false;
                        solveButton.textContent = 'Run';
                    }
                    if (teleportButton) {
                        // Disable teleport as the animation was interrupted and a clear "final" state wasn't reached.
                        teleportButton.disabled = true;
                    }
                    if (loadNewMazeButton) {
                        // Enable 'New' button if model is ready and mazes are available
                        loadNewMazeButton.disabled = !(isModelReady && MAX_MAZE_INDEX >= 0);
                    }
                    // fpsSlider remains enabled by default and by the change in handleSolveClick
                    if (fpsSlider) {
                        fpsSlider.disabled = false;
                    }
                }
                // ---- END NEW LOGIC ----
            });
        
    if (toggleModeButton) {
        toggleModeButton.classList.add('move-button-red-theme');
        toggleModeButton.addEventListener('click', () => {
            if (moveMode === 'start') {
                moveMode = 'goal'; toggleModeButton.textContent = 'Move:🟩';
                toggleModeButton.classList.remove('move-button-red-theme'); toggleModeButton.classList.add('move-button-green-theme');
            } else {
                moveMode = 'start'; toggleModeButton.textContent = 'Move:🟥';
                toggleModeButton.classList.remove('move-button-green-theme'); toggleModeButton.classList.add('move-button-red-theme');
            }
            console.log("Switched move mode to:", moveMode);
        });
    }
    if (canvasHint && canvas) {
        const hideHint = () => { if(canvasHint) canvasHint.style.display = 'none'; };
        canvas.addEventListener('click', hideHint, { once: true });
        canvas.addEventListener('contextmenu', hideHint, { once: true });
    }

    console.log("initializeApp: Basic UI ready. Triggering background load.");
    if (!hasInitialLoadStarted) {
        startInitialBackgroundLoad();
    }
}

async function startInitialBackgroundLoad() {
    if (hasInitialLoadStarted) {
        console.log("Initial load already initiated."); return;
    }
    hasInitialLoadStarted = true;
    console.log("startInitialBackgroundLoad: Starting model and maze loading.");

    if (loadingIndicator) loadingIndicator.style.display = 'flex';
    if (statusDiv) {
        statusDiv.textContent = "Loading CTM model, please wait."; // MODIFIED TEXT (or keep "Loading resources...")
        statusDiv.style.display = 'flex';
    }

    try {
        await Promise.allSettled([
            loadModel(),
            loadRandomMaze()
        ]);
        console.log("Initial resource loading process (Promise.allSettled) finished.");
        // checkResourcesReady is called internally by loadModel/loadRandomMaze
    } catch (error) {
        console.error("Error during initial resource loading sequence:", error);
        updateStatus(`❌ Error initializing demo: ${error.message}`);
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        if (statusDiv) statusDiv.style.display = 'flex';
        if (solveButton) { solveButton.disabled = true; solveButton.textContent = 'Load Error'; }
    } finally {
        // FPS slider enabled once loading attempt is over, regardless of outcome
        if (fpsSlider) fpsSlider.disabled = false;
        console.log("Loading attempt complete, FPS slider enabled.");
    }
}

// Start the application initialization when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', initializeApp);