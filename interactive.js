const verticalLinePlugin = {
    id: 'verticalLine',
    afterDatasetsDraw(chart, args, options) {
        const currentStep = chart.options.currentStep;
        if (currentStep === undefined) return;
        const ctx = chart.ctx;
        const xScale = chart.scales.x;
        let xPixel;
        if (chart.config.type === 'line') {
            xPixel = xScale.getPixelForValue(currentStep);
        }
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(xPixel, chart.scales.y.top);
        ctx.lineTo(xPixel, chart.scales.y.bottom);
        ctx.lineWidth = options.lineWidth || 1;
        ctx.strokeStyle = options.lineColor || 'black';
        ctx.stroke();
        ctx.restore();
    }
};

function initializeCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const attentionCanvas = document.getElementById("attentionCanvas");
    const attCtx = attentionCanvas.getContext("2d");
    attCtx.clearRect(0, 0, attentionCanvas.width, attentionCanvas.height);
}

function getEventPosition(event) {
    const rect = canvas.getBoundingClientRect();
    let clientX, clientY;
    if (event.touches && event.touches.length > 0) {
        clientX = event.touches[0].clientX;
        clientY = event.touches[0].clientY;
    } else {
        clientX = event.clientX;
        clientY = event.clientY;
    }
    return {
        x: clientX - rect.left,
        y: clientY - rect.top
    };
}

function startDrawing(event) {
    event.preventDefault();
    drawing = true;
    const pos = getEventPosition(event);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
}

function draw(event) {
    event.preventDefault();
    if (!drawing) return;
    const pos = getEventPosition(event);
    ctx.lineTo(pos.x, pos.y);
    ctx.strokeStyle = "white";
    ctx.lineWidth = 15;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.stroke();
}

function stopDrawing(event) {
    event.preventDefault();
    drawing = false;
}

function resetPredictionChart() {
    if (predictionsChart) {
        const numLabels = predictionsChart.data.labels.length;
        predictionsChart.data.datasets[0].data = Array(numLabels).fill(0);
        predictionsChart.options.currentStep = 0;
        predictionsChart.update();
      }
}

function resetTraceCharts() {
    hiddenChartInstances.forEach(chart => {
        const numLabels = chart.data.labels.length;
        chart.data.datasets[0].data = Array(numLabels).fill(0);
        chart.options.currentStep = 0;
        chart.update();
      });
}

function createEventListeners() {
    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDrawing);
    canvas.addEventListener("mouseleave", stopDrawing);
    canvas.addEventListener("touchstart", startDrawing);
    canvas.addEventListener("touchmove", draw);
    canvas.addEventListener("touchend", stopDrawing);
    canvas.addEventListener("touchcancel", stopDrawing);
    document.getElementById("clearBtn").addEventListener("click", clearAll);
}

function clearAll() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    const attentionCanvas = document.getElementById("attentionCanvas");
    const attCtx = attentionCanvas.getContext("2d");
    attCtx.clearRect(0, 0, attentionCanvas.width, attentionCanvas.height);
  
    resetCharts();
    resetIntervals();  
  }

function resetIntervals() {
    if (predictionInterval) clearInterval(predictionInterval);
    if (activationInterval) clearInterval(activationInterval);
    if (attentionInterval) clearInterval(attentionInterval);
}

function resetCharts() {
    resetPredictionChart();
    resetTraceCharts();
}

function downsample(sourceCanvas, targetWidth, targetHeight) {
    const tempCanvas = document.createElement("canvas");
    const tempCtx = tempCanvas.getContext("2d");
    tempCanvas.width = Math.max(sourceCanvas.width / 2, targetWidth);
    tempCanvas.height = Math.max(sourceCanvas.height / 2, targetHeight);
    tempCtx.drawImage(sourceCanvas, 0, 0, tempCanvas.width, tempCanvas.height);
    const finalCanvas = document.createElement("canvas");
    finalCanvas.width = targetWidth;
    finalCanvas.height = targetHeight;
    const finalCtx = finalCanvas.getContext("2d");
    finalCtx.drawImage(tempCanvas, 0, 0, targetWidth, targetHeight);
    return finalCanvas;
}


async function runInference(animationUpdateInterval) {
    try {
        resetIntervals();

        const session = await sessionPromise;

        const downsampledCanvas = downsample(canvas, 28, 28);
        const downsampledCtx = downsampledCanvas.getContext("2d");

        const imageData = downsampledCtx.getImageData(0, 0, 28, 28);
        const { data } = imageData;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(downsampledCanvas, 0, 0, canvas.width, canvas.height);

        const inputData = new Float32Array(28 * 28);
        for (let i = 0; i < data.length; i += 4) {
            inputData[i / 4] = data[i] / 255.0;
        }

        const inputTensor = new ort.Tensor("float32", inputData, [1, 1, 28, 28]);
        const results = await session.run({ x: inputTensor });

        const predictions = results.predictions;
        const post_activations = results.post_activations_tracking;
        const attention_weights = results.attention_tracking;

        updateHiddenCharts(post_activations);
        animatePredictions(predictions, animationUpdateInterval);
        animateActivations(post_activations.dims[0], animationUpdateInterval);
        animateAttention(attention_weights, attention_weights.dims[0], animationUpdateInterval);
    } 
    catch (error) {
        console.error("Error during inference:", error);
    }
}

function computeProbabilitiesForTimeStep(predictions, t) {
    const [B, C, T] = predictions.dims;
    const logits = [];
    for (let c = 0; c < C; c++) {
        const index = c * T + t;
        logits.push(predictions.data[index]);
    }
    const maxLogit = Math.max(...logits);
    const exps = logits.map(x => Math.exp(x - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sumExps);
}

function animatePredictions(predictions, animationUpdateInterval) {
    const [B, C, T] = predictions.dims;
    const initialProbs = computeProbabilitiesForTimeStep(predictions, 0);
    
    predictionsChart.data.datasets[0].data = initialProbs;
    predictionsChart.options.currentStep = 0;
    predictionsChart.options.totalSteps = T;
    predictionsChart.update();

    let currentStep = 0;
    predictionInterval = setInterval(() => {
        currentStep++;
        if (currentStep >= T) {
            clearInterval(predictionInterval);
            return;
        }
        const newProbs = computeProbabilitiesForTimeStep(predictions, currentStep);
        predictionsChart.data.datasets[0].data = newProbs;
        predictionsChart.options.currentStep = currentStep;
        predictionsChart.update();
    }, animationUpdateInterval);
}

function animateActivations(totalSteps, animationUpdateInterval) {
    let currentStep = 0;
    activationInterval = setInterval(() => {
        currentStep++;
        if (currentStep >= totalSteps) {
            clearInterval(activationInterval);
            return;
        }
        hiddenChartInstances.forEach(chart => {
            chart.options.currentStep = currentStep;
            chart.update();
        });
    }, animationUpdateInterval);
}

function animateAttention(attentionWeights, totalSteps, animationUpdateInterval) {
    let currentStep = 0;
    drawAttentionMapForTimeStep(attentionWeights, currentStep);
    attentionInterval = setInterval(() => {
        currentStep++;
        if (currentStep >= totalSteps) {
        clearInterval(attentionInterval);
        return;
        }
        drawAttentionMapForTimeStep(attentionWeights, currentStep);
    }, animationUpdateInterval);
}

function viridisColor(value) {
    const stops = [
        { val: 0.0, color: [68, 1, 84] },
        { val: 0.25, color: [59, 82, 139] },
        { val: 0.5, color: [33, 145, 140] },
        { val: 0.75, color: [94, 201, 98] },
        { val: 1.0, color: [253, 231, 37] }
    ];
    for (let i = 0; i < stops.length - 1; i++) {
        if (value >= stops[i].val && value <= stops[i + 1].val) {
        const t = (value - stops[i].val) / (stops[i + 1].val - stops[i].val);
        const r = Math.round(stops[i].color[0] + t * (stops[i + 1].color[0] - stops[i].color[0]));
        const g = Math.round(stops[i].color[1] + t * (stops[i + 1].color[1] - stops[i].color[1]));
        const b = Math.round(stops[i].color[2] + t * (stops[i + 1].color[2] - stops[i].color[2]));
        return [r, g, b];
        }
    }
    return [0, 0, 0];
}

function drawAttentionMapForTimeStep(attentionWeights, t) {
    const offset = t * 784;
    let maxVal = 0;
    for (let i = 0; i < 784; i++) {
        maxVal = Math.max(maxVal, attentionWeights.data[offset + i]);
    }
    if (maxVal === 0) maxVal = 1;

    const attOffscreen = document.createElement("canvas");
    attOffscreen.width = 28;
    attOffscreen.height = 28;
    const offCtx = attOffscreen.getContext("2d");

    const imgData = offCtx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
        const normalizedValue = attentionWeights.data[offset + i] / maxVal;
        const [r, g, b] = viridisColor(normalizedValue);
        const alpha = Math.min(255, Math.floor(normalizedValue * 255 * 0.7));
        const pixelIndex = i * 4;
        imgData.data[pixelIndex] = r;
        imgData.data[pixelIndex + 1] = g;
        imgData.data[pixelIndex + 2] = b;
        imgData.data[pixelIndex + 3] = alpha;
    }
    offCtx.putImageData(imgData, 0, 0);

    const attentionCanvas = document.getElementById("attentionCanvas");
    const attentionCtx = attentionCanvas.getContext("2d");
    attentionCtx.clearRect(0, 0, attentionCanvas.width, attentionCanvas.height);
    attentionCtx.drawImage(attOffscreen, 0, 0, attentionCanvas.width, attentionCanvas.height);
}

function initializeTraceCharts(T, H) {
    for (let neuron = 0; neuron < H; neuron++) {
      const dataArray = new Array(T).fill(0);
      const labels = Array.from({ length: T }, (_, i) => i);
      const ctxHidden = document.getElementById(`traceChart${neuron}`).getContext("2d");
      const hiddenChart = new Chart(ctxHidden, {
        type: "line",
        data: {
          labels: labels,
          datasets: [{
            label: `Neuron ${neuron} Post-activation`,
            data: dataArray,
            borderColor: neuron % 2 === 0 ? "red" : "blue",
            backgroundColor: "transparent",
            borderWidth: 2,
            tension: 0.1,
            pointRadius: 0
          }]
        },
        options: {
          responsive: false,
          currentStep: 0,
          scales: {
            x: { display: true, ticks: { display: false }, grid: { display: false } },
            y: { display: true, ticks: { display: false }, grid: { display: false } }
          },
          plugins: {
            legend: { display: false },
            verticalLine: { lineWidth: 1, lineColor: 'black' }
          }
        }
      });
      hiddenChartInstances.push(hiddenChart);
    }
  }
  

function updateHiddenCharts(postActivations) {
    const [T, , H] = postActivations.dims;
    hiddenChartInstances.forEach((chart, neuron) => {
        const dataArray = [];
        for (let t = 0; t < T; t++) {
            const index = t * H + neuron;
            dataArray.push(postActivations.data[index]);
        }
        chart.data.datasets[0].data = dataArray;
        chart.options.currentStep = 0;
        chart.update();
    });
}

function initializePredictionChart(T, C) {
    const ctxPred = document.getElementById("predictionChart").getContext("2d");

    const initialProbs = [];
    for (let c = 0; c < C; c++) {
        initialProbs.push(0);
    }

    predictionsChart = new Chart(ctxPred, {
        type: "bar",
        data: {
            labels: Array.from({ length: C }, (_, i) => i),
            datasets: [{
                label: "Probability",
                data: initialProbs,
                backgroundColor: "black",
                borderColor: "black",
                borderWidth: 1
            }]
        },
        options: {
            responsive: false,
            scales: { y: { beginAtZero: true, max: 1 } },
            currentStep: 0,
            totalSteps: T,
            plugins: {
                legend: { display: false },
                verticalLine: { lineWidth: 1, lineColor: 'black' }
            }
        }
    });
}

function getCanvasCoordinates(canvas, event) {
    // Get the bounding rectangle of the canvas
    const rect = canvas.getBoundingClientRect();
  
    // Calculate scale factors for x and y
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
  
    // For touch events, you might use event.touches[0] instead of event
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
  
    return { x, y };
  }

// MAIN
Chart.register(verticalLinePlugin);
Chart.defaults.animation = false;

const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");
const hiddenChartInstances = [];

const T = 30
const H = 64
const C = 10

const animationUpdateIntervalInMs = 30

let predictionsChart = null;
let predictionInterval = null;
let activationInterval = null;
let attentionInterval = null;
let drawing = false;

initializeCanvas();
initializePredictionChart(T, C);
initializeTraceCharts(T, H);
createEventListeners();

const loadingSpinner = document.getElementById("loadingSpinner");
const runBtn = document.getElementById("runBtn");
const clearBtn = document.getElementById("clearBtn");

loadingSpinner.style.display = "block";
runBtn.disabled = true;
clearBtn.disabled = true;

let sessionPromise = ort.InferenceSession.create("assets/onnx/atm_mnist.onnx")
    .then(session => {
        loadingSpinner.style.display = "none";
        runBtn.disabled = false;
        clearBtn.disabled = false;
        return session;
    })
    .catch(error => {
        console.error("Failed to load ONNX model:", error);
        loadingSpinner.style.display = "none";
    });

runBtn.addEventListener("click", async () => {
    loadingSpinner.style.display = "block";
    runBtn.disabled = true;
    clearBtn.disabled = true;

    await runInference(animationUpdateIntervalInMs);

    loadingSpinner.style.display = "none";
    runBtn.disabled = false;
    clearBtn.disabled = false;
});

clearBtn.addEventListener("click", () => {
    clearAll();
});