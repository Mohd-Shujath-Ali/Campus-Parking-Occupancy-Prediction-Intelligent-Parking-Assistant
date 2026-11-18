'use strict';

// ─────────────────────────── Webcam State ──────────────────────────
let webcamStream = null;
let detectionIntervalId = null;
let isDetecting = false;
let fpsTimer = null;
let fpsFrames = 0;

// ─────────────────────────── ROI & Upload State ────────────────────
let selectedImageFile = null;
let loadedImgElement = null;
let roiPoints = [];          // Array of [x, y] in original image pixel coordinates
let roiMode = 'polygon';     // 'polygon' | 'box'
let isDrawingBox = false;
let boxStartPoint = null;

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

// ─────────────────────────── Webcam Functions ──────────────────────
async function toggleWebcam() {
    const video    = document.getElementById('webcamVideo');
    const ph       = document.getElementById('webcamPlaceholder');
    const annImg   = document.getElementById('webcamAnnotatedImg');
    const lbl      = document.getElementById('btnWebcamLabel');
    const badge    = document.getElementById('webcamStatusBadge');
    const btnDetect = document.getElementById('btnToggleDetection');

    if (webcamStream) {
        if (isDetecting) await toggleWebcamDetection();
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
        video.srcObject = null;
        video.classList.add('hidden');
        annImg.classList.add('hidden');
        ph.classList.remove('hidden');
        lbl.textContent = 'Start Webcam';
        badge.textContent = 'Idle';
        badge.className = 'text-xs px-2.5 py-0.5 rounded-full font-bold bg-slate-100 text-slate-600 border border-slate-200';
        btnDetect.disabled = true;
    } else {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } } });
            video.srcObject = webcamStream;
            video.classList.remove('hidden');
            ph.classList.add('hidden');
            lbl.textContent = 'Stop Webcam';
            badge.textContent = 'Live';
            badge.className = 'text-xs px-2.5 py-0.5 rounded-full font-bold bg-emerald-100 text-emerald-700 border border-emerald-200';
            btnDetect.disabled = false;

            fpsFrames = 0;
            if (fpsTimer) clearInterval(fpsTimer);
            fpsTimer = setInterval(() => {
                setText('webcamFps', fpsFrames.toFixed(1));
                fpsFrames = 0;
            }, 1000);
        } catch (err) {
            alert('Camera access denied or unavailable: ' + err.message);
        }
    }
}

async function toggleWebcamDetection() {
    const lbl   = document.getElementById('btnDetectionLabel');
    const badge = document.getElementById('webcamStatusBadge');
    const stats = document.getElementById('webcamOverlayStats');
    const video = document.getElementById('webcamVideo');
    const ann   = document.getElementById('webcamAnnotatedImg');

    if (isDetecting) {
        clearInterval(detectionIntervalId);
        detectionIntervalId = null;
        isDetecting = false;
        lbl.textContent   = 'Start Inference';
        badge.textContent = 'Live';
        badge.className   = 'text-xs px-2.5 py-0.5 rounded-full font-bold bg-emerald-100 text-emerald-700 border border-emerald-200';
        stats.classList.add('hidden');
        ann.classList.add('hidden');
        video.classList.remove('hidden');
    } else {
        isDetecting = true;
        lbl.textContent   = 'Stop Inference';
        badge.textContent = 'Inferring…';
        badge.className   = 'text-xs px-2.5 py-0.5 rounded-full font-bold bg-orange-100 text-orange-700 animate-pulse border border-orange-200';
        stats.classList.remove('hidden');
        await captureAndDetect();
        detectionIntervalId = setInterval(captureAndDetect, 1800);
    }
}

async function captureAndDetect() {
    if (!webcamStream || !isDetecting) return;
    const video  = document.getElementById('webcamVideo');
    const canvas = document.getElementById('hiddenWebcamCanvas');
    if (!video.videoWidth) return;

    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
        if (!blob) return;
        const fd = new FormData();
        fd.append('file', blob, 'frame.jpg');
        try {
            const url = window.apiUrl ? window.apiUrl('/api/detect-frame') : '/api/detect-frame';
            const res = await fetch(url, { method: 'POST', body: fd });
            if (!res.ok) return;
            const d = await res.json();
            if (d.annotated_image_base64) {
                const ann = document.getElementById('webcamAnnotatedImg');
                ann.src = 'data:image/jpeg;base64,' + d.annotated_image_base64;
                ann.classList.remove('hidden');
                // Ensure video remains active in DOM so canvas.drawImage continues grabbing frames
                const videoEl = document.getElementById('webcamVideo');
                if (videoEl) videoEl.classList.remove('hidden');
                const count = d.vehicle_count !== undefined ? d.vehicle_count : (d.vehicle_count_total || 0);
                setText('webcamCarCount', count);
                setText('webcamLatency',  d.inference_time_ms + 'ms');
                fpsFrames += 1 / (1800 / 1000);
            }
        } catch (err) {
            console.error('Frame error:', err);
        }
    }, 'image/jpeg', 0.82);
}

// ─────────────────────────── ROI & Upload Functions ────────────────
function setRoiMode(mode) {
    roiMode = mode;
    document.getElementById('btnRoiPolygon')?.classList.toggle('bg-orange-50', mode === 'polygon');
    document.getElementById('btnRoiPolygon')?.classList.toggle('text-orange-600', mode === 'polygon');
    document.getElementById('btnRoiBox')?.classList.toggle('bg-orange-50', mode === 'box');
    document.getElementById('btnRoiBox')?.classList.toggle('text-orange-600', mode === 'box');
}

function clearRoi() {
    roiPoints = [];
    isDrawingBox = false;
    boxStartPoint = null;
    redrawRoiCanvas();
    updateRoiIndicator();
}

function updateRoiIndicator() {
    const ind = document.getElementById('roiPointsIndicator');
    if (!ind) return;
    if (roiPoints.length === 0) {
        ind.textContent = 'No ROI (Full Lot)';
    } else if (roiMode === 'box') {
        ind.textContent = 'Bounding Box Active';
    } else {
        ind.textContent = `Polygon: ${roiPoints.length} vertices`;
    }
}

function handleDrop(e) {
    e.preventDefault();
    document.getElementById('dropZone')?.classList.remove('border-orange-500');
    if (e.dataTransfer?.files?.length > 0) {
        processUploadedImage(e.dataTransfer.files[0]);
    }
}

function handleImageSelected(e) {
    if (e.target.files?.length > 0) {
        processUploadedImage(e.target.files[0]);
    }
}

function processUploadedImage(file) {
    selectedImageFile = file;
    const reader = new FileReader();
    reader.onload = (ev) => {
        const img = new Image();
        img.onload = () => {
            loadedImgElement = img;
            initRoiCanvas(img);
            document.getElementById('btnRunDetect').disabled = false;
            document.getElementById('uploadPlaceholder')?.classList.add('hidden');
            document.getElementById('roiCanvas')?.classList.remove('hidden');
            document.getElementById('uploadStatusBadge').textContent = 'Image Ready';
            document.getElementById('uploadStatusBadge').className = 'text-xs px-2.5 py-0.5 rounded-full font-bold bg-emerald-100 text-emerald-700 border border-emerald-200';
        };
        img.src = ev.target.result;
    };
    reader.readAsDataURL(file);
}

function initRoiCanvas(img) {
    const canvas = document.getElementById('roiCanvas');
    if (!canvas) return;
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    clearRoi();
    setupRoiCanvasEvents(canvas);
}

function setupRoiCanvasEvents(canvas) {
    canvas.onmousedown = (e) => {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const x = Math.round((e.clientX - rect.left) * scaleX);
        const y = Math.round((e.clientY - rect.top) * scaleY);

        if (roiMode === 'polygon') {
            roiPoints.push([x, y]);
            redrawRoiCanvas();
            updateRoiIndicator();
        } else if (roiMode === 'box') {
            isDrawingBox = true;
            boxStartPoint = [x, y];
            roiPoints = [boxStartPoint];
        }
    };

    canvas.onmousemove = (e) => {
        if (roiMode === 'box' && isDrawingBox && boxStartPoint) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const x = Math.round((e.clientX - rect.left) * scaleX);
            const y = Math.round((e.clientY - rect.top) * scaleY);

            const x1 = Math.min(boxStartPoint[0], x);
            const y1 = Math.min(boxStartPoint[1], y);
            const x2 = Math.max(boxStartPoint[0], x);
            const y2 = Math.max(boxStartPoint[1], y);

            roiPoints = [
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ];
            redrawRoiCanvas();
        }
    };

    canvas.onmouseup = () => {
        if (roiMode === 'box' && isDrawingBox) {
            isDrawingBox = false;
            updateRoiIndicator();
        }
    };
}

function redrawRoiCanvas() {
    const canvas = document.getElementById('roiCanvas');
    if (!canvas || !loadedImgElement) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(loadedImgElement, 0, 0, canvas.width, canvas.height);

    if (roiPoints.length < 2) return;

    ctx.save();
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = Math.max(3, Math.round(canvas.width / 400));
    ctx.fillStyle = 'rgba(245, 158, 11, 0.20)';

    ctx.beginPath();
    ctx.moveTo(roiPoints[0][0], roiPoints[0][1]);
    for (let i = 1; i < roiPoints.length; i++) {
        ctx.lineTo(roiPoints[i][0], roiPoints[i][1]);
    }
    if (roiPoints.length >= 3) {
        ctx.closePath();
        ctx.fill();
    }
    ctx.stroke();

    // Draw vertex handles
    ctx.fillStyle = '#ea580c';
    for (const pt of roiPoints) {
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], Math.max(4, Math.round(canvas.width / 200)), 0, 2 * Math.PI);
        ctx.fill();
    }
    ctx.restore();
}

async function uploadAndDetectImage() {
    if (!selectedImageFile) return;

    if (roiPoints.length > 0 && roiPoints.length < 3) {
        alert('Please complete the ROI polygon with at least 3 vertices (or click "Clear ROI" to detect the full lot).');
        return;
    }

    const btn = document.getElementById('btnRunDetect');
    const lbl = document.getElementById('btnDetectLabel');
    const modelChoice = document.getElementById('modelChoiceSelect')?.value || 'both';

    btn.disabled = true;
    lbl.textContent = 'Processing YOLO Inference…';

    const fd = new FormData();
    fd.append('file', selectedImageFile);
    fd.append('model_choice', modelChoice);

    if (roiPoints.length >= 3) {
        fd.append('roi', JSON.stringify(roiPoints));
    }

    try {
        const url = window.apiUrl ? window.apiUrl('/api/detect-image') : '/api/detect-image';
        const res = await fetch(url, { method: 'POST', body: fd });
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
        const data = await res.json();
        renderDualModelResults(data);
    } catch (err) {
        alert('Detection failed: ' + err.message);
    } finally {
        btn.disabled = false;
        lbl.textContent = 'Run Detection & Compare in ROI';
    }
}

function renderDualModelResults(data) {
    const wrap = document.getElementById('dualModelResultsWrap');
    if (!wrap) return;
    wrap.classList.remove('hidden');

    const modelChoice = document.getElementById('modelChoiceSelect')?.value || 'both';
    const comp = data.comparison || {};
    const best = data.best_model || (modelChoice === 'best' ? data : {});
    const yolo = data.yolo26l_model || (modelChoice === 'yolo26l' ? data : {});

    setText('compHeaderRegion', `Target Region: ${comp.target_region || (data.roi_defined ? 'Custom ROI' : 'Full Parking Lot')}`);
    setText('compSpeedBadge', comp.speed_summary || `${modelChoice.toUpperCase()} Inference Complete`);

    // Best Model Card
    const bestCard = document.getElementById('cardBestModel');
    if (modelChoice === 'yolo26l') {
        if (bestCard) bestCard.classList.add('opacity-40');
        setText('bestCountRoi', '—');
        setText('bestCountTotal', '—');
        setText('bestAvgConf', 'Not Run');
        setText('bestLatency', '—');
    } else {
        if (bestCard) bestCard.classList.remove('opacity-40');
        setText('bestCountRoi', best.vehicle_count_roi ?? data.vehicle_count_roi ?? 0);
        setText('bestCountTotal', best.vehicle_count_total ?? data.vehicle_count_total ?? 0);
        const conf = best.avg_confidence !== undefined ? Math.round(best.avg_confidence * 100) + '%' : (data.avg_confidence ? Math.round(data.avg_confidence * 100) + '%' : '—');
        setText('bestAvgConf', conf);
        setText('bestLatency', (best.inference_time_ms || data.inference_time_ms || 0) + ' ms');
    }

    // YOLO26l Card
    const yoloCard = document.getElementById('cardYolo26l');
    if (modelChoice === 'best') {
        if (yoloCard) yoloCard.classList.add('opacity-40');
        setText('yolo26lCountRoi', '—');
        setText('yolo26lCountTotal', '—');
        setText('yolo26lAvgConf', 'Not Run');
        setText('yolo26lLatency', '—');
    } else {
        if (yoloCard) yoloCard.classList.remove('opacity-40');
        setText('yolo26lCountRoi', yolo.vehicle_count_roi ?? (modelChoice === 'yolo26l' ? data.vehicle_count_roi : 0));
        setText('yolo26lCountTotal', yolo.vehicle_count_total ?? (modelChoice === 'yolo26l' ? data.vehicle_count_total : 0));
        const conf = yolo.avg_confidence !== undefined ? Math.round(yolo.avg_confidence * 100) + '%' : '—';
        setText('yolo26lAvgConf', conf);
        setText('yolo26lLatency', (yolo.inference_time_ms || (modelChoice === 'yolo26l' ? data.inference_time_ms : 0)) + ' ms');
    }

    // Annotated images
    const imgBest = document.getElementById('imgBestAnnotated');
    const b64Best = best.annotated_image_base64 || (modelChoice === 'best' ? data.annotated_image_base64 : '');
    if (imgBest && b64Best) imgBest.src = 'data:image/jpeg;base64,' + b64Best;

    const imgYolo = document.getElementById('imgYolo26lAnnotated');
    const b64Yolo = yolo.annotated_image_base64 || (modelChoice === 'yolo26l' ? data.annotated_image_base64 : '');
    if (imgYolo && b64Yolo) imgYolo.src = 'data:image/jpeg;base64,' + b64Yolo;

    const bulletsList = document.getElementById('compAnalysisBullets');
    if (bulletsList) {
        if (comp.analysis_bullets && comp.analysis_bullets.length > 0) {
            bulletsList.innerHTML = comp.analysis_bullets.map(b => `<li>${b}</li>`).join('');
        } else {
            bulletsList.innerHTML = `<li>Executed single-model evaluation (${modelChoice.toUpperCase()}) across selected detection region.</li>`;
        }
    }

    wrap.scrollIntoView({ behavior: 'smooth' });
}
