/**
 * LPU Smart Parking — In-Browser Hardware-Accelerated YOLO Engine
 * Executes YOLO neural networks directly on the user's graphics card / CPU via WebGPU & WASM.
 * Zero cloud dependency. Zero Render server RAM/CPU consumption.
 */

class ClientYOLOEngine {
    constructor() {
        this.sessions = {};
        this.currentModel = 'best';
        this.isLoading = false;
        this.hasWebGPU = typeof navigator !== 'undefined' && !!navigator.gpu;
        this.activeProvider = this.hasWebGPU ? 'webgpu' : 'wasm';
        this.confThreshold = 0.25;
        this.iouThreshold = 0.45;
        this.imgSize = 640;
    }

    getHardwareInfo() {
        if (this.hasWebGPU) {
            return {
                mode: 'WebGPU (Hardware Accelerated GPU)',
                badgeClass: 'bg-emerald-100 text-emerald-800 border-emerald-300',
                icon: 'fa-bolt-lightning text-emerald-600'
            };
        } else {
            return {
                mode: 'WASM SIMD (Multi-threaded CPU Acceleration)',
                badgeClass: 'bg-sky-100 text-sky-800 border-sky-300',
                icon: 'fa-microchip text-sky-600'
            };
        }
    }

    async loadModel(modelChoice = 'best', onProgress = null) {
        if (this.sessions[modelChoice]) {
            return this.sessions[modelChoice];
        }

        if (typeof ort === 'undefined') {
            throw new Error('ONNX Runtime Web library not loaded. Please ensure CDN script is active.');
        }

        this.isLoading = true;
        const modelFileName = modelChoice === 'yolo26l' ? 'yolo26l.onnx' : 'best.onnx';
        const modelUrl = `/static/models/${modelFileName}`;

        if (onProgress) onProgress(`Loading ${modelFileName} into local hardware memory...`);

        try {
            // Priority: WebGPU (GPU) -> WASM (Multi-core CPU)
            const providers = this.hasWebGPU ? ['webgpu', 'wasm'] : ['wasm'];
            const session = await ort.InferenceSession.create(modelUrl, {
                executionProviders: providers,
                graphOptimizationLevel: 'all'
            });

            this.sessions[modelChoice] = session;
            this.isLoading = false;
            return session;
        } catch (err) {
            console.warn(`WebGPU session creation failed (${err.message}), falling back to WASM SIMD...`);
            // Fallback strictly to wasm if WebGPU had a driver conflict
            const session = await ort.InferenceSession.create(modelUrl, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });
            this.sessions[modelChoice] = session;
            this.isLoading = false;
            return session;
        }
    }

    /**
     * Preprocesses an HTMLImageElement, HTMLVideoElement, or HTMLCanvasElement
     * into a planar Float32 Tensor [1, 3, 640, 640] normalized to [0, 1].
     */
    preprocess(sourceElement) {
        const offCanvas = document.createElement('canvas');
        offCanvas.width = this.imgSize;
        offCanvas.height = this.imgSize;
        const ctx = offCanvas.getContext('2d', { willReadFrequently: true });

        // Draw stretched to 640x640 (standard YOLO inference dimensions)
        ctx.drawImage(sourceElement, 0, 0, this.imgSize, this.imgSize);
        const imgData = ctx.getImageData(0, 0, this.imgSize, this.imgSize);
        const { data } = imgData;

        const floatData = new Float32Array(3 * this.imgSize * this.imgSize);
        const channelSize = this.imgSize * this.imgSize;

        for (let i = 0; i < channelSize; i++) {
            const r = data[i * 4] / 255.0;
            const g = data[i * 4 + 1] / 255.0;
            const b = data[i * 4 + 2] / 255.0;

            floatData[i] = r;                          // Channel 0: R
            floatData[channelSize + i] = g;          // Channel 1: G
            floatData[2 * channelSize + i] = b;      // Channel 2: B
        }

        return new ort.Tensor('float32', floatData, [1, 3, this.imgSize, this.imgSize]);
    }

    /**
     * Executes inference on local hardware using WebGPU/WASM.
     */
    async detect(sourceElement, modelChoice = 'best', roiPolygon = null) {
        const session = await this.loadModel(modelChoice);
        const tensor = this.preprocess(sourceElement);

        const t0 = performance.now();
        const results = await session.run({ images: tensor });
        const latencyMs = Math.round(performance.now() - t0);

        const outputTensor = results.output0;
        if (!outputTensor) {
            throw new Error('Invalid output tensor from model.');
        }

        const origW = sourceElement.naturalWidth || sourceElement.videoWidth || sourceElement.width;
        const origH = sourceElement.naturalHeight || sourceElement.videoHeight || sourceElement.height;

        const rawDetections = this.postprocess(outputTensor, origW, origH, modelChoice, roiPolygon);

        return {
            modelName: modelChoice === 'yolo26l' ? 'Pretrained YOLO26 Large' : 'Custom LPU Fine-Tuned (best.pt)',
            latencyMs,
            hardwareProvider: this.hasWebGPU ? 'WebGPU (Hardware GPU)' : 'WASM (CPU SIMD)',
            rawDetections,
            roiDetections: rawDetections.filter(d => d.inRoi),
            outsideDetections: rawDetections.filter(d => !d.inRoi),
            totalCount: rawDetections.length,
            roiCount: rawDetections.filter(d => d.inRoi).length,
        };
    }

    /**
     * Parses YOLO output tensor [1, channels, 8400], applies NMS, and checks Jordan Ray-Casting PIP.
     */
    postprocess(tensor, origW, origH, modelChoice, roiPolygon) {
        const [batch, channels, numBoxes] = tensor.dims;
        const data = tensor.data;
        const scaleX = origW / this.imgSize;
        const scaleY = origH / this.imgSize;

        const isCustom = modelChoice === 'best';
        const candidates = [];

        // Parse anchors
        for (let i = 0; i < numBoxes; i++) {
            let conf = 0;
            let classId = 0;
            let className = 'Car';

            if (isCustom) {
                conf = data[4 * numBoxes + i]; // best.onnx: class 0 confidence
            } else {
                // yolo26l.onnx: vehicle classes: 2=car, 3=motorcycle, 5=bus, 7=truck
                const vehicleClasses = [
                    { id: 2, name: 'Car' },
                    { id: 3, name: 'Motorcycle' },
                    { id: 5, name: 'Bus' },
                    { id: 7, name: 'Truck' }
                ];
                let bestScore = -1;
                let bestClass = 2;
                let bestName = 'Car';

                for (const vc of vehicleClasses) {
                    const score = data[(4 + vc.id) * numBoxes + i];
                    if (score > bestScore) {
                        bestScore = score;
                        bestClass = vc.id;
                        bestName = vc.name;
                    }
                }
                conf = bestScore;
                classId = bestClass;
                className = bestName;
            }

            if (conf >= this.confThreshold) {
                const cx = data[0 * numBoxes + i] * scaleX;
                const cy = data[1 * numBoxes + i] * scaleY;
                const w = data[2 * numBoxes + i] * scaleX;
                const h = data[3 * numBoxes + i] * scaleY;

                const x1 = Math.max(0, Math.round(cx - w / 2));
                const y1 = Math.max(0, Math.round(cy - h / 2));
                const x2 = Math.min(origW, Math.round(cx + w / 2));
                const y2 = Math.min(origH, Math.round(cy + h / 2));

                candidates.push({
                    x1, y1, x2, y2,
                    w: x2 - x1,
                    h: y2 - y1,
                    confidence: Math.round(conf * 1000) / 1000,
                    classId,
                    className
                });
            }
        }

        // Apply Non-Maximum Suppression (NMS)
        const nmsDetections = this.applyNMS(candidates, this.iouThreshold);

        // Apply Jordan Ray-Casting Polygon ROI filter
        const hasRoi = roiPolygon && roiPolygon.length >= 3;

        return nmsDetections.map(det => {
            const contactX = Math.round((det.x1 + det.x2) / 2);
            const contactY = Math.max(det.y1, det.y2);
            const inRoi = hasRoi ? this.pointInPolygon(contactX, contactY, roiPolygon) : true;

            return {
                ...det,
                contactPoint: [contactX, contactY],
                inRoi
            };
        });
    }

    /**
     * Standard Greedy Non-Maximum Suppression
     */
    applyNMS(boxes, iouThreshold) {
        boxes.sort((a, b) => b.confidence - a.confidence);
        const selected = [];
        const active = new Array(boxes.length).fill(true);

        for (let i = 0; i < boxes.length; i++) {
            if (!active[i]) continue;
            const b1 = boxes[i];
            selected.push(b1);

            for (let j = i + 1; j < boxes.length; j++) {
                if (!active[j]) continue;
                const b2 = boxes[j];
                const iou = this.computeIoU(b1, b2);
                if (iou > iouThreshold) {
                    active[j] = false;
                }
            }
        }
        return selected;
    }

    computeIoU(a, b) {
        const xA = Math.max(a.x1, b.x1);
        const yA = Math.max(a.y1, b.y1);
        const xB = Math.min(a.x2, b.x2);
        const yB = Math.min(a.y2, b.y2);

        const interW = Math.max(0, xB - xA);
        const interH = Math.max(0, yB - yA);
        const interArea = interW * interH;
        if (interArea === 0) return 0;

        const areaA = a.w * a.h;
        const areaB = b.w * b.h;
        return interArea / (areaA + areaB - interArea);
    }

    /**
     * Jordan Curve / Ray-Casting Point-in-Polygon Algorithm
     */
    pointInPolygon(x, y, polygon) {
        let inside = false;
        const n = polygon.length;
        for (let i = 0, j = n - 1; i < n; j = i++) {
            const xi = polygon[i][0], yi = polygon[i][1];
            const xj = polygon[j][0], yj = polygon[j][1];

            const intersect = ((yi > y) !== (yj > y)) &&
                (x < ((xj - xi) * (y - yi)) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }
}

window.clientYOLO = new ClientYOLOEngine();
