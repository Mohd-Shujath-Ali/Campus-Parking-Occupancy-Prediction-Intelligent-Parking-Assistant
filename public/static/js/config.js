/**
 * LPU Campus Parking Intelligence System — Global API Configuration
 * Supports both unified deployments (same-origin) and decoupled deployments (Render Static Site + Web Service).
 */
(function () {
    'use strict';

    // 1. Determine active backend URL
    const STORAGE_KEY = 'LPU_PARKING_API_BASE';
    let stored = localStorage.getItem(STORAGE_KEY);

    // Auto-clean any legacy dummy placeholder URL from localStorage
    if (stored && stored.includes('campus-parking-occupancy-prediction-intelligent.onrender.com')) {
        localStorage.removeItem(STORAGE_KEY);
        stored = null;
    }

    let defaultBase = '';
    const host = window.location.hostname;
    const port = window.location.port;

    if (stored !== null && stored.trim() !== '') {
        defaultBase = stored.trim().replace(/\/+$/, '');
    } else if (window.__API_BASE__) {
        defaultBase = window.__API_BASE__.replace(/\/+$/, '');
    } else if (host === 'localhost' || host === '127.0.0.1' || host === '') {
        // Localhost: If running directly on FastAPI (port 8000, 10000, etc.), use relative '';
        // Only point to 127.0.0.1:8000 if accessed from a separate static dev server (e.g. port 3000, 5173, 5500)
        defaultBase = (port === '8000' || port === '10000' || port === '' || !['3000', '5173', '5500'].includes(port)) ? '' : 'http://127.0.0.1:8000';
    } else {
        // Deployed on Render or any web domain:
        // Default to same origin (relative path '') so it connects to the active deployment!
        defaultBase = '';
    }

    window.API_BASE = defaultBase;

    window.apiUrl = function (path) {
        if (!path) return window.API_BASE || '';
        const cleanPath = path.startsWith('/') ? path : '/' + path;
        if (!window.API_BASE) return cleanPath;
        return window.API_BASE + cleanPath;
    };

    window.setApiBase = function (newUrl) {
        const cleaned = (newUrl || '').trim().replace(/\/+$/, '');
        if (!cleaned) {
            localStorage.removeItem(STORAGE_KEY);
            window.API_BASE = (port === '8000' || port === '10000' || host === 'localhost' || host === '127.0.0.1') ? '' : 'http://127.0.0.1:8000';
        } else {
            localStorage.setItem(STORAGE_KEY, cleaned);
            window.API_BASE = cleaned;
        }
    };

    // 2. Global Modal for Backend Configuration
    function injectApiConfigModal() {
        if (document.getElementById('apiConfigModal')) return;

        const modalHtml = `
        <div id="apiConfigModal" class="fixed inset-0 bg-black/50 backdrop-blur-xs z-50 hidden flex items-center justify-center p-4">
            <div class="bg-white rounded-2xl max-w-md w-full shadow-2xl border border-slate-200 overflow-hidden transform transition-all">
                <div class="bg-slate-900 text-white px-6 py-4 flex items-center justify-between">
                    <div class="flex items-center space-x-2">
                        <i class="fa-solid fa-server text-orange-400"></i>
                        <h3 class="font-bold text-sm">Backend API Endpoint</h3>
                    </div>
                    <button onclick="closeApiConfigModal()" class="text-slate-400 hover:text-white transition-colors">
                        <i class="fa-solid fa-xmark text-lg"></i>
                    </button>
                </div>
                <div class="p-6 space-y-4">
                    <p class="text-xs text-slate-600">
                        When hosting the Frontend as a <strong>Render Static Site</strong> and the Backend as a <strong>Render Web Service</strong>, enter your Render backend URL below.
                    </p>
                    <div>
                        <label class="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">Backend Web Service URL</label>
                        <input id="apiBaseInput" type="text" placeholder="https://your-backend.onrender.com" 
                            class="w-full bg-slate-50 border border-slate-300 rounded-xl px-3.5 py-2 text-xs font-mono focus:ring-2 focus:ring-orange-500 focus:outline-none">
                        <p class="text-[11px] text-slate-400 mt-1">Leave empty to use relative origin (same domain / localhost:8000).</p>
                    </div>
                    <div id="apiTestStatus" class="hidden text-xs p-2.5 rounded-xl border"></div>
                    <div class="flex items-center justify-end space-x-2 pt-2 border-t border-slate-100">
                        <button onclick="testApiEndpoint()" id="btnTestApi" class="px-3.5 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 font-bold text-xs rounded-xl transition-all">
                            Test Connection
                        </button>
                        <button onclick="saveApiConfig()" class="px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white font-bold text-xs rounded-xl transition-all shadow-xs">
                            Save & Reload
                        </button>
                    </div>
                </div>
            </div>
        </div>
        `;
        document.body.insertAdjacentHTML('beforeend', modalHtml);
    }

    window.openApiConfigModal = function () {
        injectApiConfigModal();
        const modal = document.getElementById('apiConfigModal');
        const input = document.getElementById('apiBaseInput');
        if (input) input.value = localStorage.getItem(STORAGE_KEY) || window.API_BASE || '';
        const statusBox = document.getElementById('apiTestStatus');
        if (statusBox) statusBox.className = 'hidden';
        if (modal) modal.classList.remove('hidden');
    };

    window.closeApiConfigModal = function () {
        const modal = document.getElementById('apiConfigModal');
        if (modal) modal.classList.add('hidden');
    };

    window.testApiEndpoint = async function () {
        const input = document.getElementById('apiBaseInput');
        const statusBox = document.getElementById('apiTestStatus');
        const btn = document.getElementById('btnTestApi');
        const rawUrl = (input ? input.value : '').trim().replace(/\/+$/, '');
        const target = (rawUrl || window.location.origin) + '/api/health';

        if (btn) btn.disabled = true;
        if (statusBox) {
            statusBox.className = 'text-xs p-2.5 rounded-xl border bg-slate-50 border-slate-200 text-slate-600';
            statusBox.textContent = `Connecting to ${target}...`;
            statusBox.classList.remove('hidden');
        }

        const t0 = performance.now();
        try {
            const res = await fetch(target, { cache: 'no-store' });
            const lat = Math.round(performance.now() - t0);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            const mlOk = data.ml_model?.xgboost_loaded ? '✓ XGBoost' : '⚠ Heuristic';
            const cvOk = data.yolo_model?.loaded ? '✓ YOLO' : '⚠ Vision offline';
            statusBox.className = 'text-xs p-2.5 rounded-xl border bg-emerald-50 border-emerald-300 text-emerald-800';
            statusBox.innerHTML = `<strong>Online (${lat}ms):</strong> ${mlOk} · ${cvOk} (12 zones active)`;
        } catch (err) {
            statusBox.className = 'text-xs p-2.5 rounded-xl border bg-rose-50 border-rose-300 text-rose-800';
            statusBox.innerHTML = `<strong>Connection Failed:</strong> ${err.message}<br><span class="text-[10px]">Note: Render Free Tier services take ~50s to wake up on first request.</span>`;
        } finally {
            if (btn) btn.disabled = false;
        }
    };

    window.saveApiConfig = function () {
        const input = document.getElementById('apiBaseInput');
        const rawUrl = (input ? input.value : '').trim();
        window.setApiBase(rawUrl);
        window.location.reload();
    };

    document.addEventListener('DOMContentLoaded', () => {
        // Wire click handler on systemStatusDot to open API configuration modal
        const dot = document.getElementById('systemStatusDot');
        if (dot) {
            dot.style.cursor = 'pointer';
            dot.title = 'Click to view / configure Backend API URL';
            dot.addEventListener('click', window.openApiConfigModal);
        }
    });
})();
