'use strict';

/*
 * Admin camera control on the vision page:
 *  - per-camera video source / enabled flag
 *  - saving the ROI drawn in the studio to a camera (versioned), loading it back, rolling back
 *  - publishing the last detected count for a camera into its ground's occupancy history
 * Uses the ROI studio globals from vision.js: roiPoints, redrawRoiCanvas(), updateRoiIndicator().
 */
(function () {
    const esc = window.escapeHtml;
    let CAMERAS = [];
    let SOURCE_TYPES = [];
    let lastDetection = null;

    // Remember the most recent detection result so it can be published.
    if (typeof window.renderDualModelResults === 'function') {
        const original = window.renderDualModelResults;
        window.renderDualModelResults = function (data) {
            lastDetection = data;
            return original.apply(this, arguments);
        };
    }

    const selectedCamera = () => document.getElementById('roiCameraSelect')?.value;
    const cam = id => CAMERAS.find(c => c.camera_id === id);

    function roiCanvasSize() {
        const c = document.getElementById('roiCanvas');
        return c && !c.classList.contains('hidden') && c.width > 0 ? { w: c.width, h: c.height } : null;
    }

    function redrawStudio() {
        if (typeof redrawRoiCanvas === 'function') redrawRoiCanvas();
        if (typeof updateRoiIndicator === 'function') updateRoiIndicator();
    }

    // ───────────── Camera table ─────────────
    async function loadCameras() {
        const body = document.getElementById('cameraAdminBody');
        try {
            const data = await window.apiJson('/api/cameras');
            CAMERAS = data.cameras;
            SOURCE_TYPES = data.source_types;
            renderCameras();
            const sel = document.getElementById('roiCameraSelect');
            if (sel) {
                const keep = sel.value;
                sel.innerHTML = CAMERAS.map(c => `<option value="${esc(c.camera_id)}">${esc(c.camera_id)} · ${esc(c.zone_name)}</option>`).join('');
                if (keep && cam(keep)) sel.value = keep;
                loadHistory();
            }
        } catch (e) {
            body.innerHTML = `<tr><td colspan="7" class="px-4 py-6 text-center text-rose-600">${esc(e.message)}</td></tr>`;
        }
    }

    function renderCameras() {
        const body = document.getElementById('cameraAdminBody');
        body.innerHTML = CAMERAS.map(c => `
            <tr class="hover:bg-slate-50" data-cam="${esc(c.camera_id)}">
                <td class="px-4 py-2.5 font-mono font-bold text-slate-900 whitespace-nowrap">${esc(c.camera_id)}</td>
                <td class="px-4 py-2.5">${esc(c.zone_name)}</td>
                <td class="px-4 py-2.5">
                    <select data-field="source_type" class="bg-slate-50 border border-slate-300 rounded-lg px-2 py-1 text-xs font-medium focus:outline-none focus:ring-1 focus:ring-orange-500">
                        ${SOURCE_TYPES.map(t => `<option value="${t}" ${t === c.source_type ? 'selected' : ''}>${t}</option>`).join('')}
                    </select>
                </td>
                <td class="px-4 py-2.5">
                    <input data-field="source_url" value="${esc(c.source_url || '')}" maxlength="500" placeholder="rtsp://… or https://…" class="w-full bg-slate-50 border border-slate-300 rounded-lg px-2 py-1 text-xs font-mono focus:outline-none focus:ring-1 focus:ring-orange-500">
                </td>
                <td class="px-4 py-2.5"><input data-field="enabled" type="checkbox" class="w-4 h-4 accent-orange-600" ${c.enabled ? 'checked' : ''} aria-label="Enabled"></td>
                <td class="px-4 py-2.5">${c.active_roi_version ? `<span class="font-mono font-bold text-emerald-700">v${c.active_roi_version}</span> <span class="text-slate-400">(${c.active_roi_points} pts)</span>` : '<span class="text-slate-400">Full frame</span>'}</td>
                <td class="px-4 py-2.5 text-right whitespace-nowrap">
                    <button data-save class="px-2.5 py-1 bg-slate-900 hover:bg-orange-600 text-white rounded-lg font-bold">Save</button>
                    <button data-edit-roi class="px-2.5 py-1 bg-white border border-slate-300 hover:bg-orange-50 rounded-lg font-bold" title="Select this camera in the ROI studio">ROI</button>
                </td>
            </tr>`).join('');

        body.querySelectorAll('tr[data-cam]').forEach(tr => {
            const id = tr.dataset.cam;
            tr.querySelector('[data-save]').addEventListener('click', async () => {
                const payload = {
                    source_type: tr.querySelector('[data-field="source_type"]').value,
                    source_url: tr.querySelector('[data-field="source_url"]').value.trim() || null,
                    enabled: tr.querySelector('[data-field="enabled"]').checked,
                };
                try {
                    await window.apiJson(`/api/cameras/${encodeURIComponent(id)}`, { method: 'PATCH', body: payload });
                    window.toast(`${id} saved.`, 'success');
                    loadCameras();
                } catch (e) { window.toast(e.message, 'error'); }
            });
            tr.querySelector('[data-edit-roi]').addEventListener('click', () => {
                const sel = document.getElementById('roiCameraSelect');
                sel.value = id;
                loadHistory();
                sel.scrollIntoView({ behavior: 'smooth', block: 'center' });
                window.toast(`ROI studio now targets ${id}. Upload a frame from this camera, draw, then Save ROI.`, 'info');
            });
        });
    }

    // ───────────── ROI save / load / history ─────────────
    async function loadHistory() {
        const id = selectedCamera();
        const hist = document.getElementById('roiHistorySelect');
        if (!id || !hist) return;
        try {
            const { versions } = await window.apiJson(`/api/cameras/${encodeURIComponent(id)}/roi/history`);
            hist.innerHTML = '<option value="">History…</option>' + versions.map(v =>
                `<option value="${v.version}">v${v.version}${v.is_active ? ' (active)' : ''} · ${esc(v.created_by)} · ${esc(window.formatWhen(v.created_at))}${v.note ? ' · ' + esc(v.note) : ''}</option>`).join('');
        } catch (e) {
            hist.innerHTML = '<option value="">History unavailable</option>';
        }
    }

    window.saveRoiToCamera = async function () {
        const id = selectedCamera();
        if (!id) return;
        if (!Array.isArray(roiPoints) || roiPoints.length < 3) {
            window.toast('Draw an ROI with at least 3 points first.', 'warning');
            return;
        }
        const size = roiCanvasSize();
        const note = prompt(`Save this ROI as the new active version for ${id}.\nOptional note:`, '');
        if (note === null) return;
        try {
            const r = await window.apiJson(`/api/cameras/${encodeURIComponent(id)}/roi`, { method: 'PUT', body: {
                polygon: roiPoints, image_width: size ? size.w : null, image_height: size ? size.h : null, note,
            }});
            window.toast(`Saved ROI v${r.version} for ${id}.`, 'success');
            loadCameras();
        } catch (e) { window.toast(e.message, 'error'); }
    };

    function applyRoi(roi, id) {
        if (!roi) { window.toast(`${id} has no saved ROI (full frame).`, 'info'); return; }
        roiPoints = roi.polygon.map(p => [p[0], p[1]]);
        redrawStudio();
        const size = roiCanvasSize();
        if (size && roi.image_width && (size.w !== roi.image_width || size.h !== roi.image_height)) {
            window.toast(`ROI v${roi.version} was drawn on a ${roi.image_width}×${roi.image_height} frame; this image is ${size.w}×${size.h}. Check the fit.`, 'warning');
        } else {
            window.toast(`Loaded ROI v${roi.version} for ${id}.`, 'success');
        }
    }

    window.loadCameraRoi = async function () {
        const id = selectedCamera();
        if (!id) return;
        try {
            const { active } = await window.apiJson(`/api/cameras/${encodeURIComponent(id)}/roi`);
            applyRoi(active, id);
        } catch (e) { window.toast(e.message, 'error'); }
    };

    document.addEventListener('change', async (e) => {
        if (e.target.id === 'roiCameraSelect') {
            loadHistory();
        } else if (e.target.id === 'roiHistorySelect' && e.target.value) {
            const id = selectedCamera();
            const version = e.target.value;
            if (!confirm(`Make ROI v${version} the active version for ${id}? (Roll back)`)) { e.target.value = ''; return; }
            try {
                const roi = await window.apiJson(`/api/cameras/${encodeURIComponent(id)}/roi/${version}/activate`, { method: 'POST' });
                applyRoi(roi, id);
                loadCameras();
            } catch (err) { window.toast(err.message, 'error'); }
        }
    });

    // ───────────── Publish a count into the ground's history ─────────────
    window.publishCount = async function () {
        const id = selectedCamera();
        const c = cam(id);
        if (!c) return;
        let count = null;
        if (lastDetection) {
            count = lastDetection.vehicle_count ?? lastDetection.best_model?.vehicle_count_roi ?? null;
        } else {
            const live = document.getElementById('webcamCarCount');
            if (live && !document.getElementById('webcamOverlayStats')?.classList.contains('hidden')) count = parseInt(live.textContent, 10);
        }
        if (count == null || isNaN(count)) {
            window.toast('Run a detection first (image or live webcam).', 'warning');
            return;
        }
        if (!confirm(`Publish ${count} vehicles as the current count for ${c.zone_name} (${id})?`)) return;
        try {
            const r = await window.apiJson('/api/ingest', { method: 'POST', body: { zone_id: c.zone_id, camera_id: id, count, source: 'camera' } });
            window.toast(`${c.zone_name}: ${r.zone_total} vehicles (${Math.round(r.occupancy_rate * 100)}% full) recorded.`, 'success');
        } catch (e) { window.toast(e.message, 'error'); }
    };

    document.addEventListener('DOMContentLoaded', loadCameras);
})();
