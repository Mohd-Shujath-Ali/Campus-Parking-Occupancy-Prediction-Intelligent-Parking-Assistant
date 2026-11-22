'use strict';

/* Estate console operations: ground overrides (warden/admin) and violation reports. */
(function () {
    const esc = window.escapeHtml;
    let ZONES = [];

    const zoneOptions = () => ZONES.map(z => `<option value="${esc(z.zone_id)}">${esc(z.name)}</option>`).join('');

    // ───────────── Overrides ─────────────
    async function loadOverrides() {
        const list = document.getElementById('overrideList');
        if (!list) return;
        try {
            const { overrides } = await window.apiJson('/api/overrides');
            if (!overrides.length) {
                list.innerHTML = '<li class="text-slate-400">No active overrides: every ground is open to recommendations.</li>';
                return;
            }
            list.innerHTML = overrides.map(o => `
                <li class="flex items-start justify-between gap-3 p-2.5 rounded-xl border ${o.status === 'CLOSED' ? 'bg-slate-100 border-slate-300' : 'bg-rose-50 border-rose-200'}">
                    <div>
                        <div class="font-bold text-slate-800">${esc(o.zone_name)} <span class="ml-1 text-[10px] uppercase tracking-wider ${o.status === 'CLOSED' ? 'text-slate-600' : 'text-rose-700'}">${esc(o.status)}</span></div>
                        <div class="text-slate-500">${o.reason ? esc(o.reason) + ' · ' : ''}by ${esc(o.set_by)} · ${o.expires_at ? 'until ' + esc(window.formatWhen(o.expires_at)) : 'until cleared'}</div>
                    </div>
                    <button data-clear="${esc(o.zone_id)}" class="shrink-0 px-2.5 py-1 bg-white border border-slate-300 hover:bg-emerald-50 hover:border-emerald-300 hover:text-emerald-700 rounded-lg font-bold">Reopen</button>
                </li>`).join('');
            list.querySelectorAll('[data-clear]').forEach(b => b.addEventListener('click', async () => {
                try {
                    await window.apiJson(`/api/zones/${encodeURIComponent(b.dataset.clear)}/override`, { method: 'DELETE' });
                    window.toast('Ground reopened.', 'success');
                    refreshAll();
                } catch (e) { window.toast(e.message, 'error'); }
            }));
        } catch (e) {
            list.innerHTML = `<li class="text-rose-600">${esc(e.message)}</li>`;
        }
    }

    const ovForm = document.getElementById('overrideForm');
    if (ovForm) ovForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const zone = document.getElementById('ovZone').value;
        const minutes = document.getElementById('ovMinutes').value;
        try {
            await window.apiJson(`/api/zones/${encodeURIComponent(zone)}/override`, { method: 'PUT', body: {
                status: document.getElementById('ovStatus').value,
                reason: document.getElementById('ovReason').value.trim(),
                minutes: minutes ? parseInt(minutes, 10) : null,
            }});
            document.getElementById('ovReason').value = '';
            window.toast('Override applied.', 'success');
            refreshAll();
        } catch (err) { window.toast(err.message, 'error'); }
    });

    // ───────────── Reports ─────────────
    async function loadReports() {
        const list = document.getElementById('reportList');
        if (!list || !window.hasPerm('view_reports')) return;
        const status = document.getElementById('reportFilter').value;
        try {
            const data = await window.apiJson('/api/reports' + (status ? `?status=${status}` : ''));
            if (!data.reports.length) {
                list.innerHTML = `<li class="text-slate-400">No ${status || ''} reports.</li>`;
                return;
            }
            const canResolve = window.hasPerm('resolve_reports');
            list.innerHTML = data.reports.map(r => `
                <li class="p-2.5 rounded-xl border ${r.status === 'open' ? 'bg-amber-50 border-amber-200' : 'bg-slate-50 border-slate-200'}">
                    <div class="flex items-start justify-between gap-2">
                        <div>
                            <div class="font-bold text-slate-800">${esc(r.category_label)}${r.vehicle_number ? ` · <span class="font-mono">${esc(r.vehicle_number)}</span>` : ''}</div>
                            <div class="text-slate-500">${esc(r.zone_name)} · ${esc(window.formatWhen(r.created_at))} · by ${esc(r.reporter_username)}</div>
                            ${r.description ? `<div class="text-slate-700 mt-1">${esc(r.description)}</div>` : ''}
                            ${r.status === 'resolved' ? `<div class="text-emerald-700 mt-1"><i class="fa-solid fa-check"></i> Resolved by ${esc(r.resolved_by)}${r.resolution_note ? ': ' + esc(r.resolution_note) : ''}</div>` : ''}
                        </div>
                        ${r.status === 'open' && canResolve ? `<button data-resolve="${r.id}" class="shrink-0 px-2.5 py-1 bg-white border border-slate-300 hover:bg-emerald-50 hover:text-emerald-700 rounded-lg font-bold">Resolve</button>` : ''}
                    </div>
                </li>`).join('');
            list.querySelectorAll('[data-resolve]').forEach(b => b.addEventListener('click', async () => {
                const note = prompt('Resolution note (optional):', '');
                if (note === null) return;
                try {
                    await window.apiJson(`/api/reports/${b.dataset.resolve}/resolve`, { method: 'POST', body: { note } });
                    window.toast('Report resolved.', 'success');
                    loadReports();
                } catch (e) { window.toast(e.message, 'error'); }
            }));
        } catch (e) {
            list.innerHTML = `<li class="text-rose-600">${esc(e.message)}</li>`;
        }
    }

    const rpForm = document.getElementById('reportForm');
    if (rpForm) rpForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        try {
            await window.apiJson('/api/reports', { method: 'POST', body: {
                zone_id: document.getElementById('rpZone').value,
                category: document.getElementById('rpCategory').value,
                vehicle_number: document.getElementById('rpPlate').value.trim() || null,
                description: document.getElementById('rpDesc').value.trim(),
            }});
            document.getElementById('rpPlate').value = '';
            document.getElementById('rpDesc').value = '';
            window.toast('Report filed.', 'success');
            loadReports();
        } catch (err) { window.toast(err.message, 'error'); }
    });
    const filter = document.getElementById('reportFilter');
    if (filter) filter.addEventListener('change', loadReports);

    function refreshAll() {
        loadOverrides();
        if (typeof refreshSnapshot === 'function') refreshSnapshot();
    }

    document.addEventListener('DOMContentLoaded', async () => {
        try {
            ZONES = await window.apiJson('/api/zones');
            ['ovZone', 'rpZone'].forEach(id => { const el = document.getElementById(id); if (el) el.innerHTML = zoneOptions(); });
        } catch (e) { /* lists stay empty; the API error is visible elsewhere */ }
        const cat = document.getElementById('rpCategory');
        if (cat) {
            try {
                const cats = await window.apiJson('/api/reports/categories');
                cat.innerHTML = cats.map(c => `<option value="${esc(c.id)}">${esc(c.label)}</option>`).join('');
            } catch (e) { /* ignore */ }
        }
        loadOverrides();
        loadReports();
        setInterval(loadReports, 30000);
    });
})();
