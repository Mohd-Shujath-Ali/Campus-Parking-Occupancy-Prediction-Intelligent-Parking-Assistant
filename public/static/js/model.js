'use strict';

const esc = window.escapeHtml;
const fmt = (v, d = 2) => v == null ? '—' : Number(v).toFixed(d);
const pct = v => v == null ? '—' : `${Math.round(v * 100)}%`;
const SOURCE = { camera: ['Camera', 'bg-emerald-50 text-emerald-700 border-emerald-200'],
                 simulated: ['Demo', 'bg-amber-50 text-amber-700 border-amber-200'],
                 profile: ['Estimate', 'bg-slate-100 text-slate-600 border-slate-300'] };
const CONF = { high: 'text-emerald-600', medium: 'text-amber-600', low: 'text-rose-600', demo: 'text-amber-600' };
const REGIME = { 'normal weekday': 'Normal weekday', weekend: 'Weekend', holiday: 'Holiday', 'exam day': 'Exam day', event: 'Event' };

function card(label, value, sub) {
    return `<div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
        <span class="text-xs font-bold text-slate-500 uppercase tracking-wider">${esc(label)}</span>
        <div class="text-lg font-black text-slate-900 mt-1 break-words">${value}</div>
        <span class="text-xs text-slate-500">${sub}</span></div>`;
}

async function loadModel() {
    let d;
    try { d = await window.apiJson('/api/model/status'); }
    catch (e) { window.toast(e.message, 'error'); return; }

    const m = d.meta || {};
    document.getElementById('syntheticBanner').classList.toggle('hidden', m.data_source !== 'synthetic');
    document.getElementById('metaCards').innerHTML = d.loaded ? [
        card('Model version', esc(m.version || '—'), `trained ${esc(window.formatWhen(m.trained_at))}`),
        card('Trained on', esc((m.data_source || '—').toUpperCase()), `${(m.rows || 0).toLocaleString()} rows · ${m.zones || '?'} grounds`),
        card('Data range', `${esc(String(m.data_from || '').slice(0, 10))} → ${esc(String(m.data_to || '').slice(0, 10))}`, `${m.features || '?'} features`),
        card('Uncertainty', esc(m.intervals === 'cqr' ? 'CQR intervals' : 'Calibrated residuals'), '80% range and free-slot chance'),
    ].join('') : card('Model', '<span class="text-rose-600">Not loaded</span>', esc(d.error || 'Run python -m models.train'));

    // data sources
    const modeChip = document.getElementById('modeChip');
    const mode = d.data_mode.mode;
    modeChip.textContent = { live: 'Live camera data', demo: 'Demo data (no camera has reported)', estimated: 'Estimates (cameras silent)' }[mode] || mode;
    modeChip.className = 'text-xs font-bold px-3 py-1 rounded-full border self-start ' +
        ({ live: 'bg-emerald-50 text-emerald-700 border-emerald-200', demo: 'bg-amber-50 text-amber-700 border-amber-200', estimated: 'bg-rose-50 text-rose-700 border-rose-200' }[mode] || '');
    document.getElementById('sourcesBody').innerHTML = d.zones.map(z => {
        const [label, tone] = SOURCE[z.count_source] || [z.count_source, ''];
        return `<tr class="hover:bg-slate-50">
            <td class="px-4 py-2.5 font-medium">${esc(z.name)}</td>
            <td class="px-4 py-2.5"><span class="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full border ${tone}">${label}</span></td>
            <td class="px-4 py-2.5">${esc(z.camera_status)}${z.frozen ? ' <span class="text-amber-600 font-bold">(frozen?)</span>' : ''}</td>
            <td class="px-4 py-2.5 font-mono">${z.count_source === 'camera' ? Math.round(z.age_seconds / 60) + ' min' : '—'}</td>
            <td class="px-4 py-2.5 font-mono font-bold">${z.current_count}</td>
            <td class="px-4 py-2.5 font-bold ${CONF[z.confidence] || ''}">${esc(z.confidence)}</td></tr>`;
    }).join('');

    // live accuracy
    const la = d.live_accuracy || {};
    document.getElementById('liveWindow').textContent = `Last ${la.window_days} days · ${la.scored_forecasts || 0} forecasts scored`;
    const dr = la.drift || {};
    const chip = document.getElementById('driftChip');
    chip.textContent = { ok: 'On track', drift: 'Drift: consider retraining', collecting: 'Collecting data', no_data: 'No live data yet' }[dr.status] || dr.status;
    chip.className = 'text-xs font-bold px-3 py-1 rounded-full border self-start ' +
        ({ ok: 'bg-emerald-50 text-emerald-700 border-emerald-200', drift: 'bg-rose-50 text-rose-700 border-rose-200' }[dr.status] || 'bg-slate-100 text-slate-600 border-slate-300');
    document.getElementById('driftDetail').textContent = dr.detail || '';
    const rows = Object.entries(la.horizons || {});
    document.getElementById('liveBody').innerHTML = rows.length ? rows.map(([k, r]) => `<tr>
        <td class="px-4 py-2.5 font-mono font-bold">+${k.slice(1)} min</td><td class="px-4 py-2.5">${r.n}</td>
        <td class="px-4 py-2.5 font-mono">${fmt(r.MAE)}</td><td class="px-4 py-2.5 font-mono text-slate-500">${fmt(r.training_MAE)}</td>
        <td class="px-4 py-2.5 font-mono font-bold ${r.mae_vs_training > 1.5 ? 'text-rose-600' : ''}">${r.mae_vs_training != null ? r.mae_vs_training + '×' : '—'}</td>
        <td class="px-4 py-2.5 font-mono">${r.bias > 0 ? '+' : ''}${fmt(r.bias)}</td>
        <td class="px-4 py-2.5 font-mono">${pct(r.interval_coverage)}</td><td class="px-4 py-2.5 font-mono">${fmt(r.free_slot_brier, 4)}</td></tr>`).join('')
        : '<tr><td colspan="8" class="px-4 py-6 text-center text-slate-400">Nothing scored yet. Forecasts made from camera counts are scored automatically once the real count for their target time arrives.</td></tr>';

    // holdout
    const bt = Object.entries(d.backtest || {}).filter(([k, v]) => k.startsWith('H') && v && v.MAE != null);
    document.getElementById('holdoutBody').innerHTML = bt.length ? bt.map(([k, r]) => `<tr>
        <td class="px-4 py-2.5 font-mono font-bold">+${k.slice(1)} min</td><td class="px-4 py-2.5 font-mono font-bold text-orange-700">${fmt(r.MAE)}</td>
        <td class="px-4 py-2.5 font-mono text-slate-500">${fmt(r.persistence_MAE)}</td>
        <td class="px-4 py-2.5 font-mono font-bold text-emerald-600">${r.skill_vs_persistence_pct}%</td>
        <td class="px-4 py-2.5 font-mono">${pct(r.interval_coverage)}</td>
        <td class="px-4 py-2.5 font-mono">${fmt(r.free_slot_brier, 4)} <span class="text-slate-400">vs ${fmt(r.free_slot_brier_climatology, 4)}</span></td></tr>`).join('')
        : '<tr><td colspan="6" class="px-4 py-6 text-center text-slate-400">No training metrics recorded.</td></tr>';
    const reg = (d.regime_metrics || []).filter(r => r.horizon === 'H60');
    document.getElementById('regimeBody').innerHTML = reg.length ? reg.map(r => `<tr>
        <td class="px-4 py-2.5 font-medium">${esc(REGIME[r.regime] || r.regime)}</td><td class="px-4 py-2.5 font-mono">${(r.rows || 0).toLocaleString()}</td>
        <td class="px-4 py-2.5 font-mono font-bold text-orange-700">${fmt(r.MAE)}</td><td class="px-4 py-2.5 font-mono text-slate-500">${fmt(r.persistence_MAE)}</td>
        <td class="px-4 py-2.5 font-mono font-bold text-emerald-600">${r.skill_vs_persistence_pct}%</td></tr>`).join('')
        : '<tr><td colspan="5" class="px-4 py-6 text-center text-slate-400">No breakdown available.</td></tr>';
}

document.addEventListener('DOMContentLoaded', () => { loadModel(); setInterval(loadModel, 60000); });
