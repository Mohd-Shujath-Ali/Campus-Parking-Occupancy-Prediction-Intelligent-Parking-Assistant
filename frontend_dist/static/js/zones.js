'use strict';

async function loadZonesData() {
    await Promise.all([loadZonesDirectory(), loadHeatmapSnapshot()]);
}

async function loadHeatmapSnapshot() {
    const grid = document.getElementById('heatmapGrid');
    const tsEl = document.getElementById('heatmapTimestamp');
    if (!grid) return;

    try {
        const url = window.apiUrl ? window.apiUrl('/api/snapshot') : '/api/snapshot';
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        
        if (tsEl && data.timestamp) {
            const d = new Date(data.timestamp);
            tsEl.textContent = `Updated: ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
        }

        renderHeatmapGrid(data.zones || []);
    } catch (err) {
        if (grid) {
            grid.innerHTML = `<div class="col-span-full py-3 text-center text-xs text-rose-500">Failed to load real-time heatmap: ${err.message}</div>`;
        }
    }
}

function renderHeatmapGrid(zones) {
    const grid = document.getElementById('heatmapGrid');
    if (!grid) return;

    if (zones.length === 0) {
        grid.innerHTML = '<div class="col-span-full py-3 text-center text-xs text-slate-400">No snapshot data available.</div>';
        return;
    }

    grid.innerHTML = zones.map(z => {
        const occRate = z.occupancy_rate !== undefined ? z.occupancy_rate : (z.predicted_vehicle_count / z.operational_capacity);
        const occPct = Math.round(occRate * 100);
        
        let colorClass = 'border-emerald-300 bg-emerald-50/60 text-emerald-800';
        let barColor = 'bg-emerald-500';
        let badgeColor = 'bg-emerald-100 text-emerald-800';
        let statusText = 'Free';

        if (occRate >= 0.85) {
            colorClass = 'border-rose-300 bg-rose-50/60 text-rose-800';
            barColor = 'bg-rose-500';
            badgeColor = 'bg-rose-100 text-rose-800 font-bold';
            statusText = 'Congested';
        } else if (occRate >= 0.65) {
            colorClass = 'border-amber-300 bg-amber-50/60 text-amber-800';
            barColor = 'bg-amber-500';
            badgeColor = 'bg-amber-100 text-amber-800';
            statusText = 'Moderate';
        }

        if (z.camera_status === 'offline') {
            colorClass = 'border-slate-300 bg-slate-50 text-slate-600';
            badgeColor = 'bg-slate-200 text-slate-600';
            statusText = 'Offline';
        }

        return `
        <div class="rounded-xl border ${colorClass} p-3 flex flex-col justify-between shadow-xs transition-all hover:shadow-sm">
            <div class="flex justify-between items-start">
                <span class="text-[10px] font-mono font-bold truncate max-w-[85px]">${z.zone_id.replace('ZONE_', '')}</span>
                <span class="text-[9px] px-1.5 py-0.5 rounded ${badgeColor}">${statusText}</span>
            </div>
            
            <div class="my-2">
                <div class="flex items-baseline justify-between">
                    <span class="text-lg font-black">${occPct}%</span>
                    <span class="text-[10px] font-mono opacity-80">${z.predicted_vehicle_count}/${z.operational_capacity}</span>
                </div>
                <div class="w-full bg-slate-200/80 rounded-full h-1.5 mt-1 overflow-hidden">
                    <div class="${barColor} h-1.5 rounded-full transition-all duration-500" style="width: ${Math.min(100, occPct)}%"></div>
                </div>
            </div>

            <div class="text-[10px] truncate text-slate-600 font-medium">${z.name}</div>
        </div>`;
    }).join('');
}

async function loadZonesDirectory() {
    try {
        const url = window.apiUrl ? window.apiUrl('/api/zones') : '/api/zones';
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const zones = await res.json();
        renderZones(zones);
        loadAnalysis(zones);
    } catch (err) {
        const grid = document.getElementById('zonesGrid');
        if (grid) {
            grid.innerHTML = `
                <div class="p-6 text-center text-rose-600 bg-rose-50 rounded-2xl border border-rose-200 col-span-full">
                    <p class="font-bold">Failed to load campus grounds</p>
                    <p class="text-xs text-rose-500 mt-1">${err.message}</p>
                </div>`;
        }
    }
}

function renderZones(zones) {
    const grid = document.getElementById('zonesGrid');
    if (!grid) return;

    if (zones.length === 0) {
        grid.innerHTML = '<div class="p-6 text-center text-slate-400 col-span-full">No grounds metadata found.</div>';
        return;
    }

    grid.innerHTML = zones.map(z => {
        const landmarks = (z.nearby_landmarks || []).slice(0, 4);
        const buffer = z.theoretical_capacity - z.operational_capacity;

        return `
        <div class="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 hover:border-orange-300 transition-all space-y-4" data-zone="${z.zone_id}">
            <div class="flex justify-between items-start">
                <div>
                    <span class="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-orange-100 text-orange-800">${z.category}</span>
                    <h4 class="font-bold text-slate-900 text-base mt-1.5">${z.name}</h4>
                    <span class="text-xs font-mono text-slate-400">${z.zone_id}</span>
                </div>
                <span class="font-mono text-xs px-2.5 py-1 bg-slate-100 rounded-lg text-slate-600 font-bold border border-slate-200">${z.camera_id || 'CAM-01'}</span>
            </div>

            <div class="grid grid-cols-2 gap-2 text-center text-xs pt-1">
                <div class="bg-slate-50 p-2.5 rounded-xl border border-slate-100">
                    <span class="text-slate-400 text-[10px] uppercase font-bold block">Operational Cap.</span>
                    <span class="text-slate-900 font-black text-lg">${z.operational_capacity} <span class="text-[10px] font-normal text-slate-400">Cars</span></span>
                </div>
                <div class="bg-slate-50 p-2.5 rounded-xl border border-slate-100">
                    <span class="text-slate-400 text-[10px] uppercase font-bold block">Physical Cap.</span>
                    <span class="text-slate-600 font-bold text-lg">${z.theoretical_capacity} <span class="text-[10px] font-normal text-slate-400">(-${buffer})</span></span>
                </div>
            </div>

            <div data-analysis class="text-xs text-slate-400">Loading usage profile…</div>

            <div>
                <span class="text-[11px] font-bold text-slate-500 uppercase tracking-wider block mb-1.5">Adjacent Landmarks &amp; Hubs</span>
                <div class="flex flex-wrap gap-1.5">
                    ${landmarks.map(l => `<span class="text-[11px] bg-slate-100 text-slate-700 px-2.5 py-1 rounded-lg border border-slate-200">${l}</span>`).join('')}
                </div>
            </div>
        </div>`;
    }).join('');
}

const hh = h => `${String(h).padStart(2, '0')}:00`;

function profileSvg(wd, we) {
    const w = 260, h = 46, pts = v => v.map((x, i) =>
        `${(i / 23 * w).toFixed(1)},${(h - Math.min(x, 1.1) / 1.1 * h).toFixed(1)}`).join(' ');
    return `<svg viewBox="0 0 ${w} ${h}" style="width:100%;height:48px"><polyline fill="none" stroke="#cbd5e1" stroke-width="1.6" points="${pts(we)}"/>` +
           `<polyline fill="none" stroke="#ea580c" stroke-width="2" points="${pts(wd)}"/></svg>`;
}

async function loadAnalysis(zones) {
    await Promise.all(zones.map(async z => {
        const box = document.querySelector(`[data-zone="${z.zone_id}"] [data-analysis]`);
        if (!box) return;
        try {
            const apiPath = `/api/zone-analysis/${z.zone_id}`;
            const res = await fetch(window.apiUrl ? window.apiUrl(apiPath) : apiPath);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const a = await res.json();
            box.className = 'text-xs space-y-1.5';
            box.innerHTML = `
              <span class="text-[11px] font-bold text-slate-500 uppercase tracking-wider block">Typical usage <span class="normal-case font-normal text-slate-400">(${a.data_source} history)</span></span>
              ${profileSvg(a.weekday_profile, a.weekend_profile)}
              <div class="flex justify-between text-[10px] text-slate-400"><span>00h</span><span><span class="text-orange-600 font-bold">━</span> weekday  <span class="text-slate-400 font-bold">━</span> weekend</span><span>23h</span></div>
              <div class="grid grid-cols-3 gap-2 text-center pt-1">
                <div class="bg-slate-50 rounded-lg p-1.5 border border-slate-100"><div class="text-[10px] text-slate-400 uppercase font-bold">Peak</div><div class="font-black text-slate-900">${hh(a.weekday_peak_hour)}</div><div class="text-[10px] text-slate-500">${Math.round(a.weekday_peak_occupancy * 100)}% full</div></div>
                <div class="bg-slate-50 rounded-lg p-1.5 border border-slate-100"><div class="text-[10px] text-slate-400 uppercase font-bold">Saturated</div><div class="font-black text-slate-900">${Math.round(a.share_time_saturated * 100)}%</div><div class="text-[10px] text-slate-500">of the time</div></div>
                <div class="bg-slate-50 rounded-lg p-1.5 border border-slate-100"><div class="text-[10px] text-slate-400 uppercase font-bold">Weekend</div><div class="font-black text-slate-900">${a.weekend_to_weekday_ratio}×</div><div class="text-[10px] text-slate-500">vs weekday</div></div>
              </div>`;
        } catch (err) {
            box.textContent = `Usage profile unavailable (${err.message})`;
        }
    }));
}

document.addEventListener('DOMContentLoaded', loadZonesData);
