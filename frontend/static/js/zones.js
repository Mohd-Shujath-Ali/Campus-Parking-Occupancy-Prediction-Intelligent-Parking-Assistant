'use strict';

async function loadZones() {
    try {
        const res = await fetch('/api/zones');
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
            const res = await fetch(`/api/zone-analysis/${z.zone_id}`);
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

document.addEventListener('DOMContentLoaded', loadZones);
