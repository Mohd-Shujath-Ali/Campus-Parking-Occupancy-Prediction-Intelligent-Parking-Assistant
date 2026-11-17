'use strict';

let currentHorizon = 15;

function statusColor(status) {
    if (status === 'AVAILABLE')    return { badge: 'bg-emerald-50 text-emerald-700 border-emerald-200', dot: 'bg-emerald-500' };
    if (status === 'FILLING_FAST') return { badge: 'bg-amber-50   text-amber-700   border-amber-200',   dot: 'bg-amber-500'   };
    return                                { badge: 'bg-rose-50    text-rose-700    border-rose-200',    dot: 'bg-rose-500'    };
}

function statusLabel(status) {
    return status === 'AVAILABLE' ? 'Available' : status === 'FILLING_FAST' ? 'Filling Fast' : 'Saturated';
}

function setHorizon(mins) {
    currentHorizon = mins;
    document.querySelectorAll('.horizon-btn').forEach(btn => {
        const h = parseInt(btn.getAttribute('data-h'));
        btn.classList.toggle('active-horizon', h === mins);
    });
    fetchRecommendations();
}

async function fetchRecommendations() {
    const dest = document.getElementById('destinationSelect')?.value || 'Block_26_CSE';
    const isTrans = document.getElementById('chkClassTransition')?.checked || false;
    const isExam  = document.getElementById('chkExamDay')?.checked || false;

    const list = document.getElementById('recommendationsList');
    if (list) {
        list.innerHTML = `
            <div class="p-10 text-center text-slate-400 bg-white rounded-2xl border border-slate-100">
                <i class="fa-solid fa-spinner fa-spin text-2xl mb-2 text-orange-600"></i>
                <p class="text-sm">Calculating real-time recommendations for ${dest} (+${currentHorizon}m)…</p>
            </div>`;
    }

    try {
        const params = new URLSearchParams({
            destination: dest,
            horizon_minutes: currentHorizon,
            is_class_transition: isTrans,
            is_exam_day: isExam
        });
        const res = await fetch(`/api/recommendations?${params}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
        const data = await res.json();
        renderRecommendations(data);
    } catch (err) {
        if (list) {
            list.innerHTML = `
                <div class="p-6 text-center text-rose-600 bg-rose-50 rounded-2xl border border-rose-200 text-sm">
                    <i class="fa-solid fa-triangle-exclamation text-xl mb-1"></i>
                    <p class="font-bold">Failed to load recommendations</p>
                    <p class="text-xs text-rose-500 mt-1">${err.message}</p>
                </div>`;
        }
    }
}

function renderRecommendations(data) {
    const list = document.getElementById('recommendationsList');
    const tipBox = document.getElementById('smartTipBox');
    const tipTxt = document.getElementById('smartTipText');
    const meta   = document.getElementById('resultMeta');

    if (meta) meta.textContent = `${data.recommended_zones.length} grounds evaluated · +${data.horizon_minutes} min forecast`;

    if (tipBox && tipTxt) {
        if (data.counterfactual_tip) {
            tipTxt.textContent = data.counterfactual_tip;
            tipBox.classList.remove('hidden');
        } else {
            tipBox.classList.add('hidden');
        }
    }

    if (!list) return;

    if (!data.recommended_zones || data.recommended_zones.length === 0) {
        list.innerHTML = '<div class="p-6 text-center text-slate-400">No parking grounds found.</div>';
        return;
    }

    list.innerHTML = data.recommended_zones.map((z, idx) => {
        const sc = statusColor(z.status);
        const occPct = Math.round(z.occupancy_rate * 100);
        const isBest = idx === 0;
        const satRiskPct = Math.round((z.saturation_probability || 0) * 100);

        return `
        <div class="bg-white rounded-2xl border ${isBest ? 'border-orange-500 ring-2 ring-orange-100 shadow-md' : 'border-slate-200 shadow-sm'} p-4 transition-all hover:border-orange-300">
            <div class="flex flex-col sm:flex-row justify-between sm:items-center gap-2">
                <div class="flex items-start space-x-3">
                    <span class="w-7 h-7 rounded-full ${isBest ? 'bg-orange-600 text-white' : 'bg-slate-100 text-slate-600'} font-black text-xs flex items-center justify-center shrink-0">
                        ${idx + 1}
                    </span>
                    <div>
                        <div class="flex items-center space-x-2">
                            <h4 class="font-bold text-slate-900 text-sm">${z.name}</h4>
                            ${isBest ? '<span class="text-[10px] font-black uppercase tracking-wider bg-orange-100 text-orange-700 px-2 py-0.5 rounded-full">Top Pick</span>' : ''}
                        </div>
                        <p class="text-xs text-slate-500 font-medium">${z.category} · ${z.zone_id}</p>
                    </div>
                </div>

                <div class="flex items-center space-x-2 self-end sm:self-auto">
                    <span class="inline-flex items-center text-xs px-2.5 py-1 rounded-full font-bold border ${sc.badge}">
                        <span class="w-1.5 h-1.5 rounded-full ${sc.dot} mr-1.5"></span>
                        ${statusLabel(z.status)}
                    </span>
                    <button onclick="dispatchToZone('${z.zone_id}', '${z.name.replace(/'/g, "\\\'")}')" 
                        id="btnNav_${z.zone_id}"
                        class="px-2.5 py-1 bg-orange-600 hover:bg-orange-700 text-white rounded-lg text-xs font-bold transition-all shadow-xs flex items-center space-x-1">
                        <i class="fa-solid fa-location-arrow text-[10px]"></i>
                        <span>Navigate</span>
                    </button>
                </div>
            </div>

            <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3 pt-3 border-t border-slate-100 text-xs">
                <div>
                    <span class="text-slate-400 block text-[10px] uppercase font-bold">Available Spots</span>
                    <span class="font-black text-sm ${z.available_spaces > 20 ? 'text-emerald-600' : z.available_spaces > 5 ? 'text-amber-600' : 'text-rose-600'}">
                        ${z.available_spaces} <span class="text-[10px] font-normal text-slate-400">/ ${z.operational_capacity}</span>
                    </span>
                </div>
                <div>
                    <span class="text-slate-400 block text-[10px] uppercase font-bold">Predicted Occupancy</span>
                    <span class="font-bold text-slate-800 text-sm">${occPct}%</span>
                    <div class="w-full bg-slate-100 rounded-full h-1.5 mt-1">
                        <div class="h-1.5 rounded-full ${occPct > 85 ? 'bg-rose-500' : occPct > 65 ? 'bg-amber-500' : 'bg-emerald-500'}" style="width: ${Math.min(100, occPct)}%"></div>
                    </div>
                </div>
                <div>
                    <span class="text-slate-400 block text-[10px] uppercase font-bold">Walking Distance</span>
                    <span class="font-bold text-slate-800 text-sm">${Math.round(z.walking_distance_meters)} m</span>
                    <span class="text-slate-400 text-[10px] block">~${z.walking_time_minutes} min walk</span>
                </div>
                <div>
                    <span class="text-slate-400 block text-[10px] uppercase font-bold">Saturation Risk</span>
                    <span class="font-bold ${satRiskPct > 70 ? 'text-rose-600' : 'text-slate-700'} text-sm">${satRiskPct}%</span>
                    <span class="text-slate-400 text-[10px] block">P(Occ ≥ 90%)</span>
                </div>
            </div>
        </div>`;
    }).join('');
}

async function dispatchToZone(zoneId, zoneName) {
    const btn = document.getElementById(`btnNav_${zoneId}`);
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<i class="fa-solid fa-circle-check text-emerald-300"></i> Locked`;
        btn.className = 'px-2.5 py-1 bg-emerald-600 text-white rounded-lg text-xs font-bold transition-all shadow-xs flex items-center space-x-1';
    }
    try {
        const res = await fetch('/api/navigate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ zone_id: zoneId })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        
        // Show non-blocking dispatch notification
        const tipBox = document.getElementById('smartTipBox');
        const tipTxt = document.getElementById('smartTipText');
        if (tipBox && tipTxt) {
            tipTxt.textContent = `Route Locked to ${zoneName}: 1 spot reserved in-flight against congestion shockwaves (active in-flight vehicles: ${data.active_in_flight_count}).`;
            tipBox.classList.remove('hidden');
        }
    } catch (err) {
        console.error('Dispatch error:', err);
    }
}
window.dispatchToZone = dispatchToZone;

document.addEventListener('DOMContentLoaded', () => {
    fetchRecommendations();
});
