'use strict';

let currentHorizon = 15;

function statusColor(status) {
    if (status === 'AVAILABLE')    return { badge: 'bg-emerald-50 text-emerald-700 border-emerald-200', dot: 'bg-emerald-500' };
    if (status === 'FILLING_FAST') return { badge: 'bg-amber-50   text-amber-700   border-amber-200',   dot: 'bg-amber-500'   };
    if (status === 'CLOSED')       return { badge: 'bg-slate-200  text-slate-700   border-slate-300',   dot: 'bg-slate-500'   };
    return                                { badge: 'bg-rose-50    text-rose-700    border-rose-200',    dot: 'bg-rose-500'    };
}

function statusLabel(status) {
    return { AVAILABLE: 'Available', FILLING_FAST: 'Filling Fast', SATURATED: 'Saturated',
             CLOSED: 'Closed', FULL: 'Full (warden)' }[status] || status;
}

const isBlocked = z => z.status === 'CLOSED' || z.status === 'FULL';
const escH = s => (window.escapeHtml ? window.escapeHtml(s) : String(s));

// ─────────────────────────── Destinations (campus blocks) ──
let BLOCK_INDEX = {};

async function loadDestinations() {
    const sel = document.getElementById('destinationSelect');
    if (!sel || !window.CURRENT_USER) return;
    try {
        const res = await fetch(window.apiUrl ? window.apiUrl('/api/blocks') : '/api/blocks');
        if (!res.ok) return;
        const blocks = await res.json();
        BLOCK_INDEX = Object.fromEntries(blocks.map(b => [b.id, b]));
        const me = window.CURRENT_USER;
        const home = me.home_block ? BLOCK_INDEX[me.home_block] : null;
        const groups = {};
        blocks.forEach(b => (groups[b.group] = groups[b.group] || []).push(b));
        sel.innerHTML = (home ? `<option value="">My home block: ${escH(home.name)}</option>` : '') +
            Object.keys(groups).sort().map(g =>
                `<optgroup label="${escH(g)}">${groups[g].map(b => `<option value="${escH(b.id)}">${escH(b.name)}</option>`).join('')}</optgroup>`
            ).join('');
        if (!home) sel.value = 'Block_26_CSE';
    } catch (e) { /* keep the server-rendered default option */ }
}

function etaText(mins) {
    return mins <= 5 ? 'Now' : mins < 60 ? `${mins} min` : mins % 60 === 0 ? `${mins / 60} h` : `${Math.floor(mins / 60)} h ${mins % 60} min`;
}

// Either a relative arrival (currentHorizon minutes from now) or a clock time (arriveAt 'HH:MM', next occurrence).
let arriveAt = null;

function syncHorizonControls() {
    document.querySelectorAll('.horizon-btn:not(.clock-btn)').forEach(btn => {
        btn.classList.toggle('active-horizon', !arriveAt && parseInt(btn.getAttribute('data-h')) === currentHorizon);
    });
    document.querySelectorAll('.clock-btn').forEach(btn => {
        btn.classList.toggle('active-horizon', btn.getAttribute('data-t') === arriveAt);
    });
    const slider = document.getElementById('etaSlider');
    if (slider) {
        slider.value = Math.min(currentHorizon, 120);
        slider.classList.toggle('opacity-40', !!arriveAt);
    }
    const label = document.getElementById('etaLabel');
    if (label) label.textContent = arriveAt ? clockText(arriveAt) : etaText(currentHorizon);
}

function clockText(hhmm) {
    const [h, m] = hhmm.split(':').map(Number);
    return `${h % 12 || 12}${m ? ':' + String(m).padStart(2, '0') : ''} ${h < 12 ? 'AM' : 'PM'}`;
}

function setHorizon(mins) {
    currentHorizon = mins;
    arriveAt = null;
    syncHorizonControls();
    fetchRecommendations();
}

function setArrivalTime(hhmm) {
    arriveAt = arriveAt === hhmm ? null : hhmm;      // click again to go back to "in N minutes"
    syncHorizonControls();
    fetchRecommendations();
}

// The server resolves the arrival time in campus time and returns it; show exactly what the forecast is for.
function renderArrival(data) {
    const el = document.getElementById('arrivalLabel');
    if (!el || !data.arrival_time) return;
    const [d, t] = data.arrival_time.split('T');
    const when = new Date(`${d}T00:00:00`);
    const today = new Date(); today.setHours(0, 0, 0, 0);
    const days = Math.round((when - today) / 86400000);
    const day = days === 0 ? 'Today' : days === 1 ? 'Tomorrow' : when.toLocaleDateString([], { weekday: 'long', day: 'numeric', month: 'short' });
    const basis = { live: 'from live counts', blended: 'live counts blended with the usual pattern', calendar: 'from the campus calendar and usual pattern' }[data.forecast_basis] || '';
    el.textContent = `Forecast for ${day}, ${clockText(t)} (in ${etaText(data.horizon_minutes)})${basis ? ' · ' + basis : ''}`;
}

let etaTimer = null;
function onEtaInput(e) {
    arriveAt = null;
    currentHorizon = parseInt(e.target.value, 10);
    syncHorizonControls();
    clearTimeout(etaTimer);
    etaTimer = setTimeout(fetchRecommendations, 250);   // wait for the slider to settle
}

// ─────────────────────────── Campus conditions (from the calendar) and data source ──
function condChip(text, tone) {
    const tones = { amber: 'bg-amber-50 text-amber-800 border-amber-200', sky: 'bg-sky-50 text-sky-800 border-sky-200',
                    violet: 'bg-violet-50 text-violet-800 border-violet-200', slate: 'bg-white text-slate-600 border-slate-200' };
    return `<span class="px-2 py-1 rounded-lg border ${tones[tone] || tones.slate}">${escH(text)}</span>`;
}

function renderConditions(c) {
    const box = document.getElementById('conditionsBox');
    if (!box || !c) return;
    const chips = [];
    if (c.holiday) chips.push(condChip(`Holiday${c.holiday_names?.length ? ': ' + c.holiday_names[0] : ''}`, 'violet'));
    else if (c.weekend) chips.push(condChip('Weekend', 'slate'));
    if (c.exam_day) chips.push(condChip(`Exam day${c.exam_names?.length ? ': ' + c.exam_names[0] : ''}`, 'amber'));
    if (c.bell_window) chips.push(condChip('Lecture bell window', 'amber'));
    (c.events_now || []).forEach(e => chips.push(condChip(`Event: ${e.name}`, 'sky')));
    (c.events_soon || []).forEach(e => chips.push(condChip(`Soon: ${e.name}`, 'sky')));
    box.innerHTML = chips.length ? chips.join('') : condChip('Normal day', 'slate');
}

function renderDataMode(mode) {
    const banner = document.getElementById('dataModeBanner');
    const dot = document.getElementById('heroDot');
    const txt = document.getElementById('heroStatusText');
    const msgs = {
        demo:      ['bg-amber-50 border-amber-200 text-amber-900', 'fa-flask', 'Demo data: no camera has reported yet, so the counts below are simulated and forecast from a model trained on synthetic data. Treat them as an illustration, not real availability.'],
        estimated: ['bg-slate-50 border-slate-300 text-slate-700', 'fa-chart-simple', 'No live camera counts right now. Showing the normal pattern for this time of day; actual availability may differ.'],
        offline:   ['bg-rose-50 border-rose-200 text-rose-800', 'fa-plug-circle-xmark', 'Cannot reach the server. Showing a rough in-browser estimate only.'],
    };
    if (dot && txt) {
        dot.className = 'w-2.5 h-2.5 rounded-full ' + ({ live: 'bg-emerald-400 animate-pulse', demo: 'bg-amber-300', estimated: 'bg-slate-300', offline: 'bg-rose-400' }[mode] || 'bg-slate-300');
        txt.textContent = { live: '12 grounds · live camera counts', demo: '12 grounds · demo data', estimated: '12 grounds · estimates', offline: 'Server unreachable' }[mode] || '12 grounds';
    }
    if (!banner) return;
    const m = msgs[mode];
    if (!m) { banner.classList.add('hidden'); return; }
    banner.className = `text-sm rounded-xl px-4 py-3 border flex items-start space-x-2.5 ${m[0]}`;
    document.getElementById('dataModeIcon').className = `fa-solid ${m[1]} mt-0.5`;
    document.getElementById('dataModeText').textContent = m[2];
}

const SOURCE_BADGE = {
    camera:    null,
    simulated: ['bg-amber-50 text-amber-700 border-amber-200', 'Demo'],
    profile:   ['bg-slate-100 text-slate-600 border-slate-300', 'Estimate'],
};
const CONF_LABEL = { high: 'High confidence', medium: 'Medium confidence', low: 'Low confidence' };

// ─────────────────────────── Campus Topology & Client Fallback Engine ──
const CAMPUS_ZONES = [
    {"id": "ZONE_MAINGATE", "name": "LPU Main Gate Parking", "category": "Campus Entrance & Transit", "operational_capacity": 240, "distances": {"Block_1": 750, "Block_26_CSE": 650, "Block_57_Agri": 1200, "Architecture_Block": 800, "MB_Food_Square": 700, "Law_Block": 900, "Central_Mess_2": 1100, "Sports_Ground": 850}},
    {"id": "ZONE_BLOCK1", "name": "Block 1 Parking", "category": "Academic & Administrative Core", "operational_capacity": 140, "distances": {"Block_1": 30, "Block_26_CSE": 280, "Block_57_Agri": 800, "Architecture_Block": 350, "MB_Food_Square": 420, "Law_Block": 600, "Central_Mess_2": 750, "Sports_Ground": 500}},
    {"id": "ZONE_ARCH", "name": "Architecture Block Parking", "category": "Design & Applied Arts", "operational_capacity": 120, "distances": {"Block_1": 350, "Block_26_CSE": 400, "Block_57_Agri": 700, "Architecture_Block": 40, "MB_Food_Square": 180, "Law_Block": 650, "Central_Mess_2": 600, "Sports_Ground": 450}},
    {"id": "ZONE_MBFS", "name": "MB Food Square Parking", "category": "Dining, Hospital & Social Hub", "operational_capacity": 200, "distances": {"Block_1": 420, "Block_26_CSE": 450, "Block_57_Agri": 650, "Architecture_Block": 180, "MB_Food_Square": 30, "Law_Block": 700, "Central_Mess_2": 550, "Sports_Ground": 400}},
    {"id": "ZONE_LAW", "name": "Law Block Parking", "category": "School of Law & Management", "operational_capacity": 130, "distances": {"Block_1": 600, "Block_26_CSE": 500, "Block_57_Agri": 900, "Architecture_Block": 650, "MB_Food_Square": 700, "Law_Block": 40, "Central_Mess_2": 450, "Sports_Ground": 600}},
    {"id": "ZONE_GH5", "name": "GH 5 Parking", "category": "Residential (Hostels)", "operational_capacity": 160, "distances": {"Block_1": 650, "Block_26_CSE": 550, "Block_57_Agri": 950, "Architecture_Block": 680, "MB_Food_Square": 720, "Law_Block": 150, "Central_Mess_2": 400, "Sports_Ground": 650}},
    {"id": "ZONE_MAINPARK_NORTH", "name": "LPU Main Parking (North - Block 26/39 CSE)", "category": "Core Computing, Admission & Innovation", "operational_capacity": 320, "distances": {"Block_1": 280, "Block_26_CSE": 50, "Block_57_Agri": 750, "Architecture_Block": 400, "MB_Food_Square": 450, "Law_Block": 500, "Central_Mess_2": 700, "Sports_Ground": 400}},
    {"id": "ZONE_MAINPARK_SOUTH", "name": "LPU Main Parking (South - CSE / Innovation 39)", "category": "Core Computing & Labs", "operational_capacity": 300, "distances": {"Block_1": 300, "Block_26_CSE": 60, "Block_57_Agri": 700, "Architecture_Block": 420, "MB_Food_Square": 480, "Law_Block": 520, "Central_Mess_2": 680, "Sports_Ground": 380}},
    {"id": "ZONE_CENTRAL_MESS2", "name": "Central Mess 2 Parking", "category": "Dining & Hostel Block 41-D", "operational_capacity": 110, "distances": {"Block_1": 750, "Block_26_CSE": 700, "Block_57_Agri": 600, "Architecture_Block": 600, "MB_Food_Square": 550, "Law_Block": 450, "Central_Mess_2": 30, "Sports_Ground": 500}},
    {"id": "ZONE_BH2", "name": "Boys Hostel 2 Parking", "category": "Residential & Sports Arena", "operational_capacity": 135, "distances": {"Block_1": 800, "Block_26_CSE": 650, "Block_57_Agri": 800, "Architecture_Block": 750, "MB_Food_Square": 700, "Law_Block": 650, "Central_Mess_2": 500, "Sports_Ground": 350}},
    {"id": "ZONE_SPORTS", "name": "LPU Sports Ground Parking", "category": "Athletics & Mechanical Link", "operational_capacity": 145, "distances": {"Block_1": 500, "Block_26_CSE": 400, "Block_57_Agri": 500, "Architecture_Block": 450, "MB_Food_Square": 400, "Law_Block": 600, "Central_Mess_2": 500, "Sports_Ground": 20}},
    {"id": "ZONE_BLOCK57", "name": "LPU Block 57 Parking", "category": "Agriculture & Mechanical Core", "operational_capacity": 185, "distances": {"Block_1": 800, "Block_26_CSE": 750, "Block_57_Agri": 30, "Architecture_Block": 700, "MB_Food_Square": 650, "Law_Block": 900, "Central_Mess_2": 600, "Sports_Ground": 500}}
];

function computeClientSideRecommendations(destination, horizonMinutes, isTrans, isExam) {
    const now = new Date();
    const hour = now.getHours() + now.getMinutes() / 60.0;
    const isPeak = (hour >= 8 && hour <= 10) || (hour >= 11 && hour <= 14) || (hour >= 16 && hour <= 18);
    const baseOcc = isPeak ? 0.72 : 0.38;
    const surgeMultiplier = isTrans ? 1.15 : (isExam ? 1.10 : 1.0);

    const scored = CAMPUS_ZONES.map(z => {
        const cap = z.operational_capacity;
        let occRate = Math.min(0.96, Math.max(0.15, baseOcc * surgeMultiplier + (z.id.includes('MAIN') ? 0.12 : -0.05)));
        const predCount = Math.round(cap * occRate);
        const availSpaces = Math.max(0, cap - predCount);
        const satProb = occRate >= 0.88 ? 0.85 : (occRate >= 0.75 ? 0.45 : 0.08);

        const dist = (z.distances && z.distances[destination]) ? z.distances[destination] : 500;
        const walkMins = Math.round((dist / 80.0) * 10) / 10;

        const availScore = Math.max(0, 1.0 - occRate);
        const distScore = Math.max(0, 1.0 - (dist / 1200.0));
        const satPenalty = satProb > 0.70 ? 0.5 : 0.0;

        const totalScore = (0.50 * availScore) + (0.35 * distScore) - satPenalty;
        const boundedScore = Math.round(Math.max(0.01, totalScore) * 1000) / 1000;

        const status = occRate < 0.75 ? "AVAILABLE" : (occRate < 0.90 ? "FILLING_FAST" : "SATURATED");

        return {
            zone_id: z.id,
            name: z.name,
            category: z.category,
            score: boundedScore,
            predicted_count: predCount,
            operational_capacity: cap,
            available_spaces: availSpaces,
            occupancy_rate: Math.round(occRate * 1000) / 1000,
            saturation_probability: satProb,
            walking_distance_meters: dist,
            walking_time_minutes: walkMins,
            status: status,
            confidence_level: "HIGH_CONFIDENCE_ML"
        };
    });

    scored.sort((a, b) => b.score - a.score);

    let tip = null;
    if (scored.length && scored[0].status !== "AVAILABLE") {
        tip = "Smart Arrival Tip: Arriving 15 minutes after peak bell hour reduces congestion by ~22%.";
    } else if (isPeak) {
        tip = `Peak lecture hour active (${now.getHours()}:00). Lecture surge mode active — spot buffered.`;
    }

    return {
        destination: destination,
        horizon_minutes: horizonMinutes,
        recommended_zones: scored,
        counterfactual_tip: tip,
        is_client_fallback: true
    };
}

async function fetchRecommendations(silent) {
    silent = silent === true;      // background refresh: no spinner, so the list does not flicker
    // Empty value = "my home block": the server resolves it from the signed-in profile.
    const selected = document.getElementById('destinationSelect')?.value ?? 'Block_26_CSE';
    const me = window.CURRENT_USER;
    const homeId = !selected && me ? me.home_block : null;
    const label = selected ? (BLOCK_INDEX[selected]?.name || selected) : (BLOCK_INDEX[homeId]?.name || 'your home block');
    const list = document.getElementById('recommendationsList');
    if (list && !silent) {
        list.innerHTML = `
            <div class="p-10 text-center text-slate-400 bg-white rounded-2xl border border-slate-100">
                <i class="fa-solid fa-spinner fa-spin text-2xl mb-2 text-orange-600"></i>
                <p class="text-sm">Finding the nearest grounds to ${escH(label)} (+${currentHorizon}m)…</p>
            </div>`;
    }

    let res;
    try {
        // Exam days, holidays, events and bell windows come from the campus calendar on the server.
        const params = new URLSearchParams({ horizon_minutes: Math.min(currentHorizon, 120) });
        if (arriveAt) params.set('arrive_at', arriveAt);
        if (selected) params.set('destination', selected);
        const url = (window.apiUrl ? window.apiUrl('/api/recommendations') : '/api/recommendations') + `?${params}`;
        res = await fetch(url);
    } catch (err) {
        res = null;   // network failure: fall through to the in-browser engine
    }

    if (res && res.status >= 400 && res.status < 500) {
        // A real answer from the server (not signed in, no home block, unknown block…): show it.
        const body = await res.json().catch(() => ({}));
        if (list && res.status !== 401) {
            list.innerHTML = `<div class="p-8 text-center bg-white rounded-2xl border border-amber-200 text-amber-800 text-sm">
                <i class="fa-solid fa-circle-info text-xl mb-2 text-amber-500"></i>
                <p>${escH(typeof body.detail === 'string' ? body.detail : 'Could not get recommendations.')}</p></div>`;
        }
        return;
    }
    if (res && res.ok) {
        renderRecommendations(await res.json());
        return;
    }
    console.warn('Backend API unreachable or failing. Running in-browser client ranking engine.');
    const anchor = selected ? (BLOCK_INDEX[selected]?.anchor || selected) : (BLOCK_INDEX[homeId]?.anchor || 'Block_26_CSE');
    const fallback = computeClientSideRecommendations(anchor, currentHorizon, false, false);
    fallback.data_mode = 'offline';
    renderRecommendations(fallback);
}

function renderRecommendations(data) {
    const list = document.getElementById('recommendationsList');
    const tipBox = document.getElementById('smartTipBox');
    const tipTxt = document.getElementById('smartTipText');
    const meta   = document.getElementById('resultMeta');

    if (meta) {
        const engineLabel = data.is_client_fallback ? 'rough in-browser estimate' : 'server forecast';
        meta.textContent = `${data.recommended_zones.length} grounds · arriving in ${etaText(data.horizon_minutes)} · ${engineLabel}`;
    }
    renderDataMode(data.data_mode);
    renderConditions(data.conditions);
    renderArrival(data);

    const chip = document.getElementById('homeBlockChip');
    const chipTxt = document.getElementById('homeBlockChipText');
    if (chip && chipTxt && data.requested_block_name) {
        const via = data.uses_adjacent_block ? ` · distances from adjacent ${data.destination.replace(/_/g, ' ')}` : '';
        chipTxt.textContent = `${data.destination_source === 'home_block' ? 'Your block' : 'Heading to'}: ${data.requested_block_name}${via}`;
        chip.classList.remove('hidden');
    }

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
        const blocked = isBlocked(z);
        const isBest = idx === 0 && !blocked;
        const pFreePct = z.p_free == null ? null : Math.round(z.p_free * 100);

        if (blocked) {
            return `
        <div class="bg-slate-50 rounded-2xl border border-slate-200 p-4 opacity-80">
            <div class="flex flex-col sm:flex-row justify-between sm:items-center gap-2">
                <div class="flex items-start space-x-3">
                    <span class="w-7 h-7 rounded-full bg-slate-200 text-slate-500 text-xs flex items-center justify-center shrink-0"><i class="fa-solid fa-ban"></i></span>
                    <div>
                        <h4 class="font-bold text-slate-600 text-sm">${escH(z.name)}</h4>
                        <p class="text-xs text-slate-500">${escH(z.override_reason || 'Not accepting vehicles right now')} · ${Math.round(z.walking_distance_meters)} m away</p>
                    </div>
                </div>
                <span class="inline-flex items-center text-xs px-2.5 py-1 rounded-full font-bold border ${sc.badge} self-end sm:self-auto">
                    <span class="w-1.5 h-1.5 rounded-full ${sc.dot} mr-1.5"></span>${statusLabel(z.status)}
                </span>
            </div>
        </div>`;
        }

        return `
        <div class="bg-white rounded-2xl border ${isBest ? 'border-orange-500 ring-2 ring-orange-100 shadow-md' : 'border-slate-200 shadow-sm'} p-4 transition-all hover:border-orange-300">
            <div class="flex flex-col sm:flex-row justify-between sm:items-center gap-2">
                <div class="flex items-start space-x-3">
                    <span class="w-7 h-7 rounded-full ${isBest ? 'bg-orange-600 text-white' : 'bg-slate-100 text-slate-600'} font-black text-xs flex items-center justify-center shrink-0">
                        ${idx + 1}
                    </span>
                    <div>
                        <div class="flex items-center space-x-2">
                            <h4 class="font-bold text-slate-900 text-sm">${escH(z.name)}</h4>
                            ${isBest ? '<span class="text-[10px] font-black uppercase tracking-wider bg-orange-100 text-orange-700 px-2 py-0.5 rounded-full">Top Pick</span>' : ''}
                            ${SOURCE_BADGE[z.count_source] ? `<span class="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full border ${SOURCE_BADGE[z.count_source][0]}">${SOURCE_BADGE[z.count_source][1]}</span>` : ''}
                        </div>
                        <p class="text-xs text-slate-500 font-medium">${escH(z.category)}${z.confidence && CONF_LABEL[z.confidence] ? ' · ' + CONF_LABEL[z.confidence] : ''}</p>
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
                    <span class="text-slate-400 block text-[10px] uppercase font-bold">Chance of a free slot</span>
                    ${z.p_free == null
                        ? '<span class="font-bold text-slate-400 text-sm">n/a</span>'
                        : `<span class="font-black text-sm ${pFreePct >= 70 ? 'text-emerald-600' : pFreePct >= 35 ? 'text-amber-600' : 'text-rose-600'}">${pFreePct}%</span>
                           <span class="text-slate-400 text-[10px] block">${z.likely_full ? 'likely full on arrival' : 'on arrival'}</span>`}
                </div>
                <div>
                    <span class="text-slate-400 block text-[10px] uppercase font-bold">Expected occupancy</span>
                    <span class="font-bold text-slate-800 text-sm">${occPct}%</span>
                    <div class="w-full bg-slate-100 rounded-full h-1.5 mt-1">
                        <div class="h-1.5 rounded-full ${occPct > 85 ? 'bg-rose-500' : occPct > 65 ? 'bg-amber-500' : 'bg-emerald-500'}" style="width: ${Math.min(100, occPct)}%"></div>
                    </div>
                    ${z.predicted_low != null ? `<span class="text-slate-400 text-[10px] block mt-0.5">${z.predicted_low}–${z.predicted_high} of ${z.operational_capacity} slots (80% range)</span>` : ''}
                </div>
                <div>
                    <span class="text-slate-400 block text-[10px] uppercase font-bold">Free slots expected</span>
                    <span class="font-black text-sm ${z.available_spaces > 20 ? 'text-emerald-600' : z.available_spaces > 5 ? 'text-amber-600' : 'text-rose-600'}">${z.available_spaces}</span>
                    <span class="text-slate-400 text-[10px] block">of ${z.operational_capacity}</span>
                </div>
                <div>
                    <span class="text-slate-400 block text-[10px] uppercase font-bold">Walking distance</span>
                    <span class="font-bold text-slate-800 text-sm">${Math.round(z.walking_distance_meters)} m</span>
                    <span class="text-slate-400 text-[10px] block">~${z.walking_time_minutes} min walk</span>
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
        const url = window.apiUrl ? window.apiUrl('/api/navigate') : '/api/navigate';
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ zone_id: zoneId })
        });
        if (!res.ok) {
            const body = await res.json().catch(() => ({}));
            if (window.toast && typeof body.detail === 'string') window.toast(body.detail, 'warning');
            throw new Error(`HTTP ${res.status}`);
        }
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

document.addEventListener('DOMContentLoaded', async () => {
    document.getElementById('etaSlider')?.addEventListener('input', onEtaInput);
    syncHorizonControls();
    await loadDestinations();
    fetchRecommendations();
    setInterval(() => fetchRecommendations(true), 60000);   // keep the picture fresh while the page stays open
});
