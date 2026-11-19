'use strict';

let barChart = null;
let snapshotTimer = null;

function statusColor(status) {
    if (status === 'AVAILABLE')    return { badge: 'bg-emerald-50 text-emerald-700 border-emerald-200', dot: 'bg-emerald-500' };
    if (status === 'FILLING_FAST') return { badge: 'bg-amber-50   text-amber-700   border-amber-200',   dot: 'bg-amber-500'   };
    return                                { badge: 'bg-rose-50    text-rose-700    border-rose-200',    dot: 'bg-rose-500'    };
}

function statusLabel(status) {
    return status === 'AVAILABLE' ? 'Available' : status === 'FILLING_FAST' ? 'Filling Fast' : 'Saturated';
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

async function refreshSnapshot() {
    try {
        const url = window.apiUrl ? window.apiUrl('/api/snapshot') : '/api/snapshot';
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        renderSnapshot(data);
    } catch (err) {
        console.error('Failed to load snapshot:', err);
    }
}

function renderSnapshot(data) {
    setText('kpiTotalCap', `${data.total_capacity} `);
    const occPct = Math.round(data.campus_occupancy_rate * 100);
    setText('kpiOccRate', `${occPct}%`);
    setText('kpiTotalOccupied', `${data.total_occupied} `);
    setText('kpiAvailLabel', `${data.total_available} Spaces Free`);

    const peakLbl = document.getElementById('kpiPeakLabel');
    if (peakLbl) {
        if (data.is_peak_hour) {
            peakLbl.textContent = '● Peak Lecture Hours Active';
            peakLbl.className = 'text-xs text-rose-600 font-semibold';
        } else {
            peakLbl.textContent = '○ Steady State Campus Inflow';
            peakLbl.className = 'text-xs text-emerald-600 font-semibold';
        }
    }

    const tEl = document.getElementById('snapshotTime');
    if (tEl && data.timestamp) {
        const d = new Date(data.timestamp);
        tEl.textContent = `Updated: ${d.toLocaleTimeString()}`;
    }

    renderAlerts(data.alerts || []);
    renderTelemetryTable(data.zones || []);
    try {
        renderBarChart(data.zones || []);
    } catch (err) {   // e.g. Chart.js CDN unreachable: keep the tables/alerts working
        console.error('Chart unavailable:', err);
        setText('chartLoading', 'Chart unavailable (Chart.js failed to load).');
    }
}

function renderBarChart(zones) {
    const ctx = document.getElementById('occupancyBarChart')?.getContext('2d');
    if (!ctx || zones.length === 0) return;

    const wrap = document.getElementById('barChartWrap');
    const loading = document.getElementById('chartLoading');
    if (wrap) wrap.classList.remove('hidden');
    if (loading) loading.classList.add('hidden');

    const labels = zones.map(z => z.name.replace(' Parking', '').replace('LPU ', ''));
    const occs   = zones.map(z => z.predicted_vehicle_count);
    const caps   = zones.map(z => z.operational_capacity);

    const colors = zones.map(z => {
        const rate = z.operational_capacity > 0 ? z.predicted_vehicle_count / z.operational_capacity : 0;
        return rate >= 0.90 ? '#ef4444' : rate >= 0.75 ? '#f59e0b' : '#10b981';
    });

    if (barChart) barChart.destroy();

    barChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Occupied Vehicles',
                    data: occs,
                    backgroundColor: colors,
                    borderRadius: 6
                },
                {
                    label: 'Ground Capacity',
                    data: caps,
                    backgroundColor: '#e2e8f0',
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { font: { size: 10 } } },
                y: { beginAtZero: true, title: { display: true, text: 'Vehicle Count', font: { size: 10 } } }
            },
            plugins: {
                legend: { position: 'top', labels: { font: { size: 11 } } }
            }
        }
    });
}

function renderAlerts(alerts) {
    const list = document.getElementById('alertList');
    if (!list) return;
    setText('alertCount', `${alerts.length} active`);
    if (alerts.length === 0) {
        list.innerHTML = '<li class="text-emerald-600 font-semibold">No active alerts: all zones within normal range.</li>';
        return;
    }
    list.replaceChildren(...alerts.map(a => {
        const li = document.createElement('li');
        const crit = a.severity === 'critical';
        li.className = 'flex items-start space-x-2 p-2.5 rounded-xl border ' +
            (crit ? 'bg-rose-50 border-rose-200 text-rose-800' : 'bg-amber-50 border-amber-200 text-amber-800');
        const tag = document.createElement('span');
        tag.className = 'font-bold uppercase text-[10px] tracking-wider shrink-0 mt-0.5';
        tag.textContent = a.type.replace('_', ' ');
        const msg = document.createElement('span');
        msg.textContent = a.message;
        li.append(tag, msg);
        return li;
    }));
}

function renderTelemetryTable(zones) {
    const tbody = document.getElementById('cameraTableBody');
    if (!tbody) return;

    if (zones.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" class="px-4 py-6 text-center text-slate-400">No telemetry records available.</td></tr>';
        return;
    }

    tbody.innerHTML = zones.map(z => {
        const sc = statusColor(z.status);
        const satRiskPct = Math.round((z.saturation_probability || 0) * 100);
        const age = Math.round(z.camera_age_seconds || 0);

        return `
        <tr class="hover:bg-slate-50 transition-colors">
            <td class="px-4 py-3 font-mono font-bold text-slate-900">${z.zone_id}</td>
            <td class="px-4 py-3 font-medium">${z.name}</td>
            <td class="px-4 py-3 font-mono text-slate-500">${z.camera_id || 'CAM-01'}</td>
            <td class="px-4 py-3 font-bold text-slate-700">${z.operational_capacity}</td>
            <td class="px-4 py-3 font-bold ${z.predicted_vehicle_count > z.operational_capacity * 0.9 ? 'text-rose-600' : 'text-slate-900'}">${z.predicted_vehicle_count}</td>
            <td class="px-4 py-3 font-bold text-emerald-600">${z.available_spaces}</td>
            <td class="px-4 py-3 font-mono text-slate-500">${z.predicted_low != null ? `${z.predicted_low}–${z.predicted_high}` : '—'}</td>
            <td class="px-4 py-3">
                <span class="font-bold ${satRiskPct > 70 ? 'text-rose-600' : 'text-slate-600'}">${satRiskPct}%</span>
            </td>
            <td class="px-4 py-3 text-slate-400 font-mono text-[11px]">${age}s ago</td>
            <td class="px-4 py-3">
                <span class="inline-flex items-center text-[11px] px-2.5 py-0.5 rounded-full font-bold border ${sc.badge}">
                    <span class="w-1.5 h-1.5 rounded-full ${sc.dot} mr-1"></span>
                    ${statusLabel(z.status)}
                </span>
                ${z.stream_frozen ? '<span class="inline-flex items-center text-[10px] px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-bold ml-1 border border-amber-300" title="Invariant sensor stream detected">⚠ Invariant</span>' : ''}
            </td>
        </tr>`;
    }).join('');
}

document.addEventListener('DOMContentLoaded', () => {
    refreshSnapshot();
    snapshotTimer = setInterval(refreshSnapshot, 10000);
});
