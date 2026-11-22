'use strict';

(function () {
    const esc = window.escapeHtml;
    const form = document.getElementById('calForm');
    const err = document.getElementById('calError');
    const KIND_STYLE = {
        event:   ['bg-sky-50 text-sky-700 border-sky-200', 'fa-champagne-glasses', 'Event'],
        exam:    ['bg-amber-50 text-amber-800 border-amber-200', 'fa-pen-to-square', 'Exams'],
        holiday: ['bg-violet-50 text-violet-700 border-violet-200', 'fa-umbrella-beach', 'Holiday'],
    };
    const HELP = {
        event: 'A one-off event at a block. Add the extra vehicles you expect; nearby grounds are loaded more than distant ones.',
        exam: 'An exam window. Weekdays inside it get the exam-day pattern (academic grounds fill earlier and stay full).',
        holiday: 'No classes: the lots behave like a weekend for these dates.',
    };
    let BLOCKS = [];

    const kind = () => form.querySelector('input[name="kind"]:checked').value;
    const pad = n => String(n).padStart(2, '0');
    const localInput = d => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;

    function applyKind() {
        const k = kind();
        document.getElementById('kindHelp').textContent = HELP[k];
        document.getElementById('eventFields').classList.toggle('hidden', k !== 'event');
        const start = document.getElementById('calStart'), end = document.getElementById('calEnd');
        const dateOnly = k !== 'event';
        // exams / holidays are whole days: switch the inputs to date pickers, keeping the chosen day
        for (const el of [start, end]) {
            const keep = el.value ? el.value.slice(0, 10) : '';
            el.type = dateOnly ? 'date' : 'datetime-local';
            if (keep) el.value = dateOnly ? keep : keep + (el === start ? 'T09:00' : 'T17:00');
        }
        document.getElementById('calEnd').required = false;
        updateImpact();
    }

    function payload() {
        const k = kind();
        const body = { kind: k, name: document.getElementById('calName').value.trim(),
                       start: document.getElementById('calStart').value, end: document.getElementById('calEnd').value || null,
                       notes: document.getElementById('calNotes').value.trim() };
        if (k === 'event') {
            body.venue_block = document.getElementById('calVenue').value;
            const v = parseInt(document.getElementById('calVehicles').value, 10);
            body.expected_vehicles = isNaN(v) ? null : v;
        }
        return body;
    }

    let timer = null;
    function updateImpact() {
        clearTimeout(timer);
        timer = setTimeout(async () => {
            const box = document.getElementById('impactBox');
            const b = payload();
            if (b.kind !== 'event' || !b.start || !b.end || !b.expected_vehicles || !b.venue_block || b.name.length < 2) {
                box.classList.add('hidden'); return;
            }
            try {
                const { entry } = await window.apiJson('/api/calendar/preview', { method: 'POST', body: b });
                document.getElementById('impactList').innerHTML = (entry.impact || []).map(i =>
                    `<li class="flex justify-between gap-2"><span>${esc(i.name)}</span><span class="font-mono font-bold">+${i.extra_vehicles} cars (+${i.extra_occupancy_pct}% full)</span></li>`).join('');
                box.classList.remove('hidden');
            } catch (e) { box.classList.add('hidden'); }
        }, 300);
    }

    function when(e) {
        const s = new Date(e.start), t = new Date(e.end);
        const dOpt = { day: 'numeric', month: 'short', year: 'numeric' }, tOpt = { hour: '2-digit', minute: '2-digit' };
        if (e.kind !== 'event') {
            return s.toDateString() === t.toDateString() ? s.toLocaleDateString([], dOpt) : `${s.toLocaleDateString([], dOpt)} to ${t.toLocaleDateString([], dOpt)}`;
        }
        return `${s.toLocaleDateString([], dOpt)}, ${s.toLocaleTimeString([], tOpt)} to ${t.toLocaleTimeString([], tOpt)}`;
    }

    async function load() {
        const days = document.getElementById('calRange').value;
        try {
            const data = await window.apiJson(`/api/calendar?days_back=14&days_ahead=${days}`);
            const c = data.context_now || {};
            const parts = [];
            if (c.holiday) parts.push('holiday today'); if (c.exam_day) parts.push('exam day');
            (c.events_now || []).forEach(e => parts.push(`event on: ${e.name}`));
            document.getElementById('nowContext').textContent = parts.length ? `Right now: ${parts.join(', ')}.` : 'Right now: a normal day.';
            const list = document.getElementById('calList');
            if (!data.entries.length) { list.innerHTML = '<li class="px-5 py-8 text-center text-slate-400">Nothing on the calendar in this range.</li>'; return; }
            const now = new Date();
            list.innerHTML = data.entries.map(e => {
                const [tone, icon, label] = KIND_STYLE[e.kind];
                const past = new Date(e.end) < now;
                return `<li class="px-5 py-3 flex items-start justify-between gap-3 ${past ? 'opacity-55' : ''}">
                    <div class="min-w-0">
                        <div class="flex items-center gap-2 flex-wrap">
                            <span class="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full border ${tone}"><i class="fa-solid ${icon}"></i> ${label}</span>
                            <span class="font-bold text-slate-900">${esc(e.name)}</span>${past ? '<span class="text-[10px] text-slate-400">past</span>' : ''}
                        </div>
                        <p class="text-xs text-slate-500 mt-0.5">${esc(when(e))}${e.venue_block ? ` · ${esc(e.venue_block.replace(/_/g, ' '))} · ~${e.expected_vehicles} extra vehicles` : ''}</p>
                        ${e.impact ? `<p class="text-[11px] text-slate-400 mt-0.5">Loads: ${e.impact.slice(0, 3).map(i => `${esc(i.name.replace(' Parking', ''))} +${i.extra_occupancy_pct}%`).join(', ')}</p>` : ''}
                    </div>
                    <button data-del="${e.id}" class="shrink-0 px-2 py-1 rounded-lg hover:bg-rose-50 text-rose-600" title="Remove"><i class="fa-solid fa-trash"></i></button>
                </li>`;
            }).join('');
            list.querySelectorAll('[data-del]').forEach(b => b.addEventListener('click', async () => {
                if (!confirm('Remove this calendar entry? Forecasts stop using it immediately.')) return;
                try { await window.apiJson(`/api/calendar/${b.dataset.del}`, { method: 'DELETE' }); window.toast('Entry removed.', 'success'); load(); }
                catch (e2) { window.toast(e2.message, 'error'); }
            }));
        } catch (e) {
            document.getElementById('calList').innerHTML = `<li class="px-5 py-8 text-center text-rose-600">${esc(e.message)}</li>`;
        }
    }

    form.addEventListener('submit', async (ev) => {
        ev.preventDefault();
        err.classList.add('hidden');
        const btn = document.getElementById('calSave');
        btn.disabled = true;
        try {
            await window.apiJson('/api/calendar', { method: 'POST', body: payload() });
            window.toast('Added to the calendar. Forecasts now take it into account.', 'success');
            document.getElementById('calName').value = '';
            document.getElementById('calNotes').value = '';
            document.getElementById('calVehicles').value = '';
            document.getElementById('impactBox').classList.add('hidden');
            load();
        } catch (e) {
            err.textContent = e.message;
            err.classList.remove('hidden');
        } finally { btn.disabled = false; }
    });

    form.querySelectorAll('input[name="kind"]').forEach(r => r.addEventListener('change', applyKind));
    ['calName', 'calStart', 'calEnd', 'calVenue', 'calVehicles'].forEach(id => document.getElementById(id).addEventListener('input', updateImpact));
    document.getElementById('calRange').addEventListener('change', load);

    document.addEventListener('DOMContentLoaded', async () => {
        try {
            BLOCKS = await window.apiJson('/api/blocks');
            document.getElementById('calVenue').innerHTML = BLOCKS.map(b => `<option value="${esc(b.id)}">${esc(b.name)}</option>`).join('');
        } catch (e) { window.toast(e.message, 'error'); }
        const d = new Date(); d.setDate(d.getDate() + 1); d.setHours(9, 0, 0, 0);
        const e = new Date(d); e.setHours(17);
        document.getElementById('calStart').value = localInput(d);
        document.getElementById('calEnd').value = localInput(e);
        applyKind();
        load();
    });
})();
