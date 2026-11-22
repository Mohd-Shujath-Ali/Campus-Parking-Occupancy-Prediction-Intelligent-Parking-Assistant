'use strict';

const ME = window.CURRENT_USER;
const esc = window.escapeHtml;
let ROLES = [];
let BLOCKS = [];
let USERS = [];
let editingUser = null;
let auditOffset = 0;
const AUDIT_PAGE = 100;

const ROLE_BADGE = {
    admin:   'bg-slate-900 text-white border-slate-900',
    warden:  'bg-violet-50 text-violet-700 border-violet-200',
    faculty: 'bg-sky-50 text-sky-700 border-sky-200',
    student: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    vendor:  'bg-amber-50 text-amber-800 border-amber-200',
    guard:   'bg-rose-50 text-rose-700 border-rose-200',
};
const AUDIT_ACTIONS = ['login_success', 'login_failed', 'logout', 'password_changed', 'password_reset', 'profile_updated',
    'user_created', 'user_updated', 'user_deleted', 'user_unlocked', 'users_imported', 'zone_override_set',
    'zone_override_cleared', 'report_created', 'report_resolved', 'camera_updated', 'roi_saved', 'roi_activated',
    'bootstrap_admin_created'];

// ─────────────────────────── Tabs & modals ───────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.className = 'tab-btn px-3.5 py-1.5 rounded-lg text-slate-600');
    btn.className = 'tab-btn px-3.5 py-1.5 rounded-lg bg-white text-orange-600 shadow-sm';
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
    document.getElementById('tab-' + btn.dataset.tab).classList.remove('hidden');
    if (btn.dataset.tab === 'audit' && auditOffset === 0) loadAudit();
}));

function closeModal(id) { document.getElementById(id).classList.add('hidden'); }
function openModal(id) { document.getElementById(id).classList.remove('hidden'); }

function roleLabel(r) { return (ROLES.find(x => x.role === r) || {}).label || r; }
function blockName(id) { return (BLOCKS.find(b => b.id === id) || {}).name || id; }
function localDate(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

// ─────────────────────────── Reference data ───────────────────────────
async function loadReference() {
    [ROLES, BLOCKS] = await Promise.all([window.apiJson('/api/roles'), window.apiJson('/api/blocks')]);
    const roleOpts = ROLES.map(r => `<option value="${r.role}">${esc(r.label)}</option>`).join('');
    document.getElementById('roleFilter').insertAdjacentHTML('beforeend', roleOpts);
    document.getElementById('fRole').innerHTML = roleOpts;
    const groups = {};
    BLOCKS.forEach(b => (groups[b.group] = groups[b.group] || []).push(b));
    document.getElementById('fBlock').innerHTML = '<option value="">— None —</option>' + Object.keys(groups).sort().map(g =>
        `<optgroup label="${esc(g)}">${groups[g].map(b => `<option value="${esc(b.id)}">${esc(b.name)}</option>`).join('')}</optgroup>`).join('');
    document.getElementById('auditAction').insertAdjacentHTML('beforeend',
        AUDIT_ACTIONS.map(a => `<option value="${a}">${a.replace(/_/g, ' ')}</option>`).join(''));
}

// ─────────────────────────── Users list ───────────────────────────
let searchTimer = null;
document.getElementById('userSearch').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadUsers, 250); });
document.getElementById('roleFilter').addEventListener('change', loadUsers);

async function loadUsers() {
    const q = document.getElementById('userSearch').value.trim();
    const role = document.getElementById('roleFilter').value;
    const params = new URLSearchParams();
    if (q) params.set('q', q);
    if (role) params.set('role', role);
    try {
        const data = await window.apiJson('/api/users?' + params);
        USERS = data.users;
        renderCounts(data.counts, data.total);
        renderUsers();
    } catch (e) {
        document.getElementById('usersBody').innerHTML = `<tr><td colspan="8" class="px-4 py-8 text-center text-rose-600">${esc(e.message)}</td></tr>`;
    }
}

function renderCounts(counts, total) {
    document.getElementById('roleCounts').innerHTML = ROLES.map(r => `
        <button onclick="filterRole('${r.role}')" class="text-left bg-white p-3 rounded-xl border border-slate-200 shadow-sm hover:border-orange-300 transition-all">
            <span class="text-[10px] font-bold text-slate-500 uppercase tracking-wider">${esc(r.label)}</span>
            <div class="text-xl font-black text-slate-900">${counts[r.role] || 0}</div>
        </button>`).join('');
}

function filterRole(role) {
    const sel = document.getElementById('roleFilter');
    sel.value = sel.value === role ? '' : role;
    loadUsers();
}

function statusChips(u) {
    const chips = [];
    if (!u.is_active) chips.push(['Disabled', 'bg-slate-100 text-slate-600 border-slate-300']);
    else if (u.is_expired) chips.push(['Expired', 'bg-rose-50 text-rose-700 border-rose-200']);
    else chips.push(['Active', 'bg-emerald-50 text-emerald-700 border-emerald-200']);
    if (u.is_locked) chips.push(['Locked', 'bg-rose-50 text-rose-700 border-rose-200']);
    if (u.must_change_password) chips.push(['Awaiting first sign-in', 'bg-amber-50 text-amber-800 border-amber-200']);
    return chips.map(([t, c]) => `<span class="inline-block text-[10px] font-bold px-2 py-0.5 rounded-full border ${c} mr-1 mb-0.5">${t}</span>`).join('');
}

function renderUsers() {
    const body = document.getElementById('usersBody');
    if (!USERS.length) {
        body.innerHTML = '<tr><td colspan="8" class="px-4 py-8 text-center text-slate-400">No users match.</td></tr>';
        return;
    }
    body.innerHTML = USERS.map(u => {
        const self = u.id === ME.id;
        return `
        <tr class="hover:bg-slate-50">
            <td class="px-4 py-3 font-mono font-bold text-slate-900">${esc(u.username)}${self ? ' <span class="text-[10px] text-orange-600 font-sans">(you)</span>' : ''}</td>
            <td class="px-4 py-3">${esc(u.full_name) || '<span class="text-slate-400">—</span>'}${u.department ? `<div class="text-[11px] text-slate-400">${esc(u.department)}</div>` : ''}</td>
            <td class="px-4 py-3"><span class="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full border ${ROLE_BADGE[u.role] || ''}">${esc(roleLabel(u.role))}</span></td>
            <td class="px-4 py-3">${u.home_block ? esc(blockName(u.home_block)) : '<span class="text-slate-400">—</span>'}</td>
            <td class="px-4 py-3">${statusChips(u)}</td>
            <td class="px-4 py-3 text-slate-500">${u.last_login_at ? esc(window.formatWhen(u.last_login_at)) : 'Never'}</td>
            <td class="px-4 py-3 text-slate-500">${u.expires_at ? esc(window.formatWhen(u.expires_at)) : '—'}</td>
            <td class="px-4 py-3 text-right whitespace-nowrap">
                <button onclick="openUserModal(${u.id})" class="px-2 py-1 rounded-lg hover:bg-slate-100 text-slate-600" title="Edit"><i class="fa-solid fa-pen"></i></button>
                <button onclick="resetPassword(${u.id})" class="px-2 py-1 rounded-lg hover:bg-amber-50 text-amber-600" title="Reset password"><i class="fa-solid fa-key"></i></button>
                ${u.is_locked ? `<button onclick="unlockUser(${u.id})" class="px-2 py-1 rounded-lg hover:bg-emerald-50 text-emerald-600" title="Unlock"><i class="fa-solid fa-unlock"></i></button>` : ''}
                ${self ? '' : `<button onclick="deleteUser(${u.id})" class="px-2 py-1 rounded-lg hover:bg-rose-50 text-rose-600" title="Delete"><i class="fa-solid fa-trash"></i></button>`}
            </td>
        </tr>`;
    }).join('');
}

// ─────────────────────────── Create / edit ───────────────────────────
function updateExpiryHint() {
    const vendor = document.getElementById('fRole').value === 'vendor';
    document.getElementById('fExpiresHint').textContent = vendor
        ? 'Vendors always expire: leave empty for 30 days (max 1 year).'
        : 'Leave empty for no expiry.';
}
document.getElementById('fRole').addEventListener('change', updateExpiryHint);

function openUserModal(id) {
    editingUser = id ? USERS.find(u => u.id === id) : null;
    const u = editingUser;
    document.getElementById('userModalTitle').textContent = u ? `Edit ${u.username}` : 'Add user';
    document.getElementById('fUserId').value = u ? u.id : '';
    const un = document.getElementById('fUsername');
    un.value = u ? u.username : '';
    un.disabled = !!u;
    document.getElementById('fRole').value = u ? u.role : 'student';
    document.getElementById('fRole').disabled = !!(u && u.id === ME.id);
    document.getElementById('fName').value = u ? (u.full_name || '') : '';
    document.getElementById('fBlock').value = u ? (u.home_block || '') : '';
    document.getElementById('fDept').value = u ? (u.department || '') : '';
    document.getElementById('fExpires').value = u ? localDate(u.expires_at) : '';
    document.getElementById('fActive').checked = u ? u.is_active : true;
    document.getElementById('fActive').disabled = !!(u && u.id === ME.id);
    document.getElementById('userFormError').classList.add('hidden');
    updateExpiryHint();
    openModal('userModal');
    (u ? document.getElementById('fName') : un).focus();
}

document.getElementById('userForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const err = document.getElementById('userFormError');
    err.classList.add('hidden');
    const btn = document.getElementById('userSaveBtn');
    btn.disabled = true;
    const expires = document.getElementById('fExpires').value;
    try {
        if (editingUser) {
            const body = {
                full_name: document.getElementById('fName').value.trim(),
                home_block: document.getElementById('fBlock').value || null,
                department: document.getElementById('fDept').value.trim() || null,
            };
            if (editingUser.id !== ME.id) {
                body.role = document.getElementById('fRole').value;
                body.is_active = document.getElementById('fActive').checked;
            }
            if (expires !== localDate(editingUser.expires_at)) body.expires_at = expires || null;
            await window.apiJson(`/api/users/${editingUser.id}`, { method: 'PATCH', body });
            window.toast(`Saved ${editingUser.username}.`, 'success');
            closeModal('userModal');
        } else {
            const res = await window.apiJson('/api/users', { method: 'POST', body: {
                username: document.getElementById('fUsername').value.trim(),
                role: document.getElementById('fRole').value,
                full_name: document.getElementById('fName').value.trim(),
                home_block: document.getElementById('fBlock').value || null,
                department: document.getElementById('fDept').value.trim() || null,
                expires_at: expires || null,
            }});
            closeModal('userModal');
            showCredential(res.user.username, res.temporary_password, 'Account created');
        }
        loadUsers();
    } catch (e2) {
        err.textContent = e2.message;
        err.classList.remove('hidden');
    } finally {
        btn.disabled = false;
    }
});

function showCredential(username, pw, title) {
    document.getElementById('credTitle').textContent = title;
    document.getElementById('credUser').textContent = username;
    document.getElementById('credPw').textContent = pw;
    openModal('credModal');
}

async function copyCred() {
    try {
        await navigator.clipboard.writeText(document.getElementById('credPw').textContent);
        window.toast('Copied.', 'success');
    } catch (e) {
        window.toast('Copy failed: select the password and copy it manually.', 'warning');
    }
}

async function resetPassword(id) {
    const u = USERS.find(x => x.id === id);
    if (!confirm(`Reset the password for ${u.username}? Their current password stops working and they are signed out everywhere.`)) return;
    try {
        const res = await window.apiJson(`/api/users/${id}/reset-password`, { method: 'POST' });
        showCredential(u.username, res.temporary_password, 'Password reset');
        loadUsers();
    } catch (e) { window.toast(e.message, 'error'); }
}

async function unlockUser(id) {
    try {
        await window.apiJson(`/api/users/${id}/unlock`, { method: 'POST' });
        window.toast('Account unlocked.', 'success');
        loadUsers();
    } catch (e) { window.toast(e.message, 'error'); }
}

async function deleteUser(id) {
    const u = USERS.find(x => x.id === id);
    if (!confirm(`Delete ${u.username} (${roleLabel(u.role)}) permanently? To keep their history, disable the account instead.`)) return;
    try {
        await window.apiJson(`/api/users/${id}`, { method: 'DELETE' });
        window.toast(`Deleted ${u.username}.`, 'success');
        loadUsers();
    } catch (e) { window.toast(e.message, 'error'); }
}

// ─────────────────────────── CSV import ───────────────────────────
function downloadBlob(name, text) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: 'text/csv' }));
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function csvCell(v) {
    const s = String(v == null ? '' : v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function downloadTemplate() {
    downloadBlob('users_template.csv',
        'username,full_name,role,home_block,department,expires_at\n' +
        '12301234,Aarav Sharma,student,Block_26_CSE,School of CSE,\n' +
        '26541,Dr. Meera Kapoor,faculty,Block_55_Mech,School of Mechanical Engineering,\n' +
        'W-BH2-01,Rajinder Singh,warden,BH2_Hostel,Boys Hostel 2,\n' +
        'V-CANTEEN-07,Punjab Caterers,vendor,MB_Food_Square,Food services,2026-12-31\n' +
        'G-GATE-03,Harpreet Kaur,guard,,Security,\n');
}

async function runImport() {
    const file = document.getElementById('importFile').files[0];
    if (!file) { window.toast('Choose a CSV file first.', 'warning'); return; }
    const btn = document.getElementById('importBtn');
    btn.disabled = true;
    const box = document.getElementById('importResult');
    try {
        const fd = new FormData();
        fd.append('file', file);
        const res = await window.apiJson('/api/users/import', { method: 'POST', form: fd });
        window.LAST_IMPORT = res.created;
        box.innerHTML = `
            <div class="flex flex-wrap gap-2 text-sm">
                <span class="px-3 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-bold">${res.created_count} created</span>
                <span class="px-3 py-1 rounded-full ${res.error_count ? 'bg-rose-50 text-rose-700 border-rose-200' : 'bg-slate-100 text-slate-600 border-slate-200'} border font-bold">${res.error_count} skipped</span>
            </div>
            ${res.created_count ? `
            <div class="bg-amber-50 border border-amber-200 rounded-xl p-3 text-xs text-amber-900 flex flex-col sm:flex-row sm:items-center gap-2">
                <span class="flex-1"><b>Download the one-time passwords now.</b> They are not stored anywhere and cannot be shown again. Hand them out, then delete the file.</span>
                <button onclick="downloadImportCreds()" class="px-3 py-2 bg-slate-900 hover:bg-orange-600 text-white rounded-lg font-bold whitespace-nowrap"><i class="fa-solid fa-download"></i> Download passwords</button>
            </div>` : ''}
            ${res.error_count ? `
            <div class="overflow-x-auto border border-slate-200 rounded-xl">
                <table class="w-full text-left text-xs">
                    <thead class="bg-slate-50 text-slate-500 font-bold uppercase tracking-wider"><tr><th class="px-3 py-2">Line</th><th class="px-3 py-2">User ID</th><th class="px-3 py-2">Problem</th></tr></thead>
                    <tbody class="divide-y divide-slate-100">${res.errors.map(e => `<tr><td class="px-3 py-2 font-mono">${e.line}</td><td class="px-3 py-2 font-mono">${esc(e.username || '')}</td><td class="px-3 py-2 text-rose-700">${esc(e.error)}</td></tr>`).join('')}</tbody>
                </table>
            </div>` : ''}`;
        box.classList.remove('hidden');
        document.getElementById('importFile').value = '';
        loadUsers();
    } catch (e) {
        window.toast(e.message, 'error');
    } finally {
        btn.disabled = false;
    }
}

function downloadImportCreds() {
    const rows = window.LAST_IMPORT || [];
    downloadBlob('one_time_passwords.csv', 'username,full_name,role,home_block,one_time_password\n' +
        rows.map(r => [r.username, r.full_name, r.role, r.home_block, r.temporary_password].map(csvCell).join(',')).join('\n') + '\n');
}

// ─────────────────────────── Audit log ───────────────────────────
let auditTimer = null;
document.getElementById('auditActor').addEventListener('input', () => { clearTimeout(auditTimer); auditTimer = setTimeout(() => loadAudit(), 300); });
document.getElementById('auditAction').addEventListener('change', () => loadAudit());

async function loadAudit(more = false) {
    if (!more) auditOffset = 0;
    const params = new URLSearchParams({ limit: AUDIT_PAGE, offset: auditOffset });
    const actor = document.getElementById('auditActor').value.trim();
    const action = document.getElementById('auditAction').value;
    if (actor) params.set('actor', actor);
    if (action) params.set('action', action);
    const body = document.getElementById('auditBody');
    try {
        const { entries } = await window.apiJson('/api/audit?' + params);
        const html = entries.map(e => {
            const bad = e.action === 'login_failed';
            const detail = e.detail ? Object.entries(e.detail).map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`).join(', ') : '';
            return `<tr class="hover:bg-slate-50">
                <td class="px-4 py-2.5 whitespace-nowrap text-slate-500">${esc(window.formatWhen(e.ts))}</td>
                <td class="px-4 py-2.5 font-mono font-bold">${esc(e.actor_username || '—')}</td>
                <td class="px-4 py-2.5"><span class="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full border ${bad ? 'bg-rose-50 text-rose-700 border-rose-200' : 'bg-slate-100 text-slate-600 border-slate-200'}">${esc(e.action.replace(/_/g, ' '))}</span></td>
                <td class="px-4 py-2.5 font-mono">${esc(e.target || '')}</td>
                <td class="px-4 py-2.5 text-slate-500 max-w-xs truncate" title="${esc(detail)}">${esc(detail)}</td>
                <td class="px-4 py-2.5 font-mono text-slate-400">${esc(e.ip || '')}</td>
            </tr>`;
        }).join('');
        if (more) body.insertAdjacentHTML('beforeend', html);
        else body.innerHTML = html || '<tr><td colspan="6" class="px-4 py-8 text-center text-slate-400">No entries.</td></tr>';
        auditOffset += entries.length;
        document.getElementById('auditMore').classList.toggle('hidden', entries.length < AUDIT_PAGE);
    } catch (e) {
        body.innerHTML = `<tr><td colspan="6" class="px-4 py-8 text-center text-rose-600">${esc(e.message)}</td></tr>`;
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    try { await loadReference(); } catch (e) { window.toast(e.message, 'error'); }
    loadUsers();
});
