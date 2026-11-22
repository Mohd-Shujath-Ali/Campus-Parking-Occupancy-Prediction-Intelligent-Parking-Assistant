'use strict';

(function () {
    const me = window.CURRENT_USER;
    const sel = document.getElementById('homeBlockSelect');
    const hint = document.getElementById('anchorHint');
    let blocks = [];

    document.querySelectorAll('[data-expiry]').forEach(el => {
        el.textContent = 'Valid until ' + window.formatWhen(el.getAttribute('data-expiry'));
    });

    function renderHint() {
        const b = blocks.find(x => x.id === sel.value);
        if (!b) { hint.textContent = 'No home block: you will pick a destination each time.'; return; }
        hint.textContent = b.is_reference
            ? 'Distances are measured from this block directly.'
            : `Adjacent to ${b.anchor.replace(/_/g, ' ')}: distances are measured from there.`;
    }

    async function loadBlocks() {
        try {
            blocks = await window.apiJson('/api/blocks');
            const groups = {};
            blocks.forEach(b => (groups[b.group] = groups[b.group] || []).push(b));
            sel.innerHTML = '<option value="">— None (choose each time) —</option>' + Object.keys(groups).sort().map(g =>
                `<optgroup label="${window.escapeHtml(g)}">` +
                groups[g].map(b => `<option value="${window.escapeHtml(b.id)}">${window.escapeHtml(b.name)}</option>`).join('') +
                '</optgroup>').join('');
            sel.value = me.home_block || '';
            renderHint();
        } catch (e) {
            sel.innerHTML = '<option value="">Could not load blocks</option>';
        }
    }
    sel.addEventListener('change', renderHint);

    document.getElementById('saveBlockBtn').addEventListener('click', async () => {
        try {
            const u = await window.apiJson('/api/auth/me', { method: 'PATCH', body: { home_block: sel.value || null } });
            me.home_block = u.home_block;
            window.toast(u.home_block ? `Home block set to ${u.home_block_name}.` : 'Home block cleared.', 'success');
        } catch (e) {
            window.toast(e.message, 'error');
        }
    });

    const form = document.getElementById('pwForm');
    const err = document.getElementById('pwError');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        err.classList.add('hidden');
        const cur = document.getElementById('curPw').value;
        const n1 = document.getElementById('newPw').value;
        const n2 = document.getElementById('newPw2').value;
        if (!cur || !n1) { err.textContent = 'Fill in all fields.'; err.classList.remove('hidden'); return; }
        if (n1 !== n2) { err.textContent = 'The new passwords do not match.'; err.classList.remove('hidden'); return; }
        const btn = document.getElementById('pwBtn');
        btn.disabled = true;
        try {
            const res = await window.apiJson('/api/auth/change-password', { method: 'POST', body: { current_password: cur, new_password: n1 } });
            window.toast('Password updated.', 'success');
            form.reset();
            if (me.must_change_password) {
                setTimeout(() => { window.location.href = res.redirect || '/'; }, 700);
            }
        } catch (e2) {
            err.textContent = e2.message;
            err.classList.remove('hidden');
        } finally {
            btn.disabled = false;
        }
    });

    loadBlocks();
})();
