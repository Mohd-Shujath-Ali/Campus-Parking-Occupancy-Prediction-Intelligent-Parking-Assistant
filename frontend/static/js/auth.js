/**
 * Session helpers shared by every page.
 * - window.CURRENT_USER is injected by the server on logged-in pages (role, permissions, home block).
 * - Any API call that comes back 401 sends the user to the login page; a 403
 *   "password_change_required" sends them to the account page to set their own password.
 * The server enforces every permission; this file only keeps the UI consistent with it.
 */
(function () {
    'use strict';

    const nativeFetch = window.fetch.bind(window);
    let redirecting = false;

    function goLogin() {
        if (redirecting) return;
        redirecting = true;
        window.location.href = '/login?next=' + encodeURIComponent(window.location.pathname + window.location.search);
    }

    window.fetch = async function (input, init) {
        const res = await nativeFetch(input, init);
        try {
            const url = typeof input === 'string' ? input : (input && input.url) || '';
            if (window.CURRENT_USER && url.includes('/api/') && !url.includes('/api/auth/login')) {
                if (res.status === 401) {
                    goLogin();
                } else if (res.status === 403) {
                    const body = await res.clone().json().catch(() => null);
                    // Already on the account page? Stay: that is where the password gets changed. Redirecting
                    // from here would reload the page forever.
                    if (body && body.detail === 'password_change_required' && !redirecting
                        && !window.location.pathname.startsWith('/account')) {
                        redirecting = true;
                        window.location.href = '/account?first=1';
                    }
                }
            }
        } catch (e) { /* never break the caller */ }
        return res;
    };

    window.hasPerm = function (perm) {
        const u = window.CURRENT_USER;
        return !!(u && Array.isArray(u.permissions) && u.permissions.includes(perm));
    };

    window.logout = async function () {
        try { await nativeFetch(window.apiUrl ? window.apiUrl('/api/auth/logout') : '/api/auth/logout', { method: 'POST' }); }
        finally { window.location.href = '/login'; }
    };

    /** fetch + JSON with readable errors. Pass {body} for JSON or {form} for FormData. */
    window.apiJson = async function (path, opts = {}) {
        const init = { method: opts.method || 'GET', headers: {} };
        if (opts.form) {
            init.body = opts.form;
        } else if (opts.body !== undefined) {
            init.headers['Content-Type'] = 'application/json';
            init.body = JSON.stringify(opts.body);
        }
        const res = await fetch(window.apiUrl ? window.apiUrl(path) : path, init);
        let data = null;
        try { data = await res.json(); } catch (e) { /* empty body */ }
        if (!res.ok) {
            const d = data && data.detail;
            const msg = typeof d === 'string' ? d
                : Array.isArray(d) ? d.map(e => `${(e.loc || []).slice(-1)[0] || 'field'}: ${e.msg}`).join('; ')
                : `Request failed (HTTP ${res.status})`;
            const err = new Error(msg);
            err.status = res.status;
            throw err;
        }
        return data;
    };

    window.escapeHtml = function (s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    };

    window.formatWhen = function (iso) {
        if (!iso) return '—';
        const d = new Date(iso);
        return isNaN(d) ? iso : d.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
    };

    window.toast = function (message, kind = 'info') {
        let wrap = document.getElementById('toastWrap');
        if (!wrap) {
            wrap = document.createElement('div');
            wrap.id = 'toastWrap';
            wrap.className = 'fixed bottom-4 right-4 left-4 sm:left-auto z-[60] space-y-2 sm:w-96';
            document.body.appendChild(wrap);
        }
        const colors = {
            info: 'bg-slate-900 text-white',
            success: 'bg-emerald-600 text-white',
            error: 'bg-rose-600 text-white',
            warning: 'bg-amber-500 text-slate-900',
        };
        const el = document.createElement('div');
        el.className = `px-4 py-3 rounded-xl shadow-lg text-sm font-medium ${colors[kind] || colors.info}`;
        el.setAttribute('role', 'status');
        el.textContent = message;
        wrap.appendChild(el);
        setTimeout(() => el.remove(), kind === 'error' ? 7000 : 4000);
    };
})();
