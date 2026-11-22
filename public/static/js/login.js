'use strict';

(function () {
    const form = document.getElementById('loginForm');
    const errBox = document.getElementById('loginError');
    const errText = document.getElementById('loginErrorText') || errBox;
    const btn = document.getElementById('loginBtn');
    const usernameInput = document.getElementById('username');
    const pw = document.getElementById('password');

    function safeNext(p) {
        return (p && p.startsWith('/') && !p.startsWith('//') && !p.includes('\\') && !p.startsWith('/login')) ? p : '';
    }

    function showError(msg) {
        if (errText) errText.textContent = msg;
        else errBox.textContent = msg;
        errBox.classList.remove('hidden');
    }

    const toggleBtn = document.getElementById('togglePw');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            const show = pw.type === 'password';
            pw.type = show ? 'text' : 'password';
            const icon = toggleBtn.querySelector('i');
            if (icon) icon.className = `fa-solid ${show ? 'fa-eye-slash' : 'fa-eye'} text-sm`;
        });
    }

    // Demo accounts: 1-click auto-fill and instant sign-in
    document.querySelectorAll('.demo-login').forEach(card => {
        card.addEventListener('click', () => {
            const user = card.dataset.user;
            const pass = card.dataset.pass;
            const name = card.dataset.name || user;

            if (!user || !pass) return;

            // Fill inputs with visual autofill animation
            usernameInput.value = user;
            pw.value = pass;

            usernameInput.classList.remove('input-autofill-flash');
            pw.classList.remove('input-autofill-flash');
            void usernameInput.offsetWidth; // trigger reflow
            usernameInput.classList.add('input-autofill-flash');
            pw.classList.add('input-autofill-flash');

            // Disable demo cards temporarily to prevent double submission
            document.querySelectorAll('.demo-login').forEach(x => { x.disabled = true; });

            // Update submit button text to show who is signing in
            const origHtml = btn.innerHTML;
            btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i><span>Signing in as ${name}...</span>`;

            // Submit the form
            form.requestSubmit();

            // Re-enable in case submission encounters client error or rate-limit
            setTimeout(() => {
                document.querySelectorAll('.demo-login').forEach(x => { x.disabled = false; });
                btn.innerHTML = origHtml;
            }, 3000);
        });
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        errBox.classList.add('hidden');
        const username = usernameInput.value.trim();
        const password = pw.value;
        if (!username || !password) {
            showError('Enter your user ID and password.');
            return;
        }
        btn.disabled = true;
        try {
            const url = window.apiUrl ? window.apiUrl('/api/auth/login') : '/api/auth/login';
            const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password }),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                showError(res.status === 429 ? 'Too many sign-in attempts. Wait a minute and try again.'
                    : (typeof data.detail === 'string' ? data.detail : 'Sign-in failed. Check user ID and password.'));
                pw.value = '';
                pw.focus();
                document.querySelectorAll('.demo-login').forEach(x => { x.disabled = false; });
                return;
            }
            const forced = data.redirect && data.redirect.startsWith('/account');
            const next = safeNext(document.getElementById('nextPath') ? document.getElementById('nextPath').value : '');
            window.location.href = forced ? data.redirect : (next || data.redirect || '/');
        } catch (err) {
            showError('Cannot reach the server. Check your connection and try again.');
            document.querySelectorAll('.demo-login').forEach(x => { x.disabled = false; });
        } finally {
            btn.disabled = false;
        }
    });

    usernameInput.focus();
})();
