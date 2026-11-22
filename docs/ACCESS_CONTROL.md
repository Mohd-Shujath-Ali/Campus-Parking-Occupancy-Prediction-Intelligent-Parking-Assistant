# Accounts, roles and access

Everyone signs in on the same page (`/login`). The role on the account decides what the server
allows; the navigation only shows pages the role can open. Every `/api` route except
`/api/health`, `/api/auth/login` and `/api/auth/logout` needs a session, and
`tests/test_auth.py` fails if a new route forgets its permission check.

## Roles

| Role | Sees | Can do |
|---|---|---|
| Student, Faculty | Find Parking, Campus Grounds, Account | Recommendations from their home block, navigate, forecasts |
| Vendor | Same as students | Same; account always expires (default 30 days, max 1 year) |
| Guard | + Estate Console | File violation reports |
| Warden | + Estate Console | Close a ground or mark it full (optionally for N minutes); resolve reports |
| Admin | Everything | Cameras and ROI, detection, users, bulk import, audit log, model benchmarks, **campus calendar** (events, exam windows, holidays), **Model Health** (version, data sources, live accuracy, reload a retrained model) |

Permissions are defined once in `api/auth.py` (`PERMISSIONS`). Everyone can *read* the calendar (`view_occupancy`); only admins
can change it (`manage_calendar`), see model health (`view_model`) or reload the model (`manage_model`). How forecasts use the
calendar and live counts: [FORECASTING.md](FORECASTING.md).

## Nearest parking from the user's block

Each account can have a home block (set by the admin, or by the user on the Account page).
`/api/recommendations` without a `destination` ranks from that block. Walking distances in
`configs/zones.yaml` exist for 8 reference blocks; every other block is mapped to the
adjacent reference block in `configs/campus_blocks.yaml` (for example Block 55 → Block 57).
**Have the estate office confirm that adjacency against the campus map.** To add a block,
add one line to that file; no code changes.

Grounds a warden has closed or marked full are listed last and cannot be navigated to.

## First run

With no accounts, the server creates `admin` with a one-time password, printed in the log and
written to `data/live/initial_admin_password.txt` (deleted once the admin sets a password).
Set `ADMIN_USERNAME` / `ADMIN_PASSWORD` to choose them instead.

Issue real accounts from **Users → Bulk import** (CSV: `username, role, full_name, home_block,
department, expires_at`). Each account gets a one-time password, downloadable once from the
import result, which must be changed at first sign-in. Use registration numbers / employee IDs
as user IDs. There is no self-registration.

Demo accounts for local testing: `python -m scripts.seed_demo_users [--known-password X]`.

## Security behaviour

- Passwords: scrypt with per-user salt; 8+ characters (12+ for admins) with a letter and a digit,
  not containing the user ID.
- Sessions: random token in an HttpOnly, SameSite=Lax cookie; only its SHA-256 is stored.
  12-hour lifetime. Password change/reset, disabling, role change and expiry end sessions.
- 5 failed sign-ins lock the account for 15 minutes (admin can unlock). Sign-in is also
  rate-limited per IP.
- Every sign-in, account change, override, report, camera and ROI change goes to the audit log.
  Stream URLs (which may hold camera credentials) are never logged.
- Camera devices post counts to `/api/ingest` with the `X-API-Key` header (`INGEST_API_KEY`);
  without a key, only an admin session can ingest.

## Settings

| Variable | Default | Purpose |
|---|---|---|
| `AUTH_DB` | `data/live/app.db` | Accounts, sessions, audit log, overrides, reports, cameras, ROI |
| `ADMIN_USERNAME`, `ADMIN_PASSWORD` | `admin`, generated | First admin account |
| `COOKIE_SECURE` | `0` | Set to `1` behind HTTPS (required in production) |
| `SESSION_TTL_HOURS` | `12` | Session lifetime |
| `LOGIN_MAX_FAILURES`, `LOGIN_LOCKOUT_MINUTES` | `5`, `15` | Lockout policy |
| `LOGIN_RATE_LIMIT` | `10/minute` | Per-IP sign-in limit |
| `INGEST_API_KEY` | unset | Key for camera devices posting to `/api/ingest` |

## Deployment note

Sessions are same-origin cookies, so run the frontend and API as one service (the current
`render.yaml` setup). The decoupled static build in `frontend_dist/` has no sign-in and is not
supported with access control. `AUTH_DB` must be on persistent storage, or every redeploy
wipes the accounts.
