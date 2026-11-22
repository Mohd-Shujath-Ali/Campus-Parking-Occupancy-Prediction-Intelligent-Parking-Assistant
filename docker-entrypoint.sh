#!/bin/sh
# Container entrypoint.
#
# Volumes (a Render disk, a Docker named volume) are usually mounted owned by root, which the unprivileged app user
# cannot write to, so the first login would fail with "unable to open database file". Start as root, give the app user
# the folders that hold the databases, then drop privileges and run the real command.
set -e

if [ "$(id -u)" = "0" ]; then
    for target in "${AUTH_DB:-/app/data/live/app.db}" "${PARKING_DB:-/app/data/live/observations.db}"; do
        dir="$(dirname "$target")"
        mkdir -p "$dir"
        chown -R appuser:appgroup "$dir"
    done
    exec setpriv --reuid=1000 --regid=1000 --init-groups "$@"
fi

exec "$@"
