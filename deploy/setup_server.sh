#!/usr/bin/env bash
# One-time (and re-runnable) setup of the website's app on an Ubuntu server.
#   sudo bash setup_server.sh <server files zip>
# The zip holds what the repo does not: the Perch model, EarthChirp's library, data/perch_windows,
# the ESC-50 outgroup, data/review and data/uploads (see README: Website > Deploy). The code comes
# from the public repo. After this, the app runs as a service on 127.0.0.1:8765 only; Caddy
# (deploy/Caddyfile) is the way in once the server has a domain name.
set -euo pipefail

ZIP=${1:?usage: sudo bash setup_server.sh <server files zip>}
APP=/srv/natgeo
RUNAS=natgeo
REPO=https://github.com/jeeven0099/Natgeo-Biodiversity-Detection.git
PKGS="numpy==2.4.6 pandas==2.3.3 pyarrow==25.0.1 scipy==1.17.1 scikit-learn==1.9.1 soundfile==0.14.0 onnxruntime==1.22.0 pillow==13.0.0"

[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }
[ -f "$ZIP" ] || { echo "no file $ZIP" >&2; exit 1; }

echo "== packages"
apt-get update -q
apt-get install -y -q python3-venv git unzip

echo "== swap (a long scan peaks near 2 GB; swap keeps a 4 GB server from killing the app)"
if [ "$(swapon --show --noheadings | wc -l)" -eq 0 ]; then
    fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
    grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
swapon --show

echo "== user and code"
id -u "$RUNAS" >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin "$RUNAS"
if [ -d "$APP/.git" ]; then
    sudo -u "$RUNAS" git -C "$APP" pull --ff-only
else
    mkdir -p "$APP" && chown "$RUNAS:$RUNAS" "$APP"
    sudo -u "$RUNAS" git clone -q "$REPO" "$APP"
fi
git -C "$APP" -c safe.directory="$APP" log --oneline -1

echo "== data (from $ZIP; files already on the server are kept, so a re-run never rolls back reviews)"
unzip -n -q "$ZIP" -d "$APP"
chown -R "$RUNAS:$RUNAS" "$APP"

echo "== python"
[ -x "$APP/.venv-perch/bin/python" ] || sudo -u "$RUNAS" python3 -m venv "$APP/.venv-perch"
sudo -u "$RUNAS" "$APP/.venv-perch/bin/pip" install -q --upgrade pip
# shellcheck disable=SC2086
sudo -u "$RUNAS" "$APP/.venv-perch/bin/pip" install -q $PKGS
sudo -u "$RUNAS" "$APP/.venv-perch/bin/python" -c "import onnxruntime, numpy, pandas, pyarrow, scipy, sklearn, soundfile; print('imports ok, onnxruntime', onnxruntime.__version__)"

echo "== service"
cp "$APP/deploy/callfinder.service" /etc/systemd/system/callfinder.service
cp "$APP/deploy/natgeo-train.service" "$APP/deploy/natgeo-train.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable callfinder
systemctl enable --now natgeo-train.timer      # weekly refit of the learners (scripts/weekly_train.py)
systemctl list-timers natgeo-train.timer --no-pager
systemctl restart callfinder
for _ in $(seq 1 90); do
    journalctl -u callfinder --since "-3 min" --no-pager | grep -q "website at" && break
    sleep 2
done
journalctl -u callfinder --since "-3 min" --no-pager | tail -n 15
if curl -fsS http://127.0.0.1:8765/api/meta | grep -q '"can_add": true'; then
    echo "OK: the app answers on 127.0.0.1:8765 and accepts uploads"
else
    echo "the app is not answering or adding is off: see the log above" >&2
    exit 1
fi
