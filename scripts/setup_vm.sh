#!/usr/bin/env bash
# Set up the DigitalOcean VM (Ubuntu 24.04) as the MAIN machine. Step by step guide: docs/vm_setup.md
#
#   sudo bash setup_vm.sh system    # [root]     timezone Asia/Bangkok + NTP, packages, user "forecast" (+ your SSH key)
#   bash setup_vm.sh app            # [forecast] git clone, Python 3.12 venv, requirements-live.txt (no torch)
#   sudo bash setup_vm.sh harden    # [root]     ufw: SSH only; SSH key login only (no passwords, no root password)
#
# Never put .env in git or chat: create it by hand on the VM (nano) and chmod 600 (see docs/vm_setup.md).
set -euo pipefail

APP_USER="forecast"
REPO="https://github.com/hansera1/CN340-Final-StockPrediction.git"
APP_DIR="/home/${APP_USER}/kbank_advanc_forecast"

need_root() { [ "$(id -u)" -eq 0 ] || { echo "run with sudo / as root"; exit 1; }; }

system() {
    need_root
    echo "== timezone + NTP"
    timedatectl set-timezone Asia/Bangkok
    timedatectl set-ntp true
    echo "== packages"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y python3.12 python3.12-venv python3-pip git unzip ufw cron util-linux
    systemctl enable --now cron
    systemctl restart cron          # cron picks up the new timezone
    echo "== user ${APP_USER} (runs the system; not root)"
    if ! id "$APP_USER" >/dev/null 2>&1; then
        adduser --disabled-password --gecos "" "$APP_USER"
    fi
    if [ -s /root/.ssh/authorized_keys ]; then
        install -d -m 700 -o "$APP_USER" -g "$APP_USER" "/home/${APP_USER}/.ssh"
        install -m 600 -o "$APP_USER" -g "$APP_USER" /root/.ssh/authorized_keys "/home/${APP_USER}/.ssh/authorized_keys"
        echo "copied root's SSH key to ${APP_USER}: ssh ${APP_USER}@<IP> should work now"
    else
        echo "WARNING: /root/.ssh/authorized_keys is empty - add your public key for ${APP_USER} before 'harden'"
    fi
    echo "== check"
    timedatectl | grep -E "Time zone|System clock synchronized|NTP service"
    date
}

app() {
    [ "$(id -un)" = "$APP_USER" ] || { echo "run as ${APP_USER}: sudo -iu ${APP_USER} bash $0 app"; exit 1; }
    if [ -d "$APP_DIR/.git" ]; then
        git -C "$APP_DIR" pull --ff-only
    else
        git clone "$REPO" "$APP_DIR"
    fi
    cd "$APP_DIR"
    python3.12 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements-live.txt
    mkdir -p outputs/logs data/raw
    chmod +x scripts/*.sh
    echo
    echo "next (docs/vm_setup.md):"
    echo "  1) nano $APP_DIR/.env   then   chmod 600 $APP_DIR/.env"
    echo "  2) scp the backup zip from the laptop, then: .venv/bin/python scripts/11_backup_bundle.py --restore <zip>"
    echo "  3) .venv/bin/python scripts/00_collect_daily.py --force     (VM's own data, draft, stage 2)"
    echo "  4) .venv/bin/python scripts/08_check_ready.py               (must have no FAIL)"
    echo "  5) bash scripts/install_cron.sh"
}

harden() {
    need_root
    if [ ! -s "/home/${APP_USER}/.ssh/authorized_keys" ] && [ ! -s /root/.ssh/authorized_keys ]; then
        echo "STOP: no SSH key installed - disabling password login would lock you out"; exit 1
    fi
    echo "== ufw: allow SSH only"
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow OpenSSH
    ufw --force enable
    ufw status verbose
    echo "== sshd: key login only"
    cat > /etc/ssh/sshd_config.d/99-kbank-hardening.conf <<'EOF'
PermitRootLogin prohibit-password
PasswordAuthentication no
KbdInteractiveAuthentication no
PubkeyAuthentication yes
EOF
    sshd -t
    systemctl reload ssh
    echo "done - keep this session open and test 'ssh ${APP_USER}@<IP>' from another terminal before closing"
}

case "${1:-}" in
    system) system ;;
    app) app ;;
    harden) harden ;;
    *) echo "usage: $0 system|app|harden"; exit 1 ;;
esac
