#!/usr/bin/env bash
# =============================================================================
# SOALM V2 — one-shot setup for an Oracle Cloud (OCI) Always Free VM.
#
# Run ON THE VM (Ubuntu 22.04/24.04 or Oracle Linux, ARM or x86):
#
#   curl -fsSL https://raw.githubusercontent.com/SparrowSyntax3210/SOALM/main/deploy/oracle-setup.sh | bash
#
# or, after cloning the repo:
#
#   bash deploy/oracle-setup.sh
#
# Safe to re-run: it pulls the latest code and rebuilds. Use it to update.
#
# Optional overrides:
#   REPO_URL=...            git URL to clone   (default: this repo)
#   APP_DIR=...             install directory  (default: ~/SOALM)
#   SOALM_DOMAIN=...        public hostname    (default: <public-ip>.sslip.io)
#   SOALM_CORS_ORIGINS=...  allowed origins    (default: *)
# =============================================================================
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/SparrowSyntax3210/SOALM.git}"
APP_DIR="${APP_DIR:-$HOME/SOALM}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# -----------------------------------------------------------------------------
# 1. Base packages + Docker
# -----------------------------------------------------------------------------
say "Installing git, curl and Docker (if missing)"
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y git curl ca-certificates
elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y git curl ca-certificates
fi

if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | sudo sh
fi
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER" || true

# -----------------------------------------------------------------------------
# 2. Host firewall
#    Oracle's Ubuntu images ship iptables rules that REJECT everything except
#    SSH, even after you open ports in the OCI Security List. Open 80 and 443.
# -----------------------------------------------------------------------------
say "Opening ports 80 and 443 in the VM firewall"
if command -v firewall-cmd >/dev/null 2>&1 && sudo firewall-cmd --state >/dev/null 2>&1; then
    # Oracle Linux (firewalld)
    sudo firewall-cmd --permanent --add-service=http
    sudo firewall-cmd --permanent --add-service=https
    sudo firewall-cmd --reload
else
    # Ubuntu (iptables). Insert at the top so they come before the REJECT rule.
    for port in 80 443; do
        if ! sudo iptables -C INPUT -p tcp --dport "$port" -m conntrack --ctstate NEW -j ACCEPT 2>/dev/null; then
            sudo iptables -I INPUT 1 -p tcp --dport "$port" -m conntrack --ctstate NEW -j ACCEPT
        fi
    done
    if ! command -v netfilter-persistent >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
        echo iptables-persistent iptables-persistent/autosave_v4 boolean true | sudo debconf-set-selections
        echo iptables-persistent iptables-persistent/autosave_v6 boolean true | sudo debconf-set-selections
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent
    fi
    sudo netfilter-persistent save || true
fi

# -----------------------------------------------------------------------------
# 3. Get / update the code
# -----------------------------------------------------------------------------
if [ -d "$APP_DIR/.git" ]; then
    say "Updating code in $APP_DIR"
    git -C "$APP_DIR" pull --ff-only
else
    say "Cloning $REPO_URL into $APP_DIR (about 400 MB, takes a minute)"
    git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

# -----------------------------------------------------------------------------
# 4. Settings (.env) — created once, kept on later runs
# -----------------------------------------------------------------------------
if [ ! -f .env ]; then
    PUBLIC_IP="$(curl -fsS https://ifconfig.me || curl -fsS https://api.ipify.org)"
    DEFAULT_DOMAIN="${PUBLIC_IP//./-}.sslip.io"
    say "Writing .env (domain: ${SOALM_DOMAIN:-$DEFAULT_DOMAIN})"
    cat > .env <<EOF
SOALM_DOMAIN=${SOALM_DOMAIN:-$DEFAULT_DOMAIN}
SOALM_CORS_ORIGINS=${SOALM_CORS_ORIGINS:-*}
EOF
fi
# shellcheck disable=SC1091
set -a; . ./.env; set +a

# -----------------------------------------------------------------------------
# 5. Build and start (sudo: the docker group only applies after re-login)
# -----------------------------------------------------------------------------
say "Building and starting SOALM + Caddy (first build takes several minutes)"
sudo docker compose --profile https up -d --build

# -----------------------------------------------------------------------------
# 6. Wait for the models to load
# -----------------------------------------------------------------------------
say "Waiting for the API to become healthy"
for _ in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:5000/api/health >/dev/null 2>&1; then
        curl -fsS http://127.0.0.1:5000/api/health; echo
        say "SOALM is live at: https://${SOALM_DOMAIN}/api/health"
        echo "If HTTPS doesn't answer yet, give Caddy ~30s to get its certificate."
        echo "Logs: sudo docker compose -f $APP_DIR/docker-compose.yml logs -f"
        exit 0
    fi
    sleep 5
done

echo "The API did not become healthy within 5 minutes. Recent logs:" >&2
sudo docker compose logs --tail=80 soalm >&2
exit 1
