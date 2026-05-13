# Cloud deployment — step by step

Three paths are documented. **GitHub Actions is the free path and the one we
recommend for first deployment.** VPS paths are kept for when you outgrow
Actions (sub-minute response time, custom hardware, etc.).

  Path 0 — GitHub Actions cron        ($0/mo, simplest, recommended)
  Path A — VPS + bare Python + systemd ($4-5/mo, more control)
  Path B — VPS + Docker container      ($4-5/mo, more portable)

----------------------------------------------------------------
## Path 0 — GitHub Actions (recommended for first deploy)

What you get: the bot runs once an hour, for ~30 seconds, on GitHub's free
runners. Total monthly usage is ~360 minutes — well under the 2,000-minute free
tier for private repos. Public repos are unlimited. No server to maintain.

### Step 0.1 — Create a GitHub repo

1. Go to https://github.com/new
2. Repository name: anything you want (e.g., `crypto-trading-bot`).
3. **Visibility: Private.** Strongly recommended — the workflow file is public,
   but a private repo at least keeps the strategy code out of search engines.
4. Don't add a README, .gitignore, or license (we already have them locally).
5. Create repository.

### Step 0.2 — Push the local code to GitHub

On your Windows machine, in this project directory:

```powershell
git init
git add .
git commit -m "Initial bot scaffold"
git branch -M main
git remote add origin https://github.com/<your-username>/crypto-trading-bot.git
git push -u origin main
```

If you've never used Git before, install Git for Windows first:
https://git-scm.com/download/win — accept all defaults.

(If `git push` asks for credentials, GitHub now requires a Personal Access
Token instead of a password — generate one at
https://github.com/settings/tokens with `repo` scope, paste it as the password.)

### Step 0.3 — Set repository secrets

In your repo on github.com:

1. **Settings → Secrets and variables → Actions → Secrets tab → New repository secret**
2. Add three secrets, one at a time:

   | Name                     | Value                            |
   | ------------------------ | -------------------------------- |
   | `BITGET_API_KEY`         | your Bitget demo API key         |
   | `BITGET_API_SECRET`      | your Bitget demo API secret      |
   | `BITGET_API_PASSPHRASE`  | your Bitget demo passphrase      |

3. Then switch to the **Variables** tab and add one repository VARIABLE
   (not secret):

   | Name        | Value     |
   | ----------- | --------- |
   | `BOT_LIVE`  | `false`   |

   Keep this at `false` while you watch the first day or two of dry-run output.

### Step 0.4 — Manually trigger the first run

The workflow is set to run hourly, but waiting an hour is annoying. Trigger it
manually to confirm everything works:

1. **Actions** tab → **Crypto trading bot** (in the left sidebar).
2. Click **Run workflow** (top-right).
3. Leave "Submit real orders?" as `false` for now.
4. Click the green **Run workflow** button.
5. Wait ~30s, then click into the run to watch the logs stream.

Expected output: the standard `STRATEGY EVALUATION` block, intent + diff,
dry-run order bodies, `[OK] Evaluation complete.`

### Step 0.5 — Verify the hourly schedule

Wait for the next top-of-hour. A new run should appear in the Actions tab
without you doing anything. (Note: GitHub's scheduled triggers can be delayed
5-15 minutes during peak load — this is documented behavior and fine for our
hourly cadence.)

### Step 0.6 — After a clean day or two, flip to live demo trading

1. **Settings → Secrets and variables → Actions → Variables**.
2. Edit `BOT_LIVE`, set value to `true`. Save.
3. Trigger a manual run (Path 0.4) to verify, or wait for the next hourly run.

The workflow output will include a warning banner reading
"Running in LIVE mode — orders will actually be submitted." Orders now appear
in your Bitget Demo UI.

To revert: set `BOT_LIVE` back to `false`. Takes effect on the next run.

### Step 0.7 — Where to find logs

Each run uploads its log files as a downloadable artifact. From the Actions tab,
open any run and scroll down to **Artifacts** → `bot-logs-<run-id>` — that's a
zip with `bot.log` and `heartbeat` exactly as they would be on a VPS.

### Step 0.8 — Discord notifications (optional, ~2 min)

Want a ping on your phone whenever the bot opens or closes a position, hits
an error, or sends the daily check-in? Add a Discord webhook.

What gets notified (only when `BOT_LIVE=true`):
  - Every live order placed (entry, exit, rotation, EMA in/out)
  - Cycle crashes and safety halts
  - One daily check-in at 00:00 UTC (equity, current positions)

What does NOT get notified (intentionally):
  - Successful hourly cycles where nothing changed — would be 24 pings/day of pure noise
  - Dry-run order bodies — log file is the right place for those

**Setup:**

1. In Discord, open the server + channel where you want notifications.
2. Channel settings (gear icon) → **Integrations** → **Webhooks** → **New Webhook**.
3. Name it (e.g., "bitget-bot"), optionally upload an icon, click **Copy Webhook URL**.
4. Back in your GitHub repo: **Settings → Secrets and variables → Actions → Secrets tab → New repository secret**.
   - Name:  `DISCORD_WEBHOOK_URL`
   - Value: paste the webhook URL
5. Save.

**Test it before flipping to live trading.** From your local machine:

```powershell
# Put the same webhook URL in your local .env file:
#   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

run.bat notify.py "hello from the bot"
```

A bright blue test embed should appear in the Discord channel within a second.
If you don't get one, the URL is wrong or your channel revoked the webhook —
re-copy and re-paste.

To stop notifications later: delete the GitHub secret, or revoke the webhook
in Discord. The bot keeps running normally; every `notify()` call becomes a
silent no-op.

----------------------------------------------------------------
## When to graduate from GitHub Actions to a VPS

GitHub Actions is great until one of these becomes true:

- You want sub-minute reaction time (Actions has 5-15 min cron jitter).
- You want to add a websocket connection for tighter monitoring (Actions
  runs are too short for that).
- You're putting real money on the line and want a process you can SSH
  into and debug live.

Until then, Actions is honestly the right call.

The VPS paths below remain available for that future graduation. They're also
documented for completeness — you might prefer a VPS if you'd rather have the
process up 24/7 as one long-running daemon rather than 720 short-lived runs
per month.

(VPS provider setup unchanged — same Hetzner / DigitalOcean steps as below.)

----------------------------------------------------------------
## VPS — still documented for later

This path takes you from a fresh VPS to a continuously running demo bot in
about 20 minutes. Cost: ~$4–5/month.

  Path A — bare Python + systemd      (simplest VPS path)
  Path B — Docker container           (more portable)

----------------------------------------------------------------
## Step 1 — Create the VPS

Use whichever provider you prefer. Both of these are good for our use case
and accept Philippines-issued payment cards.

### Hetzner Cloud (cheapest, recommended)
1. Sign up: https://www.hetzner.com/cloud
2. Console → Add Server
   - Location: **Singapore** (lowest latency to Philippines + Bitget servers)
   - Image: **Ubuntu 22.04**
   - Type: **CX22** (2 vCPU / 4 GB / €4.51/mo) — the smallest CX11 is also fine
     for our bot but CX22 gives room to grow.
   - SSH Key: paste your public key, or use password if you don't have one yet
   - Click **Create & Buy Now**
3. Note the **public IPv4 address** shown after creation.

### DigitalOcean (friendlier UX, slightly pricier)
1. Sign up: https://www.digitalocean.com
2. Create → Droplets
   - Region: **Singapore (SGP1)**
   - Image: **Ubuntu 22.04**
   - Plan: **Basic Regular $4/mo (512 MB / 1 vCPU)** — works, or $6 for 1 GB
   - Authentication: SSH key preferred
   - Click **Create Droplet**
3. Note the **public IPv4** shown on the droplet page.

----------------------------------------------------------------
## Step 2 — SSH into your VPS

From your Windows machine, open PowerShell or Command Prompt:

    ssh root@<your-server-ip>

(If you set a password instead of a key, you'll be prompted for it.)
You should land at a shell like `root@your-server:~#`.

----------------------------------------------------------------
## Step 3 — Initial setup (run these once on the server)

```bash
# Get the latest package list and upgrade
apt update && apt upgrade -y

# Create a dedicated non-root user (don't run the bot as root)
adduser --disabled-password --gecos "" bot
usermod -aG sudo bot

# Allow your SSH key for the new user
mkdir -p /home/bot/.ssh
cp /root/.ssh/authorized_keys /home/bot/.ssh/
chown -R bot:bot /home/bot/.ssh
chmod 700 /home/bot/.ssh
chmod 600 /home/bot/.ssh/authorized_keys

# Set the system clock to UTC + enable time sync (matters for API signing)
timedatectl set-timezone UTC
apt install -y systemd-timesyncd
systemctl enable --now systemd-timesyncd
```

----------------------------------------------------------------
## Step 4 — Upload the bot code

From your **local Windows machine** (in PowerShell, from inside the
`C:\Users\Lenovo\Desktop\crypto_trading` directory):

```powershell
# Easiest: zip the project (excluding .env and caches) then scp it
Compress-Archive -Path *.py,requirements.txt,Dockerfile,run.bat -DestinationPath crypto_bot.zip
scp crypto_bot.zip bot@<your-server-ip>:/home/bot/
```

Back on the **server**, as `bot` user (`ssh bot@<server-ip>`):

```bash
cd ~
mkdir -p crypto_trading
cd crypto_trading
unzip ../crypto_bot.zip
mkdir -p data logs
```

----------------------------------------------------------------
## Step 5 — Provide credentials

The bot reads `.env` for Bitget credentials. **Never paste this file into a
remote terminal session that you don't trust** — type it locally instead.

```bash
# Create .env on the server with your DEMO credentials
nano .env
```

Paste this template, fill in your real values, then Ctrl+O / Enter / Ctrl+X:

    BITGET_ENV=demo
    BITGET_API_KEY=...your demo key...
    BITGET_API_SECRET=...your demo secret...
    BITGET_API_PASSPHRASE=...your demo passphrase...

Lock it down:

```bash
chmod 600 .env       # only the 'bot' user can read it
```

----------------------------------------------------------------
## Path A — bare Python + systemd

### Install Python deps

```bash
sudo apt install -y python3.11 python3.11-venv python3-pip
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Verify connection

```bash
python check_connection.py
```

You should see `[OK] All checks passed.` If not, see the troubleshooting section
in the main project notes.

### Run one dry-run pass

```bash
python live_bot.py
```

If the output looks sensible, you're ready for continuous mode.

### Create a systemd service

As **root** (or with `sudo`):

```bash
sudo tee /etc/systemd/system/crypto-bot.service > /dev/null <<'EOF'
[Unit]
Description=Crypto carry+EMA bot (Bitget demo)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=bot
WorkingDirectory=/home/bot/crypto_trading
ExecStart=/home/bot/crypto_trading/venv/bin/python /home/bot/crypto_trading/live_bot.py --loop
Restart=on-failure
RestartSec=30
StandardOutput=append:/home/bot/crypto_trading/logs/systemd.log
StandardError=append:/home/bot/crypto_trading/logs/systemd.log

[Install]
WantedBy=multi-user.target
EOF
```

Notice: this runs WITHOUT `--live`. Keep it in dry-run for the first 24 hours,
read the logs, then flip to live (see below).

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now crypto-bot.service
```

### Verify it's running

```bash
sudo systemctl status crypto-bot          # green = good
tail -f /home/bot/crypto_trading/logs/bot.log
```

You should see one cycle immediately, then a `Next wake: ...` line, then another
cycle on the next hour. Ctrl+C to stop tailing (the bot keeps running).

### Flip to live demo trading

After 24+ hours of clean dry-run output:

```bash
sudo systemctl stop crypto-bot
sudo sed -i 's/--loop$/--loop --live/' /etc/systemd/system/crypto-bot.service
sudo systemctl daemon-reload
sudo systemctl start crypto-bot
tail -f /home/bot/crypto_trading/logs/bot.log
```

The next cycle will actually submit orders to the Bitget demo account.

----------------------------------------------------------------
## Path B — Docker

Install Docker once:

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker bot
exit       # log out and back in so the group change takes effect
```

Build and run:

```bash
ssh bot@<server-ip>
cd ~/crypto_trading
docker build -t crypto-bot .

# Dry-run, continuous loop — logs visible
docker run --rm \
  --name crypto-bot \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  crypto-bot

# Or detached, restarts on failure or reboot
docker run -d \
  --name crypto-bot \
  --restart unless-stopped \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  crypto-bot \
  python live_bot.py --loop
```

To go live, append `--live`:

```bash
docker stop crypto-bot
docker rm crypto-bot
docker run -d --name crypto-bot --restart unless-stopped --env-file .env \
  -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs \
  crypto-bot python live_bot.py --loop --live
```

Watch the logs at any time:

```bash
docker logs -f crypto-bot
# or
tail -f ~/crypto_trading/logs/bot.log
```

----------------------------------------------------------------
## Operations cheat sheet

| What you want to do                | Command (systemd)                              | Command (docker)                          |
| ---------------------------------- | ---------------------------------------------- | ----------------------------------------- |
| See live logs                      | `tail -f ~/crypto_trading/logs/bot.log`        | `docker logs -f crypto-bot`               |
| Stop the bot                       | `sudo systemctl stop crypto-bot`               | `docker stop crypto-bot`                  |
| Start it again                     | `sudo systemctl start crypto-bot`              | `docker start crypto-bot`                 |
| Check it's alive                   | `sudo systemctl status crypto-bot`             | `docker ps`                               |
| Check last heartbeat               | `cat ~/crypto_trading/logs/heartbeat`          | same                                      |
| Flat all positions urgently        | Bitget UI → Demo Trading → Close All Positions | same                                      |
| Update bot code                    | scp new zip, `unzip -o`, restart service       | re-build image, recreate container        |

The heartbeat file's mtime is the simplest health check: if it hasn't been
touched in ≥2 hours, the bot has stalled. The Docker image's HEALTHCHECK
does this automatically; for systemd you can check by hand or add a cron line
that emails you on staleness.

----------------------------------------------------------------
## Security checklist

- [ ] `.env` is `chmod 600` (only the bot user can read it)
- [ ] Bitget API key has **Trade** permission only — **not** Withdraw
- [ ] Bitget API key has IP whitelist set to the VPS public IP
- [ ] SSH is key-based, not password (disable password auth in `/etc/ssh/sshd_config`: `PasswordAuthentication no`)
- [ ] Firewall: `ufw allow OpenSSH && ufw enable` (nothing else needs an open port)
- [ ] You ran `apt update && apt upgrade` recently
- [ ] You've verified you can log into Bitget Demo on the web and see what the bot is doing
