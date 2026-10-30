# Deploying SOALM V2 as a service

This turns the Flask app (`soalm_api.py`) into a production HTTP service your
website can call. It serves the API with [Waitress](https://docs.pylonsproject.org/projects/waitress/)
instead of the Flask dev server.

## Endpoints

| Method | Path                   | Purpose                                  |
| ------ | ---------------------- | ---------------------------------------- |
| GET    | `/api/health`          | Liveness/readiness + model count         |
| GET    | `/api/models`          | Model metadata                           |
| GET    | `/api/routes`          | List of available routes                 |
| POST   | `/api/predict`         | Single-point rainfall/flood prediction   |
| POST   | `/api/spatial-predict` | Grid of predictions around a point       |
| POST   | `/api/spatial`         | Alias of `/api/spatial-predict`          |
| POST   | `/api/what-if`         | Prediction with modified input features  |

> The `/api/chat` LLM route has been removed from this deployment. It still
> exists as a stub that returns HTTP 503 (`chat disabled`) so old clients get a
> clear response; no LLM backend is required or run.

## Configuration (environment variables)

| Variable             | Default                  | Notes                                          |
| -------------------- | ------------------------ | ---------------------------------------------- |
| `SOALM_HOST`         | `0.0.0.0`                | Bind address                                   |
| `SOALM_PORT`         | `5000`                   | Listen port                                    |
| `SOALM_THREADS`      | `8`                      | Waitress worker threads                        |
| `SOALM_CORS_ORIGINS` | `*`                      | Comma-separated allowed origins for `/api/*`   |

In production set `SOALM_CORS_ORIGINS` to your real site, e.g.
`https://yoursite.com,https://www.yoursite.com`, so the browser only allows
your frontend to call the API.

---

## Option 1 — Run locally with Waitress (no Docker)

```powershell
pip install -r requirements-service.txt
python serve.py
```

The service comes up on http://localhost:5000. Test it:

```powershell
curl http://localhost:5000/api/health
```

---

## Option 2 — Docker (recommended for hosting)

Build and run the prediction service (no LLM):

```powershell
docker compose up --build soalm
```

Or with the Dockerfile directly:

```powershell
docker build -t soalm-service .
docker run -p 5000:5000 -e SOALM_CORS_ORIGINS="https://yoursite.com" soalm-service
```

---

## Where to host it

The container runs anywhere that accepts a Docker image:

- **Render / Railway / Fly.io** — push the repo or image; set the env vars above.
- **A VPS (DigitalOcean, Hetzner, EC2)** — `docker compose up -d soalm`.

The prediction service is light: it loads 29 models (loaded once at
startup) and the `weather/weather.json` offline cache. Only
`data/era5/models_v2_final/` and `weather/weather.json` are needed at runtime;
the `.dockerignore` excludes the large training/checkpoint data.

---

## Calling it from your website

```js
const res = await fetch("https://your-service-host/api/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ latitude: 19.07, longitude: 72.87 }),
});
const data = await res.json();
```

Put the service behind HTTPS (a reverse proxy like Caddy/Nginx, or the
platform's built-in TLS) before pointing a production site at it.

---

## Option 3 — Oracle Cloud Always Free (free, always on)

The service needs about 1.3 GB RAM (1.6 GB peak), so 512 MB free tiers are too
small. An OCI Always Free Ampere A1 VM (up to 2 OCPU / 12 GB) runs it for free.

### 1. Create the VM (OCI console)

1. **Compute → Instances → Create instance**
2. Image: **Canonical Ubuntu 24.04** (or 22.04)
3. Shape: **Ampere → VM.Standard.A1.Flex**, 1–2 OCPU, 6–12 GB RAM
   (marked "Always Free-eligible")
4. Networking: keep "Assign a public IPv4 address" on
5. SSH keys: download the private key (or paste your own public key)
6. Create, then copy the instance's **Public IP**

If you get "Out of capacity", try another availability domain or retry later.

### 2. Open ports 80 and 443 (OCI console)

Instance → **Subnet** → **Security List** → **Add Ingress Rules**:

| Source CIDR | Protocol | Destination port |
| ----------- | -------- | ---------------- |
| 0.0.0.0/0   | TCP      | 80               |
| 0.0.0.0/0   | TCP      | 443              |

The setup script opens the same ports in the VM's own firewall.

### 3. Run the setup script (on the VM)

```powershell
ssh -i path\to\private.key ubuntu@<PUBLIC_IP>
```

```bash
curl -fsSL https://raw.githubusercontent.com/SparrowSyntax3210/SOALM/main/deploy/oracle-setup.sh | bash
```

It installs Docker, clones the repo, builds the image, and starts SOALM behind
Caddy with a free Let's Encrypt certificate at `https://<ip-with-dashes>.sslip.io`.

### Updating / settings

- Deploy new code: push to GitHub, then re-run the same command on the VM.
- Change domain or CORS: edit `~/SOALM/.env`, then
  `cd ~/SOALM && sudo docker compose --profile https up -d`
- Logs: `cd ~/SOALM && sudo docker compose logs -f soalm`
