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
