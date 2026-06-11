---
title: WC26 Pitchside Hub
emoji: ⚽
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# WC26 Pitchside Hub
FIFA World Cup 2026 schedule, AI-refreshed live scores, a Monte Carlo
prediction engine, and on-demand match breakdowns. Frontend is a single
HTML file; FastAPI proxies Claude API calls server-side.

## Configuration (Space secrets)

| Secret | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Used server-side by `/api/claude` to call the Anthropic API. Never exposed to the browser. |
| `APP_PASSCODE` | Optional. If set, visitors must enter this passcode before the page loads (client-side gate, to limit who can trigger Claude API calls). Leave unset to disable the gate. |
