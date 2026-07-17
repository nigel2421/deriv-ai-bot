# Migrating to Deriv New OAuth / Developers API

## Why the old setup failed

| | Legacy | New OAuth web app |
|--|--------|-------------------|
| App ID | Numeric (`1089`) | Alphanumeric (`33R2Z6…`) |
| Token | API token from app.deriv.com | **PAT** or **OAuth access_token** |
| WebSocket | `wss://ws.derivws.com/…?app_id=NUMBER` + `{authorize: token}` | REST OTP → `wss://api.derivws.com/…/ws/demo?otp=…` |

This bot now **auto-detects**:

- numeric `DERIV_APP_ID` → **legacy** mode  
- alphanumeric `DERIV_APP_ID` → **v2** mode (your OAuth app)

## What you need on developers.deriv.com

1. App type: **OAuth** (web) **or PAT** (simpler for bots).
2. **Redirect URLs** (OAuth only):
   ```text
   https://deriv-ai-bot-842806243906.us-central1.run.app/
   https://deriv-ai-bot-842806243906.us-central1.run.app/oauth/callback
   http://localhost:8080/
   http://localhost:8080/oauth/callback
   ```
3. Scopes: include **trade** (and read/account as required).
4. Token:
   - **PAT app:** generate PAT in the developer dashboard → `DERIV_API_TOKEN`
   - **OAuth app:** use browser **Login** on the Cloud Run site (`/oauth/login`) so the bot stores an access token

## Env / secrets

```env
DERIV_APP_ID=33R2Z6MTElnIWrId8aH3m
DERIV_API_MODE=auto
DERIV_API_BASE=https://api.derivws.com
DERIV_OAUTH_REDIRECT_URI=https://deriv-ai-bot-842806243906.us-central1.run.app/oauth/callback
# Optional if you already have a PAT / access token:
DERIV_API_TOKEN=...
# Optional pin demo account:
# DERIV_ACCOUNT_ID=VRTC...
```

Cloud Run secrets still map:

- `DERIV_APP_ID` → secret `deriv-app-id`
- `DERIV_API_TOKEN` → secret `deriv-api-token` (optional if using OAuth login each deploy)

## Flow after deploy

1. Open https://deriv-ai-bot-842806243906.us-central1.run.app/
2. Click **OAuth Login**
3. Approve on Deriv
4. Callback stores token and restarts the trading loop
5. `/status` should show `"status": "running"` and a balance

## Limitations

- New Options API surface may differ for some contract types; ticks/proposal/buy still use classic JSON frames on the authenticated socket when supported.
- OAuth tokens on Cloud Run are stored under `data/oauth_token.json` (ephemeral unless you add a volume). Prefer a **PAT** for always-on bots.
- If OTP/accounts endpoints return 401/403, scopes or app type need adjustment in the developer portal.
