# Deriv AI Trading Bot 🚀

Production-ready automated trading bot for **Deriv.com** Digits contracts (Over/Under/Even/Odd) powered by hybrid LSTM + XGBoost AI.

## Features
- **AI Predictions**: LSTM (time-series) + XGBoost hybrid with technical indicators
- **Strategies**: Martingale (with caps), Zuno (switch on loss), XML-configurable
- **Multi-market**: Real-time scanning (R_100, R_75, etc.)
- **Risk Management**: Daily loss limits, consecutive loss pauses, balance checks
- **Telegram**: Full notifications + commands (/status, /pause, /resume, /stats)
- **Deployment**: Docker + cloud-ready
- **Self-Optimizing**: Retraining from trade history

**⚠️ WARNING**: Trading bots involve significant financial risk. Use demo mode first. Paper trade extensively before live.

## Installation & Setup

1. **Clone the repo**
   ```bash
   git clone <your-repo>
   cd deriv-ai-bot
   ```

2. **Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configuration**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with:
   - `DERIV_APP_ID` & `DERIV_API_TOKEN` (from Deriv dashboard)
   - `TELEGRAM_BOT_TOKEN` & `TELEGRAM_CHAT_ID`

4. **Train AI Model**
   ```bash
   python scripts/train_model.py
   ```

5. **Test Connection**
   ```bash
   python scripts/test_connection.py
   ```

6. **Run Bot (Demo first!)**
   ```bash
   python src/main.py --mode demo
   ```

7. **Docker**
   ```bash
   docker-compose up --build
   ```

## Project Structure
(See original prompt for full details)

## Next Steps
- Collect historical data
- Backtest with `scripts/backtest.py` (coming)
- Monitor via Telegram

**Always start in demo mode!** Questions? Check logs in `data/logs/`.
