MarketMind

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Gemini AI](https://img.shields.io/badge/Gemini_AI-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Upload a CSV of your sales data and get a full analytics dashboard — charts, forecasts, anomaly detection, customer segments, and an AI that can write you an executive report or just answer questions about your numbers.

---

What it does

Drop in a CSV and you get:

- KPI cards with period-over-period deltas
- Revenue chart with a 7-day EMA and automatic anomaly flagging
- ROAS trend, ad spend vs revenue scatter, dual-axis revenue + orders
- Correlation heatmap, waterfall chart, RFM customer segmentation
- 30-day forecast with confidence intervals and a bear/base/bull scenario planner
- A health score (0–100) that grades your business across five criteria
- Gemini-powered executive report, auto insights, and a Q&A that knows your data

The AI features are optional — everything else works without an API key.

---

Setup

```bash
git clone https://github.com/YOUR_USERNAME/MarketMind.git
cd MarketMind
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For the AI features, grab a free Gemini key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and add it:

```bash
cp .env.example .env
# open .env and set GEMINI_API_KEY=your-key
```

Then run:

```bash
streamlit run MarketMind.py
```

You can also just paste the key in the sidebar at runtime — no restart needed.

---

CSV format

The app auto-detects whatever columns you have. It works with any CSV that has at least one numeric column. For the full experience, these columns are used:

| Column | Description |
|---|---|
| `date` | YYYY-MM-DD |
| `revenue` | Daily revenue |
| `orders` | Order count |
| `ad_spend` | Marketing spend |
| `avg_order_value` | Revenue ÷ orders |
| `discount_spend` | Promo/discount spend |
| `returns` | Return count |
| `conversion_rate` | Sessions-to-order % |
| `sessions` | Website sessions |
| `new_customers` | First-time buyers |
| `repeat_customers` | Returning buyers |

Not sure what to upload? Download the sample dataset from the sidebar — it's a synthetic 90-day retail store.

---

Project layout

```
MarketMind/
├── MarketMind.py           # the app
├── requirements.txt
├── generate_sample.py      # regenerate the sample CSV if needed
├── sample_data/
│   └── retail_sample.csv
└── utils/
    ├── analytics.py        # health score, forecasting, RFM, anomaly detection
    ├── ai_engine.py        # Gemini streaming (report, Q&A, insights)
    ├── charts.py           # all Plotly figures
    └── styles.py           # CSS
```

---

Deploying

Works on Streamlit Community Cloud out of the box. Push to GitHub, connect the repo at [share.streamlit.io](https://share.streamlit.io), then add your key under **Settings → Secrets**:

```toml
GEMINI_API_KEY = "AIzaSy-..."
```

---
 Tweaking things

**Change the AI model** — edit `_MODEL` in `utils/ai_engine.py`:
```python
_MODEL = "gemini-1.5-pro"    # default
_MODEL = "gemini-1.5-flash"  # faster, cheaper
_MODEL = "gemini-2.0-flash"  # latest
```

Add a chart** — write a function in `utils/charts.py` that returns a `go.Figure`, call it wherever you want in `MarketMind.py`.

Extend the health score** — add criteria inside `compute_health_score()` in `utils/analytics.py`.

---
License

MIT
