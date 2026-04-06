# CESAR v2 — Property Valuation System

> **CentraleSupélec-ESSEC System for Asset Rating**
> MLOps Final Project — Extending v1

## About this project

For our MLOps final project, we extended the professor's CESAR system (v1 branch) into a production-ready property valuation platform. Rather than building from scratch, we took the existing prototype and improved it across three areas: model accuracy, API reliability, and deployment readiness. We simplified where complexity added no value (replaced the JavaScript UI with Streamlit), and added features where they made a real difference (target-encoded postal codes, automated data cleaning, request logging, and a CI pipeline).

## What CESAR does

CESAR estimates the market value of French properties using real transaction data from DVF (Demandes de Valeurs Foncières). You provide a few details about a property — surface area, number of rooms, department, postal code, and property type — and CESAR returns an estimated price along with a label indicating whether the price is underpriced, fair, or overpriced compared to the local market.

The system exposes a FastAPI-based REST API with endpoints for single predictions, batch predictions, health checks, model metadata, prediction monitoring, and department statistics. It also includes a Streamlit web interface for non-technical users and a monitoring dashboard for operators.

---

## Why it is useful

A property buyer or real estate analyst can use CESAR to:

- Get a rough valuation before visiting a property
- Check if an asking price seems reasonable for the area
- Compare properties across different postal codes and departments
- Price multiple properties at once using the batch endpoint

The price adequacy label is the most practical feature: instead of just "this costs 500,000€", CESAR says "this is overpriced for the 75th department" — which is what a decision-maker actually needs to know.

---

## What we improved in v2

We forked the professor's v1 branch and extended the system across three areas: model improvement, data quality, and deployment.

### Model & API (Anupama Ajith)

- **Added postal code as a new feature** using target encoding — each postal code is replaced with the average property price in that area, giving the model meaningful location information. This improved prediction accuracy by 7% MAE on 30,000+ transactions.
- **Fixed the `/health` endpoint** to actually verify that the model is loaded. The original always returned "ok" even when the model file was missing. Our version returns 503 when the model isn't ready, which is what Docker healthchecks and Kubernetes probes rely on.
- **Added `/model_info` endpoint** that returns the model version, feature list, and target variable. Useful for operators checking which model is deployed.
- **Added a price adequacy label** on every estimate — tells the user whether the predicted price is underpriced, fair, or overpriced relative to the department average.
- **Added `/estimate/batch` endpoint** for bulk valuation of up to 100 properties in a single API call.
- **Built request logging middleware** that records every API request to a CSV file with timestamp, endpoint, response time, payload, and client IP.
- **Built a data enrichment script** that validates postal codes, computes price per m² for each transaction, and generates department-level price statistics as a JSON file. The API loads these statistics for data-driven price adequacy instead of relying solely on hardcoded values.
- **Integrated data cleaning into the training pipeline** so the model always trains on deduplicated, outlier-free data.
- **Added 80,000+ rows of Paris transaction data** covering all 20 arrondissements for broader training coverage.
- Registered all team endpoints in the main API file — connected  prediction history endpoint and department statistics endpoint to the FastAPI app so all 7 endpoints work together as a unified API.

### Data Quality & Testing (Siya Sinha)

- **Built a data cleaning script** that handles the messy raw DVF data: removes duplicate lots from multi-lot transactions, drops rows with missing surface or value, removes price outliers using the IQR method, and fills missing room counts.
- **Built a model evaluation** comparing v1 (without postal code) vs v2 (with target-encoded postal code) using 5-fold cross-validation. The evaluation computes target encoding on training folds only to avoid data leakage.
- **Expanded acceptance tests** from 2 to 5 cases, covering edge cases like small studios, properties in different departments, and rare property types.
- **Set up GitHub Actions CI** that automatically trains, serves, and tests the API on every push. If anything breaks, the pull request shows a failure.
- **Added `/predictions/history` endpoint** that returns recent prediction logs for monitoring.

### Deployment & UI (Nilay Purayar)

- **Replaced the JavaScript frontend with a Streamlit UI** — a single Python file that anyone can run without Node.js or a build step.
- **Added a Docker Compose healthcheck** so the UI container only starts after the API is confirmed healthy.
- **Built a monitoring dashboard** showing API health, model metadata, and live prediction testing with response time measurement.
- **Added API documentation** with realistic request/response examples in the Swagger UI at `/docs`.
- **Built a Makefile** with shortcut commands for all common operations (`make train`, `make serve`, `make pipeline`, etc.).
- **Added `/stats/departments` endpoint** that shows the department averages and thresholds used for price adequacy, making the pricing logic fully transparent.

---

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns 200 if model is loaded, 503 if not |
| `/model_info` | GET | Returns model version, features, and target |
| `/estimate/` | POST | Single property valuation with price adequacy |
| `/estimate/batch` | POST | Batch valuation for up to 100 properties |
| `/predictions/history` | GET | Returns recent prediction logs |
| `/stats/departments` | GET | Shows department price averages and thresholds |
| `/docs` | GET | Interactive Swagger UI with examples |

---

## Screenshots

### Streamlit UI — Property Valuation
<!-- Screenshot of the Streamlit property form with estimate result -->
 *<img width="1774" height="940" alt="Screenshot 2026-04-06 at 6 17 50 PM" src="https://github.com/user-attachments/assets/c94c9707-900b-4c13-ac61-5d3ac8e148fd" />*


### Monitoring Dashboard
<!-- Screenshot of the monitoring dashboard showing health, model info, live test -->
*<img width="1793" height="916" alt="Screenshot 2026-04-06 at 6 17 28 PM" src="https://github.com/user-attachments/assets/a618d7d3-3882-48b6-93b8-91d48476539a" />*

### API Response Example
<!-- Screenshot of curl or /docs showing JSON response -->
*<img width="1745" height="640" alt="Screenshot 2026-04-06 at 6 17 34 PM" src="https://github.com/user-attachments/assets/5af890a0-b79c-4658-ba87-ff6125e7ec75" />*


---

## How to run

### Prerequisites
- Python 3.11+
- Docker (optional, for containerised deployment)

### Quick start (local)
```bash
# Install
python -m venv venv
source venv/bin/activate
pip install -e .

# Run full pipeline (clean → enrich → train)
make pipeline

# Start API
make serve

# Start UI (in another terminal)
make ui

# Start monitoring dashboard (in another terminal)
make monitoring
```

### Docker (one command)
```bash
make docker-up
# API:         http://localhost:8000
# API docs:    http://localhost:8000/docs
# UI:          http://localhost:8501
```

### Run tests
```bash
make test
```

### Run model evaluation
```bash
make evaluate
```

### All available commands
```bash
make help
```

---


## Known limitations

We want to be transparent about what CESAR cannot do and where it falls short:

1. **Target encoding can overfit.** If a postal code has only 2 transactions, its "average price" is unreliable. Smoothing techniques (blending with the global mean based on sample size) would reduce this risk but were not implemented.

2. **Arbitrary adequacy thresholds.** We use 0.85 and 1.15 as cutoffs for underpriced/overpriced. These have no statistical basis — a more rigorous approach would use percentiles from the actual price distribution in each department.

3. **Department averages ignore property type.** A house and an apartment in the same department have very different price profiles. The current implementation uses one average per department regardless of type.

4. **No confidence intervals.** The response schema has placeholder fields for `value_low_eur` and `value_high_eur`, but quantile regression is not implemented. The single-point estimate gives no indication of uncertainty.

5. **Training data is limited to Paris.** The model has seen transactions from Paris only. Predictions for other departments rely on limited generalisation ability.

6. **Static department averages.** While the API can load data-driven averages from a JSON file, these are only updated when the enrichment script is rerun manually. In production, this should be automated (e.g. quarterly).

---

## Future work

If CESAR were to continue development, these improvements would have the most impact:

1. **Expand training data** — add DVF CSVs from more departments to improve coverage and generalisation.
2. **Smoothed target encoding** — blend postal code averages with the global mean based on sample size to reduce overfitting on rare postal codes.
3. **Property-type-specific averages** — compute separate price per m² averages per department AND property type for more accurate adequacy labels.
4. **Confidence intervals** — implement quantile regression or use individual Random Forest tree predictions to estimate prediction uncertainty.
5. **Automated retraining** — schedule periodic retraining with fresh DVF data using Airflow or a cron job.
6. **CSV batch upload in the UI** — allow users to upload a CSV of properties and get all estimates in a downloadable table.
7. **Production monitoring** — replace the Streamlit dashboard with Prometheus + Grafana for real-time metrics and alerting.

---

## How it should be operated and maintained

1. **Retrain periodically.** DVF data is updated regularly. Run `make pipeline` monthly or quarterly with fresh data.
2. **Monitor prediction drift.** If the model consistently over- or under-estimates, the market has moved and retraining is needed. The request logs can be analysed for this.
3. **Update department averages.** Running `make enrich` recomputes averages from the latest data, which the API picks up on next restart.
4. **Check CI status.** Every push runs the full test suite. Red builds should be investigated before merging.
5. **Review request logs.** The `/predictions/history` endpoint and the CSV logs help identify unusual inputs or increasing error rates.

---

## Team contributions

| Team member | Area | Key contributions |
|-------------|------|-------------------|
| **Anupama Ajith** | Model, API & Data Enrichment | Target-encoded postal code feature, /health fix, /model_info, price adequacy, /estimate/batch, request logging, data enrichment, pipeline integration, endpoint registration for all team members, project documentation |
| **Siya Sinha** | Data Quality, Evaluation & CI | Data cleaning script, v1 vs v2 evaluation, acceptance tests, GitHub Actions CI, /predictions/history |
| **Nilay Purayar** | Deployment, UI & Developer Experience | Streamlit UI, monitoring dashboard, Docker healthcheck, API docs, Makefile, /stats/departments |

---

## AI usage disclosure

We used AI tools (primarily for documentation formatting, understanding certain FastAPI and scikit-learn concepts, and refining code structure). All design decisions — which features to add, which endpoints to create, how to structure the pipeline, and what to include in this documentation — were made by us. We understand and can explain every piece of code in this project.

Specifically, AI assisted with:
- Formatting and structuring documentation
- Understanding FastAPI middleware patterns and Pydantic model configuration
- Refining code comments for clarity
- Understanding target encoding and cross-validation best practices

All architectural decisions, feature choices, pipeline design, and the overall project direction were ours.
