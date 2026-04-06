# Monitoring dashboard for the CESAR API.
import os
import time
import requests
import streamlit as st

API_URL = os.environ.get("CESAR_API_URL", "http://localhost:8000")

st.set_page_config(page_title="CESAR Monitoring", layout="wide")
st.title("CESAR — Monitoring dashboard")
st.caption("Operational view: health, model info, and live testing.")

col_health, col_model = st.columns(2)

with col_health:
    st.subheader("API health")
    try:
        start = time.time()
        resp = requests.get(f"{API_URL}/health", timeout=5)
        latency_ms = (time.time() - start) * 1000

        if resp.status_code == 200 and resp.json().get("status") == "ok":
            st.success(f"Healthy — responded in {latency_ms:.0f} ms")
        else:
            st.error(f"Unhealthy — status {resp.status_code}")
            st.json(resp.json())
    except requests.RequestException as e:
        st.error(f"Cannot reach API at {API_URL}")
        st.text(str(e))

with col_model:
    st.subheader("Model info")
    try:
        resp = requests.get(f"{API_URL}/model_info", timeout=5)
        if resp.status_code == 200:
            info = resp.json()
            st.metric("Model version", info.get("model_version", "unknown"))
            st.write(f"**Target:** {info.get('target_name', '?')}")
            st.write(f"**Features ({len(info.get('feature_names', []))}):** "
                     f"{', '.join(info.get('feature_names', []))}")
            st.write(f"**Property types:** {', '.join(info.get('type_local_categories', []))}")
        else:
            st.warning(f"/model_info returned {resp.status_code}")
    except requests.RequestException:
        st.warning("Could not fetch model info")

st.divider()

st.subheader("Live prediction test")
st.caption("Send a sample request to the API and see the response + timing.")

col_input, col_result = st.columns(2)

with col_input:
    test_surface = st.number_input("Surface (m²)", value=50.0, step=5.0, key="mon_surface")
    test_pieces = st.number_input("Rooms", value=3.0, step=1.0, key="mon_pieces")
    test_dept = st.text_input("Department", value="75", key="mon_dept")
    test_postal = st.text_input("Postal code", value="75015", key="mon_postal")
    test_type = st.selectbox("Type", ["Appartement", "Maison", "Dépendance",
                                       "Local industriel. commercial ou assimilé"],
                              key="mon_type")
    run_test = st.button("Send test request")

with col_result:
    if run_test:
        payload = {
            "surface_reelle_bati": test_surface,
            "nombre_pieces_principales": test_pieces,
            "code_departement": test_dept,
            "code_postal": test_postal,
            "type_local": test_type,
        }

        try:
            start = time.time()
            resp = requests.post(f"{API_URL}/estimate/", json=payload, timeout=10)
            elapsed_ms = (time.time() - start) * 1000

            if resp.status_code == 200:
                data = resp.json()
                st.metric("Estimated value", f"{data['estimated_value_eur']:,.0f} €")
                st.metric("Response time", f"{elapsed_ms:.0f} ms")

                adequacy = data.get("price_adequacy")
                price_m2 = data.get("price_per_m2")
                if adequacy and price_m2:
                    color_map = {"underpriced": "green", "fair": "blue", "overpriced": "red"}
                    color = color_map.get(adequacy, "gray")
                    st.markdown(f"**Adequacy:** :{color}[{adequacy.capitalize()}] — {price_m2:,.0f} €/m²")

                with st.expander("Raw API response"):
                    st.json(data)
            else:
                st.error(f"API error {resp.status_code}")
                st.text(resp.text)

        except requests.RequestException as e:
            st.error(f"Request failed: {e}")

st.divider()

st.subheader("API endpoints")

endpoints = {
    "GET /health": "Returns {\"status\": \"ok\"} if model is loaded, 503 otherwise. "
                   "Used by Docker healthcheck and K8s probes.",
    "GET /model_info": "Returns model version, feature names, and target name. "
                       "Useful for checking which model is deployed.",
    "POST /estimate/": "Main prediction endpoint. Send property details, get estimated value "
                       "and price adequacy label.",
    "GET /docs": f"Interactive API documentation (Swagger UI) at {API_URL}/docs",
}

for endpoint, description in endpoints.items():
    st.markdown(f"**`{endpoint}`** — {description}")
