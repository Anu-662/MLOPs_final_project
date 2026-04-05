# Streamlit UI for the CESAR property valuation tool.
import os
import requests
import streamlit as st

API_URL = os.environ.get("CESAR_API_URL", "http://localhost:8000")

st.title("CESAR — Property valuation")
st.caption("Estimate the value of a property and check if the price is fair.")

# Sidebar shows API health
with st.sidebar:
    st.header("API status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code == 200 and health.json().get("status") == "ok":
            st.success("API is healthy")
            info = requests.get(f"{API_URL}/model_info", timeout=5).json()
            st.write(f"**Model version:** {info.get('model_version', '?')}")
            st.write(f"**Features:** {', '.join(info.get('feature_names', []))}")
        else:
            st.error("API returned unhealthy status")
    except requests.RequestException as e:
        st.error(f"Cannot reach API: {e}")

with st.form("estimate_form"):
    st.subheader("Property details")

    col1, col2 = st.columns(2)
    with col1:
        surface = st.number_input("Surface (m²)", min_value=1.0, value=50.0, step=5.0)
        pieces = st.number_input("Number of rooms", min_value=1.0, value=3.0, step=1.0)
    with col2:
        dept = st.text_input("Department code", value="75", max_chars=3)
        postal = st.text_input("Postal code", value="75015", max_chars=5)
        type_local = st.selectbox(
            "Property type",
            ["Appartement", "Maison", "Dépendance", "Local industriel. commercial ou assimilé"],
        )

    submitted = st.form_submit_button("Estimate value")

if submitted:
    payload = {
        "surface_reelle_bati": surface,
        "nombre_pieces_principales": pieces,
        "code_departement": dept,
        "code_postal": postal,
        "type_local": type_local,
    }

    try:
        resp = requests.post(f"{API_URL}/estimate/", json=payload, timeout=10)

        if resp.status_code == 200:
            data = resp.json()
            value = data["estimated_value_eur"]

            st.subheader("Estimate")
            st.metric("Estimated value", f"{value:,.0f} €")

            adequacy = data.get("price_adequacy")
            price_m2 = data.get("price_per_m2")
            if adequacy and price_m2:
                color_map = {
                    "underpriced": "green",
                    "fair": "blue",
                    "overpriced": "red",
                }
                color = color_map.get(adequacy, "gray")
                st.markdown(
                    f"**Price adequacy:** :{color}[{adequacy.capitalize()}] "
                    f"— {price_m2:,.0f} €/m²"
                )

        elif resp.status_code == 422:
            st.error(f"Invalid input: {resp.json().get('detail', resp.text)}")
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")

    except requests.RequestException as e:
        st.error(f"Could not reach API: {e}")
