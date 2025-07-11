import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Customer Behavior Tool", layout="wide")
st.title("📊 Customer Behavior Dashboard")

file = st.file_uploader("📁 Upload your customer CSV file", type="csv")

if file:
    df = pd.read_csv(file)
    st.subheader("🧾 Data Preview")
    st.dataframe(df.head())

    # ---------- Analyze RFM ----------
    if st.button("📈 Analyze Behavior (RFM Score)"):
        try:
            response = requests.post("http://localhost:8000/score", json=df.to_dict(orient="records"))


            st.code(response.text)  # ✅ Show raw response from API

            if response.status_code == 200:
                result = response.json()

                if isinstance(result, list):
                    rfm_scores = pd.DataFrame(result)
                    st.success("✅ RFM Scoring Completed")
                    st.dataframe(rfm_scores.head())
                elif isinstance(result, dict) and "error" in result:
                    st.error(f"❌ API Error: {result['error']}")
                else:
                    st.warning("⚠️ Unexpected response format.")
                    st.write(result)
            else:
                st.error(f"❌ Server Error: {response.status_code}")
                st.code(response.text)

        except Exception as e:
            st.error(f"❌ Connection Error: {e}")

    # ---------- Churn Prediction ----------
    if st.button("💀 Predict Customer Churn"):
        try:
            response = requests.post("http://localhost:8000/churn", json=df.to_dict(orient="records"))

            st.code(response.text)  # ✅ Show raw response

            if response.status_code == 200:
                result = response.json()

                if isinstance(result, list):
                    churn_df = pd.DataFrame(result)
                    st.success("✅ Churn Prediction Completed")
                    st.dataframe(churn_df.head())
                elif isinstance(result, dict) and "error" in result:
                    st.error(f"❌ API Error: {result['error']}")
                else:
                    st.warning("⚠️ Unexpected response format.")
                    st.write(result)
            else:
                st.error(f"❌ Server Error: {response.status_code}")
                st.code(response.text)

        except Exception as e:
            st.error(f"❌ Connection Error: {e}")
