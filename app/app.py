import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import io

# Set page configuration
st.set_page_config(
    page_title="Startup Success Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    h1, h2, h3 {
        color: #1e3d59;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #1e3d59;
    }
    .metric-label {
        font-size: 16px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(MODELS_DIR, 'model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    with open(os.path.join(MODELS_DIR, 'deployment_config.json'), 'r') as f:
        config = json.load(f)
    return model, scaler, label_encoder, config

@st.cache_data
def load_insights_data():
    try:
        # Load a sample to avoid memory issues in Streamlit
        df = pd.read_csv(os.path.join(DATA_DIR, 'train_processed.csv'))
        return df
    except Exception:
        return None

try:
    model, scaler, label_encoder, config = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

def preprocess_dataframe(df, config, scaler):
    df = df.copy()
    # 1. Fill missing
    df['funding_total_usd'] = df.get('funding_total_usd', pd.Series(dtype=float)).fillna(config["numerical_medians"]["funding_total_usd"])
    df['funding_rounds'] = df.get('funding_rounds', pd.Series(dtype=float)).fillna(1.0)
    df['company_age'] = df.get('company_age', pd.Series(dtype=float)).fillna(config["numerical_medians"]["company_age"])
    df['funding_duration_days'] = df.get('funding_duration_days', pd.Series(dtype=float)).fillna(config["numerical_medians"]["funding_duration_days"])
    df['time_to_first_funding_days'] = df.get('time_to_first_funding_days', pd.Series(dtype=float)).fillna(config["numerical_medians"]["time_to_first_funding_days"])
    df['country_code'] = df.get('country_code', pd.Series(dtype=str)).fillna(config["mode_country"])
    df['primary_category'] = df.get('primary_category', pd.Series(dtype=str)).fillna("Unknown")
    
    # 2. Capping
    for feature, bounds in config["capping_bounds"].items():
        if feature in df.columns:
            df[feature] = df[feature].clip(lower=bounds["lower"], upper=bounds["upper"])
            
    # 3. Create DataFrame for Dummy Encoding matching `feature_columns`
    processed_dict = {col: np.zeros(len(df)) for col in config["feature_columns"]}
    
    for feature in ["funding_total_usd", "funding_rounds", "company_age", "funding_duration_days", "time_to_first_funding_days"]:
        processed_dict[feature] = df[feature].values
        
    for idx, row in df.reset_index(drop=True).iterrows():
        country_col = f"country_code_{row['country_code']}"
        if country_col in processed_dict:
            processed_dict[country_col][idx] = 1.0
            
        category_col = f"primary_category_{row['primary_category']}"
        if category_col in processed_dict:
            processed_dict[category_col][idx] = 1.0
            
    processed_df = pd.DataFrame(processed_dict)
    scaled = scaler.transform(processed_df)
    return scaled


# Application Header
st.markdown("<h1 style='text-align: center;'>🚀 Startup Success Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #6c757d;'>AI-powered analysis to predict if your startup will thrive.</p>", unsafe_allow_html=True)
st.divider()

# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3233/3233483.png", width=150)
    st.markdown("### 🧭 Navigation")
    view = st.radio("", ["🎯 Single Prediction", "📁 Bulk Prediction", "📈 Data Insights"])
    st.markdown("---")
    st.info("Built with Machine Learning for high accuracy predictions.")


# ==========================================
# VIEW 1: SINGLE PREDICTION
# ==========================================
if view == "🎯 Single Prediction":
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 🏢 Company Profile")
        categories = config["top_20_categories"] + ["Other", "Unknown"]
        primary_category = st.selectbox("Primary Category", options=categories, help="The main sector the startup operates in.")
        
        countries = config["top_10_countries"] + ["Other"]
        country_code = st.selectbox("Country", options=countries, help="The country where the startup is based.")
        
        company_age = st.number_input("Company Age (Years)", min_value=0.0, max_value=100.0, value=float(config["numerical_medians"]["company_age"]), step=1.0)

    with col2:
        st.markdown("### 💰 Financial Data")
        funding_total_usd = st.number_input("Total Funding (USD)", min_value=0.0, value=float(config["numerical_medians"]["funding_total_usd"]), step=100000.0, format="%f")
        funding_rounds = st.number_input("Number of Funding Rounds", min_value=1, value=1, step=1)
        funding_duration_days = st.number_input("Funding Duration (Days)", min_value=0.0, value=float(config["numerical_medians"]["funding_duration_days"]), step=10.0)
        time_to_first_funding_days = st.number_input("Time to First Funding (Days)", min_value=0.0, value=float(config["numerical_medians"]["time_to_first_funding_days"]), step=10.0)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 Predict Success"):
        st.session_state['predict_clicked'] = True

    if st.session_state.get('predict_clicked', False):
        raw_dict = {
            "funding_total_usd": funding_total_usd, "funding_rounds": funding_rounds,
            "company_age": company_age, "funding_duration_days": funding_duration_days,
            "time_to_first_funding_days": time_to_first_funding_days, "country_code": country_code,
            "primary_category": primary_category
        }
        
        raw_df = pd.DataFrame([raw_dict])
        
        with st.spinner("Analyzing startup profile..."):
            scaled_features = preprocess_dataframe(raw_df, config, scaler)
            
            prediction_encoded = model.predict(scaled_features)[0]
            prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
            
            try:
                probabilities = model.predict_proba(scaled_features)[0]
                confidence = max(probabilities) * 100
            except AttributeError:
                confidence = None

            # Results
            st.divider()
            st.markdown("<h2 style='text-align: center;'>Analysis Result</h2>", unsafe_allow_html=True)
            
            res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
            with res_col2:
                if 'success' in prediction_label.lower() or 'operating' in prediction_label.lower():
                    st.markdown("""
                    <div style="background-color: #d4edda; color: #155724; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745; text-align: center;">
                        <h1 style="margin: 0;">🌟 SUCCESS</h1>
                        <p style="font-size: 18px; margin-top: 10px;">High potential for success.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""
                    <div style="background-color: #f8d7da; color: #721c24; padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545; text-align: center;">
                        <h1 style="margin: 0;">⚠️ {prediction_label.upper()}</h1>
                        <p style="font-size: 18px; margin-top: 10px;">High risk based on current parameters.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                if confidence is not None:
                    st.markdown(f"""
                    <div class="metric-card" style="margin-top: 20px;">
                        <div class="metric-label">Model Confidence</div>
                        <div class="metric-value">{confidence:.1f}%</div>
                        <progress value="{confidence}" max="100" style="width: 100%; height: 20px;"></progress>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Download Report CSV
                report_dict = raw_dict.copy()
                report_dict["Prediction"] = prediction_label
                report_dict["Confidence (%)"] = round(confidence, 2) if confidence else "N/A"
                report_csv = pd.DataFrame([report_dict]).to_csv(index=False)
                st.download_button(label="📥 Download Full Report (CSV)", data=report_csv, file_name="startup_prediction_report.csv", mime="text/csv", use_container_width=True)

            # --- Analysis Section ---
            st.markdown("<br><hr>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center; color: #1e3d59;'>📊 Analysis & Explanation</h3>", unsafe_allow_html=True)
            
            ana_col1, ana_col2 = st.columns(2)
            
            with ana_col1:
                st.markdown("#### 1. Feature Importance")
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    imp_df = pd.DataFrame({"Feature": config["feature_columns"], "Importance": importances}).sort_values(by="Importance", ascending=False).head(7)
                    st.bar_chart(imp_df.set_index("Feature"))
                    
            with ana_col2:
                st.markdown("#### 2. Your Startup vs. Median")
                comp_data = {
                    "Metric": ["Funding", "Age", "Funding Duration", "Time to First Funding"],
                    "Your Startup": [funding_total_usd, company_age, funding_duration_days, time_to_first_funding_days],
                    "Industry Median": [
                        config["numerical_medians"]["funding_total_usd"],
                        config["numerical_medians"]["company_age"],
                        config["numerical_medians"]["funding_duration_days"],
                        config["numerical_medians"]["time_to_first_funding_days"]
                    ]
                }
                comp_df = pd.DataFrame(comp_data).set_index("Metric")
                comp_df["Your Startup (Normalized)"] = comp_df["Your Startup"] / comp_df["Industry Median"].replace(0, 1)
                comp_df["Industry Median (Normalized)"] = 1.0
                st.bar_chart(comp_df[["Your Startup (Normalized)", "Industry Median (Normalized)"]])
                st.caption("*Normalized relative to the industry median (Median = 1.0)*")

        # --- WHAT-IF SIMULATOR ---
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🎛️ What-If Simulator (Test different scenarios)"):
            st.markdown("Adjust the values below to see how they impact your success probability instantly.")
            
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                sim_funding = st.slider("Simulate Total Funding", min_value=0.0, max_value=float(config["capping_bounds"]["funding_total_usd"]["upper"]), value=float(funding_total_usd), step=500000.0)
                sim_rounds = st.slider("Simulate Funding Rounds", min_value=1, max_value=10, value=int(funding_rounds))
            with sim_col2:
                sim_age = st.slider("Simulate Company Age", min_value=0.0, max_value=20.0, value=float(company_age), step=1.0)
                sim_time_first = st.slider("Simulate Time to 1st Funding", min_value=0.0, max_value=2000.0, value=float(time_to_first_funding_days), step=30.0)
                
            sim_df = pd.DataFrame([{
                "funding_total_usd": sim_funding, "funding_rounds": sim_rounds,
                "company_age": sim_age, "funding_duration_days": funding_duration_days,
                "time_to_first_funding_days": sim_time_first, "country_code": country_code,
                "primary_category": primary_category
            }])
            
            sim_scaled = preprocess_dataframe(sim_df, config, scaler)
            sim_pred = label_encoder.inverse_transform([model.predict(sim_scaled)[0]])[0]
            try:
                sim_conf = max(model.predict_proba(sim_scaled)[0]) * 100
                st.success(f"**Simulated Outcome:** {sim_pred.upper()} (Confidence: {sim_conf:.1f}%)")
            except AttributeError:
                st.success(f"**Simulated Outcome:** {sim_pred.upper()}")


# ==========================================
# VIEW 2: BULK PREDICTION
# ==========================================
elif view == "📁 Bulk Prediction":
    st.markdown("### 📁 Bulk Prediction (CSV Upload)")
    st.write("Upload a CSV file containing data for multiple startups to get predictions for all of them at once.")
    
    # Template CSV
    template_data = {
        "funding_total_usd": [1500000, 50000],
        "funding_rounds": [2, 1],
        "company_age": [4, 1],
        "funding_duration_days": [365, 0],
        "time_to_first_funding_days": [200, 50],
        "country_code": ["USA", "GBR"],
        "primary_category": ["Software", "E-Commerce"]
    }
    template_df = pd.DataFrame(template_data)
    csv_template = template_df.to_csv(index=False)
    
    st.download_button("📝 Download CSV Template", data=csv_template, file_name="startup_template.csv", mime="text/csv")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            bulk_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(bulk_df)} records.")
            
            with st.spinner("Processing data & making predictions..."):
                bulk_scaled = preprocess_dataframe(bulk_df, config, scaler)
                bulk_preds = model.predict(bulk_scaled)
                bulk_labels = label_encoder.inverse_transform(bulk_preds)
                
                bulk_df["AI_Prediction"] = bulk_labels
                
                try:
                    probs = model.predict_proba(bulk_scaled)
                    bulk_df["AI_Confidence_%"] = np.max(probs, axis=1) * 100
                    bulk_df["AI_Confidence_%"] = bulk_df["AI_Confidence_%"].round(2)
                except AttributeError:
                    pass
                
                st.success("✅ Predictions complete!")
                st.dataframe(bulk_df.head(50), use_container_width=True) # Show first 50
                
                output_csv = bulk_df.to_csv(index=False)
                st.download_button("📥 Download Results with Predictions", data=output_csv, file_name="bulk_predictions_result.csv", mime="text/csv")
                
        except Exception as e:
            st.error(f"Error processing file. Please ensure it matches the template format. Details: {e}")


# ==========================================
# VIEW 3: DATA INSIGHTS
# ==========================================
elif view == "📈 Data Insights":
    st.markdown("### 📈 Market & Data Insights")
    st.write("Explore historical data from our startup database.")
    
    df_insights = load_insights_data()
    
    if df_insights is not None:
        st.success(f"Data loaded successfully. Total historical records: {len(df_insights):,}")
        
        ins_col1, ins_col2 = st.columns(2)
        
        with ins_col1:
            st.markdown("#### Funding by Status")
            try:
                # Group by status to show avg funding (Assuming 'status' column exists in processed data)
                # Note: Processed data usually has 'status' as a target or we can use our config
                st.info("Visualizing the relationship between funding amounts and company status.")
                if 'status' in df_insights.columns:
                    status_funding = df_insights.groupby('status')['funding_total_usd'].mean().reset_index()
                    st.bar_chart(status_funding.set_index('status'))
                elif 'label' in df_insights.columns: # fallback if column is named label
                     status_funding = df_insights.groupby('label')['funding_total_usd'].mean().reset_index()
                     st.bar_chart(status_funding.set_index('label'))
                else:
                    st.write("Status column not directly available in this sample.")
            except Exception as e:
                st.write("Could not generate chart.")

        with ins_col2:
            st.markdown("#### Company Age Distribution")
            try:
                # Basic histogram approach using bar_chart
                counts, bins = np.histogram(df_insights['company_age'].dropna(), bins=20, range=(0,20))
                hist_df = pd.DataFrame({"Age": [f"{int(b)}-{int(b+1)}" for b in bins[:-1]], "Count": counts})
                st.bar_chart(hist_df.set_index("Age"))
            except Exception:
                st.write("Could not generate chart.")
                
    else:
        st.warning("Historical data is not available or is too large to load in the viewer.")
