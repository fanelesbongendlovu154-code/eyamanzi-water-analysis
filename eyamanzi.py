import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Prophet for time-series forecasting
try:
  from prophet import Prophet

  PROPHET_AVAILABLE = True
except ImportError:
  PROPHET_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="eYamanzi - Water Pollution Analysis", page_icon="💧", layout="wide"
)

st.title("💧 eYamanzi: Water Quality Monitoring & Analytics Dashboard")
st.markdown(
    """
    ### Data Science, Machine Learning, and Predictive Water Quality Analysis

    Welcome to **eYamanzi**. This interactive dashboard analyzes water-quality monitoring data using:
    - 📊 **Exploratory Data Analysis** with dynamic provincial and city filters[cite: 5]
    - 🌲 **Random Forest Classification** for predicting unsafe low pH conditions[cite: 5]
    - 🔵 **K-Means Clustering** for grouping water sample profiles[cite: 5]
    - 📈 **Prophet Time-Series Forecasting** for future pH trend estimation[cite: 5]
    """
)


# ============================================================
# LOAD DATA
# ============================================================


@st.cache_data
def load_data(uploaded_file=None):
  if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
  else:
    data = pd.read_csv("china_water_pollution_data.csv")
  return data


# Sidebar Navigation & Settings
st.sidebar.header("⚙️ eYamanzi Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload custom water pollution CSV", type=["csv"]
)

try:
  df = load_data(uploaded_file)
except FileNotFoundError:
  st.error(
      "CSV file not found. Please ensure `china_water_pollution_data.csv` is"
      " placed in the directory or uploaded via the sidebar[cite: 5]."
  )
  st.stop()

# Convert Date column if present
if "Date" in df.columns:
  df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

page = st.sidebar.radio(
    "Navigate Modules",
    [
        "🏠 Dashboard",
        "📊 Data Analysis",
        "🌲 Random Forest",
        "🔵 K-Means Clustering",
        "📈 Prophet Forecast",
    ],
)


# ============================================================
# DASHBOARD MODULE
# ============================================================

if page == "🏠 Dashboard":
  st.header("🏠 eYamanzi Overview Dashboard")

  col1, col2, col3, col4 = st.columns(4)

  with col1:
    st.metric("Total Records", f"{len(df):,}")

  with col2:
    st.metric("Total Columns", f"{len(df.columns)}")

  with col3:
    st.metric("Average pH", f"{df['pH'].mean():.2f}")

  with col4:
    st.metric("Average WQI", f"{df['Water_Quality_Index'].mean():.2f}")

  st.divider()

  st.subheader("📋 Dataset Preview")
  st.dataframe(df.head(20), use_container_width=True)

  st.subheader("📌 Dataset Information")
  col1, col2 = st.columns(2)

  with col1:
    st.write("**Number of records:**", len(df))
    st.write("**Number of columns:**", len(df.columns))

  with col2:
    st.write("**Missing values:**", int(df.isnull().sum().sum()))
    st.write("**Duplicate rows:**", int(df.duplicated().sum()))

  st.info(
      "The dataset contains key water-quality parameters including pH,"
      " dissolved oxygen, turbidity, nutrients, chemical oxygen demand (COD),"
      " biochemical oxygen demand (BOD), heavy metals, coliform count, and the"
      " Water Quality Index (WQI)[cite: 5]."
  )


# ============================================================
# DATA ANALYSIS MODULE
# ============================================================

elif page == "📊 Data Analysis":
  st.header("📊 Exploratory Data Analysis")

  st.sidebar.subheader("Data Filters")
  filtered_df = df.copy()

  if "Province" in df.columns:
    provinces = st.sidebar.multiselect(
        "Select Province", sorted(df["Province"].dropna().unique()), default=[]
    )
    if provinces:
      filtered_df = filtered_df[filtered_df["Province"].isin(provinces)]

  if "City" in df.columns:
    cities = st.sidebar.multiselect(
        "Select City", sorted(df["City"].dropna().unique()), default=[]
    )
    if cities:
      filtered_df = filtered_df[filtered_df["City"].isin(cities)]

  st.write(f"Showing **{len(filtered_df):,}** records after filtering.")

  # pH Distribution
  st.subheader("📈 pH Distribution")
  fig, ax = plt.subplots(figsize=(10, 5))
  ax.hist(filtered_df["pH"].dropna(), bins=30, color="skyblue", edgecolor="black")
  ax.axvline(
      6.5, color="red", linestyle="--", linewidth=2, label="Unsafe Threshold (pH = 6.5)"
  )
  ax.set_xlabel("pH")
  ax.set_ylabel("Number of Records")
  ax.set_title("Distribution of Water pH")
  ax.legend()
  st.pyplot(fig)

  # Water Quality Index Distribution
  st.subheader("💧 Water Quality Index Distribution")
  fig, ax = plt.subplots(figsize=(10, 5))
  ax.hist(
      filtered_df["Water_Quality_Index"].dropna(),
      bins=30,
      color="teal",
      edgecolor="black",
  )
  ax.set_xlabel("Water Quality Index")
  ax.set_ylabel("Number of Records")
  ax.set_title("Water Quality Index Distribution")
  st.pyplot(fig)

  # Pollution Level Counts
  if "Pollution_Level" in filtered_df.columns:
    st.subheader("🚨 Pollution Levels")
    pollution_counts = filtered_df["Pollution_Level"].value_counts()
    st.bar_chart(pollution_counts)

  # Correlation Matrix
  st.subheader("🔗 Correlation Matrix")
  numeric_df = filtered_df.select_dtypes(include=np.number)
  if not numeric_df.empty:
    correlation = numeric_df.corr()
    st.dataframe(correlation.round(2), use_container_width=True)


# ============================================================
# RANDOM FOREST MODULE
# ============================================================

elif page == "🌲 Random Forest":
  st.header("🌲 Random Forest – Low pH Prediction Model")
  st.markdown(
      """
        The Random Forest classifier predicts whether a water sample falls into the **Unsafe Low pH** class (\(pH < 6.5\))[cite: 5].
        """
  )

  model_df = df.copy()
  model_df["Unsafe_Low_pH"] = (model_df["pH"] < 6.5).astype(int)

  features = [
      "Water_Temperature_C",
      "Dissolved_Oxygen_mg_L",
      "Conductivity_uS_cm",
      "Turbidity_NTU",
      "Nitrate_mg_L",
      "Nitrite_mg_L",
      "Ammonia_N_mg_L",
      "Total_Phosphorus_mg_L",
      "Total_Nitrogen_mg_L",
      "COD_mg_L",
      "BOD_mg_L",
      "Heavy_Metals_Pb_ug_L",
      "Heavy_Metals_Cd_ug_L",
      "Heavy_Metals_Hg_ug_L",
      "Coliform_Count_CFU_100mL",
      "Water_Quality_Index",
  ]
  features = [f for f in features if f in model_df.columns]

  X = model_df[features].copy().fillna(model_df[features].median())
  y = model_df["Unsafe_Low_pH"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.20, random_state=42, stratify=y
  )

  rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
  rf_model.fit(X_train, y_train)
  predictions = rf_model.predict(X_test)
  accuracy = accuracy_score(y_test, predictions)

  col1, col2, col3 = st.columns(3)
  col1.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
  col2.metric("Training Records", f"{len(X_train):,}")
  col3.metric("Testing Records", f"{len(X_test):,}")

  st.divider()

  st.subheader("📋 Classification Report")
  report = classification_report(
      y_test, predictions, output_dict=True, zero_division=0
  )
  st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

  st.subheader("⭐ Feature Importance")
  importance_df = pd.DataFrame(
      {"Feature": features, "Importance": rf_model.feature_importances_}
  ).sort_values("Importance", ascending=False)
  st.bar_chart(importance_df.set_index("Feature"))

  # Interactive Prediction Interface
  st.divider()
  st.subheader("🔮 Test a New Water Sample")

  input_values = {}
  cols = st.columns(2)
  for index, feature in enumerate(features):
    with cols[index % 2]:
      default_value = float(X[feature].median())
      input_values[feature] = st.number_input(feature, value=default_value)

  if st.button("🔍 Predict Water Quality", type="primary"):
    input_df = pd.DataFrame([input_values])
    prediction = rf_model.predict(input_df)[0]
    probability = rf_model.predict_proba(input_df)[0]

    if prediction == 1:
      st.error(f"⚠️ Prediction: UNSAFE LOW pH (Probability: {probability[1] * 100:.2f}%)")
    else:
      st.success(f"✅ Prediction: SAFE pH (Probability: {probability[0] * 100:.2f}%)")


# ============================================================
# K-MEANS CLUSTERING MODULE
# ============================================================

elif page == "🔵 K-Means Clustering":
  st.header("🔵 K-Means Water Quality Clustering")
  st.markdown(
      "Unsupervised K-Means clustering groups water samples with similar characteristics into **5 clusters**[cite: 5]."
  )

  cluster_features = [
      "pH",
      "Dissolved_Oxygen_mg_L",
      "Conductivity_uS_cm",
      "Turbidity_NTU",
      "COD_mg_L",
      "BOD_mg_L",
      "Water_Quality_Index",
  ]
  cluster_features = [f for f in cluster_features if f in df.columns]

  X_cluster = df[cluster_features].copy().fillna(df[cluster_features].median())
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X_cluster)

  kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
  clusters = kmeans.fit_predict(X_scaled)

  cluster_df = df.copy()
  cluster_df["Cluster"] = clusters

  cluster_pH = cluster_df.groupby("Cluster")["pH"].mean().sort_values()
  status_labels = {
      cluster_pH.index[0]: "Very Unsafe",
      cluster_pH.index[1]: "Unsafe",
      cluster_pH.index[2]: "Moderate",
      cluster_pH.index[3]: "Safe",
      cluster_pH.index[4]: "Very Safe",
  }
  cluster_df["KMeans_Status"] = cluster_df["Cluster"].map(status_labels)

  st.subheader("📊 Cluster Summary")
  summary = (
      cluster_df.groupby(["Cluster", "KMeans_Status"])
      .agg(
          Samples=("pH", "count"),
          Average_pH=("pH", "mean"),
          Average_WQI=("Water_Quality_Index", "mean"),
      )
      .reset_index()
  )
  st.dataframe(summary.round(2), use_container_width=True)

  st.subheader("📈 Samples per Cluster Category")
  st.bar_chart(cluster_df["KMeans_Status"].value_counts())


# ============================================================
# PROPHET FORECAST MODULE
# ============================================================

elif page == "📈 Prophet Forecast":
  st.header("📈 pH Time-Series Forecast")

  if not PROPHET_AVAILABLE:
    st.error("Prophet package is not installed. Please run `pip install prophet`.")
    st.stop()

  ts_data = df[["Date", "pH"]].dropna()
  ts_data = ts_data.groupby("Date")["pH"].mean().reset_index()
  ts_data = ts_data.rename(columns={"Date": "ds", "pH": "y"}).sort_values("ds")

  st.subheader("📊 Historical pH Trend")
  fig, ax = plt.subplots(figsize=(12, 5))
  ax.plot(ts_data["ds"], ts_data["y"], color="royalblue")
  ax.axvline(
      6.5, color="red", linestyle="--", label="pH Threshold = 6.5"
  )
  ax.set_xlabel("Date")
  ax.set_ylabel("pH")
  ax.legend()
  st.pyplot(fig)

  forecast_days = st.slider("Number of days to forecast", 7, 90, 30)

  with st.spinner("Training Prophet forecasting model..."):
    prophet_model = Prophet()
    prophet_model.fit(ts_data)
    future = prophet_model.make_future_dataframe(
        periods=forecast_days, freq="D"
    )
    forecast = prophet_model.predict(future)

  st.subheader(f"🔮 {forecast_days}-Day pH Forecast")
  fig, ax = plt.subplots(figsize=(12, 6))
  historical = forecast[forecast["ds"] <= ts_data["ds"].max()]
  future_forecast = forecast[forecast["ds"] > ts_data["ds"].max()]

  ax.plot(
      historical["ds"], historical["yhat"], label="Historical/Fitted", color="blue"
  )
  ax.plot(
      future_forecast["ds"],
      future_forecast["yhat"],
      linestyle="--",
      label="Forecast",
      color="darkorange",
  )
  ax.fill_between(
      future_forecast["ds"],
      future_forecast["yhat_lower"],
      future_forecast["yhat_upper"],
      color="orange",
      alpha=0.2,
      label="Uncertainty Interval",
  )
  ax.axhline(6.5, linestyle=":", color="red", label="pH = 6.5")
  ax.set_xlabel("Date")
  ax.set_ylabel("pH")
  ax.legend()
  st.pyplot(fig)

  avg_forecast = future_forecast["yhat"].mean()
  st.metric("Average Forecasted pH", f"{avg_forecast:.2f}")

  if avg_forecast < 6.5:
    st.warning(
        "⚠️ Average forecasted pH is below 6.5, indicating a potential low-pH"
        " risk period requiring attention."
    )
  else:
    st.success(
        "✅ Average forecasted pH is within safe parameters (above 6.5)."
    )

# ============================================================
# FOOTER
# ============================================================
st.sidebar.divider()
st.sidebar.info("💧 eYamanzi Analytics Suite\n\nMachine Learning & Forecasting")
st.caption("Water Pollution Analysis & Prediction Dashboard")