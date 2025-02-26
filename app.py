import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px


st.set_page_config(page_title="Crop Yield Prediction App", layout="wide", initial_sidebar_state="expanded", page_icon="🌾")


def load_data():
    df = pd.read_csv("Dataset\yield.csv")
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    
    df.drop_duplicates(inplace=True)
    
    def isStr(obj):
        try:
            float(obj)
            return False
        except:
            return True

    to_drop = df[df['average_rainfall_mm_per_year'].apply(isStr)].index

    df = df.drop(to_drop)

    df['average_rainfall_mm_per_year'] = df['average_rainfall_mm_per_year'].astype(np.float64)
    
    cols = ['Year', 'average_rainfall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
    df = df[cols]
    
    return df

def train_and_save_model():    
    df = st.session_state.data if "data" in st.session_state else load_data()
    
    col_order = ['Year', 'average_rainfall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
    df = df[col_order]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, shuffle=True)
    
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    
    ohe = OneHotEncoder(drop='first')
    scale = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ('StandardScale', scale, [0, 1, 2, 3]),
            ('OHE', ohe, [4, 5])
        ],
        remainder='passthrough'
    )
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    from sklearn.svm import SVR
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr.fit(X_train_transformed, y_train)
    
    with open("svr.pkl", "wb") as f:
        pickle.dump(svr, f)
    with open("preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
        
    return svr, preprocessor

def load_model_and_preprocessor():
    if os.path.exists("svr.pkl") and os.path.exists("preprocessor.pkl"):
        with open("svr.pkl", "rb") as f:
            model = pickle.load(f)
        with open("preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        return model, preprocessor
    else:
        st.info("Model or preprocessor not found. Training a new model. Please wait...")
        model, preprocessor = train_and_save_model()
        st.success("Model trained and saved!")
        return model, preprocessor

def generate_plots(df):
    plots = {}
    
    fig_heat = px.imshow(df.isnull(),
                         color_continuous_scale="blues",
                         title="Missing Values Heatmap")
    plots["heatmap"] = fig_heat

    sample_pesticides = df.sample(min(250, len(df)))
    fig_line_pesticides = px.bar(sample_pesticides,
                                  x="Year", 
                                  y="pesticides_tonnes",
                                  title="Pesticides Tonnes Over Years (Sample 250)")
    plots["line_pesticides"] = fig_line_pesticides

    sample_rainfall = df.sample(min(99, len(df)))
    fig_line_rainfall = px.bar(sample_rainfall,
                                x="Year", 
                                y="average_rainfall_mm_per_year",
                                title="Average Rainfall (mm per year) Over Years (Sample 99)")
    fig_line_rainfall.update_xaxes(tickangle=90)
    plots["line_rainfall"] = fig_line_rainfall

    sample_temp = df.sample(min(99, len(df)))
    fig_line_temp = px.bar(sample_temp,
                            x="Year", 
                            y="avg_temp",
                            title="Average Temperature Over Years (Sample 99)")
    fig_line_temp.update_xaxes(tickangle=90)
    plots["line_temp"] = fig_line_temp
    
    count_item = df.groupby("Item")["Item"].count().reset_index(name="count")
    fig_pie = px.pie(count_item,
                     names="Item",
                     values="count",
                     title="Item Distribution (Average Crop)")
    plots["pie_crop"] = fig_pie

    count_year = df.groupby("Year")["Year"].count().reset_index(name="count")
    fig_bar_year = px.bar(count_year,
                          x="Year", 
                          y="count",
                          title="Count by Year",
                          labels={"count": "Count", "Year": "Year"})
    fig_bar_year.update_xaxes(tickangle=90)
    plots["bar_year_count"] = fig_bar_year
    
    count_area = df['Area'].value_counts().reset_index()
    count_area.columns = ["Area", "count"]
    fig_count_area = px.bar(count_area,
                            x="count",
                            y="Area",
                            orientation="h",
                            title="Count of Areas")
    plots["count_area"] = fig_count_area

    yield_country = df.groupby("Area")["hg/ha_yield"].sum().reset_index()
    fig_yield_country = px.bar(yield_country,
                               x="hg/ha_yield",
                               y="Area",
                               orientation="h",
                               title="Yield per Country (hg/ha)")
    plots["bar_yield_country"] = fig_yield_country

    count_item2 = df['Item'].value_counts().reset_index()
    count_item2.columns = ["Item", "count"]
    fig_count_item = px.bar(count_item2,
                            x="count",
                            y="Item",
                            orientation="h",
                            title="Count of Items")
    plots["count_item"] = fig_count_item

    yield_crop = df.groupby("Item")["hg/ha_yield"].sum().reset_index()
    fig_yield_crop = px.bar(yield_crop,
                            x="hg/ha_yield",
                            y="Item",
                            orientation="h",
                            title="Yield per Crop (hg/ha)")
    plots["bar_yield_crop"] = fig_yield_crop

    return plots

def get_column_info():
    info = {
        "Area": "Geographical region or country (e.g., Albania).",
        "Item": "Type of crop or item (e.g., Maize).",
        "Year": "Year when the data was recorded (e.g., 1990).",
        "hg/ha_yield": "Crop yield measured in hectograms per hectare.",
        "average_rainfall_mm_per_year": "Average annual rainfall in millimeters.",
        "pesticides_tonnes": "Amount of pesticides used in tonnes.",
        "avg_temp": "Average temperature (°C) during the year."
    }
    return info

if "data" not in st.session_state:
    st.session_state.data = load_data()

if "model" not in st.session_state or "preprocessor" not in st.session_state:
    model, preprocessor = load_model_and_preprocessor()
    st.session_state.model = model
    st.session_state.preprocessor = preprocessor

if "plots" not in st.session_state:
    st.session_state.plots = generate_plots(st.session_state.data)

if "col_info" not in st.session_state:
    st.session_state.col_info = get_column_info()

tab_pred, tab_viz, tab_info = st.tabs(["Prediction", "Visualizations", "Info"])

with tab_pred:
    st.header("Yield Prediction")
    st.markdown("Input the details below to predict the yield (in hg/ha).")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.number_input("Year", min_value=1900, max_value=2100, value=1990, step=1)
            avg_rainfall = st.number_input("Average Rainfall (mm per year)", value=1485.0)
            pesticides = st.number_input("Pesticides (tonnes)", value=121.0)
            
        with col2:
            avg_temp = st.number_input("Average Temperature (°C)", value=16.37)
            area = st.selectbox("Area", options=sorted(st.session_state.data["Area"].unique()))
            item = st.selectbox("Item", options=sorted(st.session_state.data["Item"].unique()))
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            input_features = np.array([[year, avg_rainfall, pesticides, avg_temp, area, item]], dtype=object)
            
            transformed_features = st.session_state.preprocessor.transform(input_features)
            
            prediction = st.session_state.model.predict(transformed_features)
            
            st.success(f"Predicted Yield (hg/ha): {prediction[0]:.2f}")

with tab_viz:
    st.header("Data Visualizations")
    st.markdown("Explore various visualizations of the yield data below.")
    
    st.subheader("Missing Values Heatmap")
    st.plotly_chart(st.session_state.plots["heatmap"], use_container_width=True)
    
    st.subheader("Pesticides Tonnes Over Years")
    st.plotly_chart(st.session_state.plots["line_pesticides"], use_container_width=True)
    
    st.subheader("Average Rainfall Over Years")
    st.plotly_chart(st.session_state.plots["line_rainfall"], use_container_width=True)
    
    st.subheader("Average Temperature Over Years")
    st.plotly_chart(st.session_state.plots["line_temp"], use_container_width=True)
    
    st.subheader("Item Distribution (Pie Chart)")
    st.plotly_chart(st.session_state.plots["pie_crop"], use_container_width=True)
    
    st.subheader("Count by Year")
    st.plotly_chart(st.session_state.plots["bar_year_count"], use_container_width=True)
    
    st.subheader("Count of Areas")
    st.plotly_chart(st.session_state.plots["count_area"], use_container_width=True)
    
    st.subheader("Yield per Country")
    st.plotly_chart(st.session_state.plots["bar_yield_country"], use_container_width=True)
    
    st.subheader("Count of Items")
    st.plotly_chart(st.session_state.plots["count_item"], use_container_width=True)
    
    st.subheader("Yield per Crop")
    st.plotly_chart(st.session_state.plots["bar_yield_crop"], use_container_width=True)

with tab_info:
    st.header("Data Column Information")
    st.markdown("Below is a description of each column in the dataset:")
    
    col_info = st.session_state.col_info
    info_df = pd.DataFrame({
        "Column": list(col_info.keys()),
        "Description": list(col_info.values())
    })
    
    st.table(info_df)
