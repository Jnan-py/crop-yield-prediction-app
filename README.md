# Crop Yield Prediction App

This Streamlit application predicts crop yield (measured in hectograms per hectare, hg/ha) using a machine learning model and provides various data visualizations to explore the underlying dataset. The app is designed for users interested in agricultural data analysis, enabling them to both forecast crop yield based on input features and gain insights through interactive charts.

## Features

- **Yield Prediction:**  
  Input key features such as year, average rainfall, pesticide usage, average temperature, area, and crop item to predict the crop yield using a trained Support Vector Regression (SVR) model.
- **Data Visualizations:**  
  Explore the dataset through multiple visualizations including:
  - Missing values heatmap
  - Bar charts for pesticides, rainfall, and temperature trends over the years
  - Pie chart for crop item distribution
  - Count plots for year, area, and crop items
  - Bar plots for yield per country and yield per crop
- **Data Information:**  
  View detailed descriptions for each column in the dataset.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Jnan-py/crop-yield-prediction-app.git
   cd crop-yield-prediction-app
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Dataset:**
   - Ensure that the dataset file (`yield.csv`) is placed inside the `Dataset` folder.
   - The app expects the file to be available at `Dataset/yield.csv`.

## Usage

1. **Run the Application:**

   ```bash
   streamlit run app.py
   ```

2. **Navigate the App:**

   - **Prediction Tab:** Input crop and weather details to get a yield prediction.
   - **Visualizations Tab:** Explore various interactive plots to understand the data.
   - **Info Tab:** Read about the dataset columns and their descriptions.

3. **Model Training and Saving:**
   - On first run, if no pre-trained model is found (`svr.pkl` and `preprocessor.pkl`), the app will train a new SVR model using the provided dataset and save it locally.

## Project Structure

```
crop-yield-prediction-app/
│
├── app.py                  # Main Streamlit application
├── Dataset/
│   └── yield.csv           # Dataset file (ensure it exists)
├── svr.pkl                 # Saved SVR model (generated after training)
├── preprocessor.pkl        # Saved preprocessor (generated after training)
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## Technologies Used

- **Python** for programming logic
- **Streamlit** for building the interactive web interface
- **Pandas & NumPy** for data manipulation
- **Scikit-Learn** for machine learning (model training, preprocessing, and evaluation)
- **Plotly Express** for interactive visualizations
- **Pickle** for model serialization
