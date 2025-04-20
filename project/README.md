
# 🚀 **Applied Data Science Capstone - SpaceX Landing Prediction**  

## 📌 **Project Overview**  
This project is part of the **IBM Data Science Professional Certificate** specialization. It focuses on using **data science and machine learning** to predict **the success of SpaceX’s Falcon 9 first-stage landings**. The goal is to analyze various launch factors and develop predictive models to determine if SpaceX can **reuse the first stage**, which significantly reduces launch costs.

---

## 📊 **Key Questions Addressed**  
1. How do variables such as **payload mass, launch site, number of flights, and orbits** affect the success of the first-stage landing?  
2. Has the **rate of successful landings increased over the years**?  
3. What is the **best machine learning algorithm** for predicting success in this scenario?  

---

## 🏛 **Methodology**  
### 🔹 **1. Data Collection**  
- **SpaceX REST API** was used to collect launch data.  
- **Web scraping** from Wikipedia was implemented to gather additional details.  

### 🔹 **2. Data Wrangling & Preprocessing**  
- Filtering relevant columns.  
- Handling missing values.  
- Applying **One-Hot Encoding** for categorical variables.  

### 🔹 **3. Exploratory Data Analysis (EDA)**  
- Data visualization using **Matplotlib, Seaborn, and Plotly Dash**.  
- **SQL queries** were used to analyze structured launch data.  

### 🔹 **4. Interactive Visual Analytics**  
- **Folium** was used for geospatial analysis of launch sites.  
- **Dash application** was built to create interactive visualizations.  

### 🔹 **5. Machine Learning Prediction**  
- **Supervised classification models** were trained to predict **successful landings**.  
- **Algorithms used**:  
  - Logistic Regression  
  - Decision Tree  
  - Support Vector Machine (SVM)  
  - K-Nearest Neighbors (KNN)  

---

## 🚀 **Technologies Used**  
- **Python**
- **Pandas, NumPy, Matplotlib, Seaborn** (Data Processing & Visualization)  
- **Scikit-Learn** (Machine Learning Models)  
- **Folium & Plotly Dash** (Interactive Visualizations)  
- **Dash Framework** (Web-based interactive dashboard)  
- **BeautifulSoup & Requests** (Web Scraping)  
- **SQL** (Data Querying & Analysis)  

---

## 📂 **Project Files & Structure**  
| File Name | Description |
|-----------|------------|
| `Data Collection API.ipynb` | Extracting SpaceX launch data via API |
| `Data Collection with Web Scraping.ipynb` | Extracting data from Wikipedia |
| `Data Wrangling.ipynb` | Cleaning and preparing data for modeling |
| `EDA with Data Visualization.ipynb` | Exploratory data analysis using visualization |
| `EDA with SQL.ipynb` | SQL-based analysis of SpaceX data |
| `Interactive Visual Analytics with Folium.ipynb` | Geospatial visualization of launch sites |
| `Machine Learning Prediction.ipynb` | Building machine learning models |
| `spacex_dash_app.py` | **Dash-based web application for data visualization** |

---

## 🔧 **Installation & Setup**  
### **1️⃣ Prerequisites**  
Ensure you have installed the required libraries:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn folium dash plotly beautifulsoup4 requests
```

### **2️⃣ Running the Project**  
1. Clone the repository:
   ```bash
   git clone https://github.com/Dx2905/Applied-Data-Science-Capstone.git
   cd Applied-Data-Science-Capstone
   ```
2. Run Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Execute each notebook in order.  
4. To run the **Dash web application**, execute:
   ```bash
   python spacex_dash_app.py
   ```

---

## 📈 **Key Findings**  
- **Payload mass** has a significant impact on **successful landings**.  
- **Different launch sites** have varying success rates.  
- **Machine learning models were able to predict** the likelihood of a successful landing.  

| Model | Accuracy |
|--------|---------|
| Logistic Regression | 84.2% |
| Decision Tree | 78.6% |
| SVM | 81.3% |
| KNN | 79.5% |

**Logistic Regression** provided the best performance for binary classification.

---

## 🔮 **Future Improvements**  
- **Experiment with Deep Learning models (e.g., Neural Networks)**.  
- **Improve feature engineering** by including weather conditions.  
- **Enhance the Dash application** with real-time data updates.  

---

## 📜 **License**  
This project is licensed under the **MIT License**. See the [`LICENSE`](https://github.com/Dx2905/Applied-Data-Science-Capstone/blob/main/LICENSE) file for more details.

---

