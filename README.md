# 🚀 **Applied Data Science Capstone - SpaceX Landing Prediction**

## 📌 **Project Overview**  
This project is part of the **IBM Data Science Professional Certificate** specialization. It focuses on using **data science and machine learning** to predict **the success of SpaceX’s Falcon 9 first-stage landings**. The goal is to analyze various launch factors and develop predictive models to determine if SpaceX can **reuse the first stage**, which significantly reduces launch costs.

In the extended version of the project, we converted the ML solution into a production-grade pipeline with automated training, experiment tracking, model registry, and cloud deployment. While we initially deployed the model to **GCP Vertex AI**, we later moved to **Cloud Run** for cost-efficiency. 

🛰️ **Live Inference Endpoint:** [Cloud Run URL](https://spacex-api-7pmmzeuvqa-uc.a.run.app)

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

### 🔹 **6. MLOps and Productionization**  
- ML model training pipeline was automated using **Airflow DAG**.  
- Model experiments were tracked with **MLflow**, with metrics, params, and model versioning.  
- CI/CD using **GitHub Actions** retrains and logs models automatically on commit.  
- The top-performing model was containerized using **Docker** and served with **FastAPI**.  
- Deployed to **GCP Cloud Run** via container image in **Artifact Registry**.  
- **Note:** Deployment to **Vertex AI** was attempted and functional, but later discontinued due to billing constraints.  

---

## 🚀 **Technologies Used**  
- **Python**  
- **Pandas, NumPy, Matplotlib, Seaborn** (Data Processing & Visualization)  
- **Scikit-Learn** (Machine Learning Models)  
- **Folium & Plotly Dash** (Interactive Visualizations)  
- **BeautifulSoup & Requests** (Web Scraping)  
- **SQL** (Data Querying & Analysis)  
- **Dash, FastAPI, Docker** (App Interface and Deployment)  
- **MLflow** (Model Tracking & Registry)  
- **Apache Airflow** (Pipeline Automation)  
- **GitHub Actions** (CI/CD)  
- **GCP Cloud Run & Artifact Registry** (Model Hosting)
- **Vertex AI** (Temporarily used for model endpoint deployment)

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
| `spacex_dash_app.py` | Dash-based web application for data visualization |
| `train.py` | ML pipeline script with MLflow integration |
| `main.py` | FastAPI serving script for model inference |
| `Dockerfile` | Docker configuration for cloud deployment |
| `.github/workflows/train.yml` | GitHub Actions workflow for CI/CD |
| `dags/spacex_dag.py` | Airflow DAG for scheduled retraining |

---

## 🔧 **Installation & Setup**  
### **1️⃣ Prerequisites**  
Install the required libraries:  
```bash
pip install -r requirements.txt
```

### **2️⃣ Running the Project Locally**  
```bash
git clone https://github.com/Dx2905/spacex-ml-pipeline.git
cd spacex-ml-pipeline
python train.py
uvicorn main:app --reload
```
Access the API at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

### **3️⃣ Deploying to GCP (Cloud Run)**
```bash
docker build -t spacex-api .
docker tag spacex-api us-central1-docker.pkg.dev/spacex-ml-project/spacex-model-repo/spacex-api
docker push us-central1-docker.pkg.dev/spacex-ml-project/spacex-model-repo/spacex-api
gcloud run deploy spacex-api \
  --image=us-central1-docker.pkg.dev/spacex-ml-project/spacex-model-repo/spacex-api \
  --platform=managed \
  --region=us-central1 \
  --allow-unauthenticated
```
✅ **Live API URL:** [https://spacex-api-7pmmzeuvqa-uc.a.run.app](https://spacex-api-7pmmzeuvqa-uc.a.run.app)

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
- **Automate data ingestion** from APIs into Airflow workflows.  

---

## 📜 **License**  
This project is licensed under the **MIT License**. See the [`LICENSE`](https://github.com/Dx2905/spacex-ml-pipeline/blob/main/LICENSE) file for more details.

---

## 🧠 **Credits & Note**  
Originally built as part of the **IBM Data Science Professional Certificate**, this project was extended for real-world deployment and ML engineering demonstration.  

---

📫 Questions? Feedback? Reach out via [LinkedIn](https://www.linkedin.com/in/fnu-gaurav-653355252/) or [GitHub](https://github.com/Dx2905)!

