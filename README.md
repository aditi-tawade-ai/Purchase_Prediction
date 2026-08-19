# Purchase_Prediction
# KNN Social Network Purchase Prediction

## 📌 Project Overview

This project uses the **K-Nearest Neighbors (KNN)** machine learning algorithm to predict whether a person is likely to purchase a product based on their:

* Gender
* Age
* Estimated Salary

The trained model is deployed using **Streamlit**, allowing users to enter their details and receive a purchase prediction.

---

## 🎯 Objective

The objective of this project is to build a classification model that predicts whether a person will purchase a product based on demographic and salary information.

The target variable is:

* `0` → Not Purchased
* `1` → Purchased

---

## 📊 Dataset

The dataset contains **400 records** and 5 columns:

| Column          | Description                  |
| --------------- | ---------------------------- |
| User ID         | Unique identification number |
| Gender          | Male/Female                  |
| Age             | Age of the person            |
| EstimatedSalary | Estimated salary             |
| Purchased       | Target variable              |

There are no missing values in the dataset.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit

---

## 🔄 Machine Learning Workflow

The project follows these steps:

```text
Data Collection
      ↓
Data Analysis
      ↓
Data Preprocessing
      ↓
Gender Encoding
      ↓
Feature Selection
      ↓
Standardization
      ↓
Train-Test Split
      ↓
KNN Model Training
      ↓
K Value Selection
      ↓
Model Evaluation
      ↓
Model Saving
      ↓
Streamlit Deployment
```

---

## 🧹 Data Preprocessing

### 1. Gender Encoding

The categorical `Gender` column was converted into numerical values:

```python
Male → 0
Female → 1
```

This encoding was performed before training the model.

### 2. Feature Selection

The following three features were used for prediction:

```text
Gender
Age
EstimatedSalary
```

`User ID` was not used as a prediction feature.

### 3. Feature Scaling

Since KNN is a distance-based algorithm, the features were standardized using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)
```

---

## 🤖 KNN Model

K-Nearest Neighbors is a supervised machine learning algorithm used for classification and regression.

For classification, KNN finds the nearest data points to a new data point and assigns the class based on the majority of the nearest neighbors.

---

## 🔍 Selecting the Value of K

Different values of K were tested and compared using evaluation metrics including:

* Precision
* Recall
* F1 Score
* Error Rate

The error-rate analysis was performed for K values from 1 to 10.

After comparing the graphs, **K = 7** was selected as the final value.

```python
from sklearn.neighbors import KNeighborsClassifier

fm = KNeighborsClassifier(n_neighbors=7)
fm.fit(X_train, y_train)
```

---

## 📈 Model Performance

The final KNN model achieved:

**Test Accuracy: 92.5%**

The final confusion matrix was:

```text
[[54, 3],
 [ 3, 20]]
```

---

## 💾 Model Saving

The trained KNN model and the StandardScaler are saved using Joblib.

```python
import joblib

joblib.dump(fm, "knn.pkl")
joblib.dump(sc, "scaler.pkl")
```

The saved files are then used during deployment.

---

## 🌐 Deployment

The model is deployed using **Streamlit**.

The application allows the user to enter:

```text
Gender
Age
Estimated Salary
```

The input is encoded and scaled using the same preprocessing applied during training. The KNN model then predicts whether the person is likely to purchase the product.

---

## 📁 Project Structure

```text
KNN_Purchase_Prediction/
│
├── app.py
├── knn.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository or download the project files.

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Run the following command in the VS Code terminal:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

## 🖥️ Application Input

The user provides:

* **Gender**
* **Age**
* **Estimated Salary**

The application then displays one of two results:

```text
The person is likely to PURCHASE the product.
```

or

```text
The person is NOT likely to purchase the product.
```

---

## 📌 Key Learning Outcomes

Through this project, I learned:

* Data preprocessing
* Handling categorical variables
* Feature scaling
* Train-test splitting
* KNN classification
* Selecting the optimal K value
* Confusion matrix
* Precision, Recall and F1 Score
* Error-rate analysis
* Model serialization using Joblib
* Machine learning model deployment using Streamlit

---

## 👩‍💻 Author

**Aditi Tawade**

B.Tech — Electronics & Telecommunication Engineering

## 👨‍🏫 Guide

**Yameen Hakim Sir**


---


## ⭐ Conclusion

This project demonstrates the complete machine learning workflow, starting from dataset preprocessing and feature scaling to KNN model training, evaluation, model saving, and deployment using Streamlit.
