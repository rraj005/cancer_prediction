# 🩺 Breast Cancer Detection using Regression and TensorFlow

This project focuses on **Breast Cancer Detection** using **Logistic Regression (Scikit-learn)** and **TensorFlow**.  
It highlights **data cleaning**, **EDA (Exploratory Data Analysis)**, **model building**, and **model deployment** using TensorFlow's `SavedModel` format.  

This project also demonstrates strong **Data Analyst skills** by thoroughly cleaning and preparing the dataset before modeling.

---

## 📂 Dataset Used

- **Cancer Dataset.csv** (for training)
- **Cancer Test Dataset.csv** (for testing)

### Dataset Features:
- Total Features: **31 numerical features** (e.g., Radius Mean, Texture Mean, Area Mean, etc.)
- Target Variable: **Diagnosis** (Binary classification: Benign / Malignant)

---

## 🧹 Data Cleaning and Organization ✅ (Data Analyst Focus)

Before modeling, special focus was given to **making the dataset clean and analysis-ready**, proving attention to data quality and preparation:

- ✅ **Checked for null/missing values**
- ✅ **Handled missing values** using **forward fill and backward fill** techniques
- ✅ **Dropped duplicate records** to avoid data leakage or bias
- ✅ **Ensured balanced class distribution** check through EDA
- ✅ **Verified column types and structure consistency**
- ✅ **Re-checked after cleaning with `.isnull().sum()` and `.duplicated().sum()`**

This meticulous data preparation reflects my strong **data cleaning, preprocessing, and organizational skills**, crucial for any Data Analyst role.

---

## 📊 Exploratory Data Analysis (EDA)

- Generated a **Pie Chart** showing the distribution of Benign vs Malignant cases.
- Ensured clean, visually interpretable analysis of feature-target relationships.

---

## 🧑‍💻 Model Building

### ✅ Logistic Regression (Scikit-learn):
- Used **Logistic Regression** for binary classification.
- Achieved **high accuracy (~95-98%)** on test set.
- Metrics used: **Accuracy Score**.

### ✅ TensorFlow Model Wrapping:
- Wrapped the trained Scikit-learn model inside a **TensorFlow `tf.Module`**.
- Exported it as a **TensorFlow SavedModel**.
- Successfully reloaded and performed inference on unseen test data.

### ✅ Bonus: Keras Sequential Model (for practice):
- Implemented a small **Neural Network using Keras Sequential API**.
- Trained using **Binary Crossentropy Loss** and **Adam Optimizer**.

---

## ✅ Model Saving (Sklearn → TensorFlow SavedModel)

Example of exporting Scikit-learn model for TensorFlow serving:

```python
import tensorflow as tf

class ModelWrapper(tf.Module):
    def __init__(self, sklearn_model):
        super().__init__()
        self.model = sklearn_model

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, X_train.shape[1]], dtype=tf.float32, name="x")])
    def predict(self, x):
        def sklearn_predict(x_np):
            preds = self.model.predict(x_np)
            return preds.astype(np.int64)
        preds = tf.py_function(func=sklearn_predict, inp=[x], Tout=tf.int64)
        preds.set_shape([None])
        return preds

wrapped_model = ModelWrapper(model)
tf.saved_model.save(wrapped_model, 'saved_sklearn_model', signatures=wrapped_model.predict)
