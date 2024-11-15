# Algorithms_and_AI_Project
## Group Members
1. **D Shanmukha** (22BDS019)  
2. **G Leeladitya** (22BDS024)  
3. **Rajdeep Manik** (22BDS048)  
4. **Vansh Lal Tolani** (22BDS061)

# Integrating Graph Analytics with AI Algorithms for Enhanced Breast Cancer Classification and Interpretability 

## Project Overview
This project explores the application of graph-based analysis and machine learning algorithms on the breast cancer dataset. We incorporated **PageRank**, **Triangle Counting**, and **Connected Components** from PySpark's GraphX and used these graph-based features in combination with the original dataset for advanced AI models, including **ANN**, **DNN**, and **GNN**. Additionally, model interpretability was achieved using **SHAP** and **LIME** to provide insights into the decisions made by the DNN model.

---

## Dataset Description
- **Original Dataset**:  
  - Source: [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).  
  - The dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast masses.  
  - Target: Classification of tumors as malignant or benign.  

---

## Methodology

### Step 1: AI Models
- **Traditional Machine Learning Models**:  
  - Algorithms: Random Forest, Decision Tree, Support Vector Machine (SVM), K-Nearest Neighbors (KNN).  
  - Evaluated on metrics: Accuracy, Precision, Recall, and F1-Score.  

### Step 2: Graph Analysis
- The dataset was transformed into a graph-based structure where features were treated as nodes, and relationships were derived based on domain-specific knowledge.
- **Tools Used**:  
  - **PySpark GraphX** for computing graph-based metrics.  
- **Features Extracted**:  
  - **PageRank**: Centrality of nodes.  
  - **Triangle Counting**: Clustering coefficients.  
  - **Connected Components**: Groups of connected nodes.  

### Step 3: Advanced AI Models with Graph-Enhanced Data
- After extracting graph-based features, these were merged with the original dataset. The following advanced models were applied to the enhanced dataset:  
  - **ANN** (Artificial Neural Networks).  
  - **DNN** (Deep Neural Networks).  
  - **GNN** (Graph Neural Networks).  

### Step 4: Explainable AI
- Integrated **SHAP** and **LIME** with the DNN model to analyze and interpret its predictions.  
- Visualized feature importance and local explanations for model outputs.  

---

## Results and Analysis

### Traditional Machine Learning Models
- Traditional ML algorithms (e.g., Random Forest, Decision Tree, SVM, KNN) performed well, with SVM achieving the highest accuracy.

| **Algorithms**                     | **Accuracy** | **Precision**          | **Recall** | **F1-Score** |
|------------------------------------|--------------|------------------------|------------|--------------|
| **Traditional AI Algorithms**      |              |                        |            |              |
| Random Forest                      | 0.96         | 0.97 (Class 0)         | 0.97       | 0.97         |
|                                    |              | 0.96 (Class 1)         |            |              |
| Decision Tree                      | 0.91         | 0.94 (Class 0)         | 0.91       | 0.92         |
|                                    |              | 0.88 (Class 1)         |            |              |
| SVM                                | 0.98         | 0.97 (Class 0)         | 1.00       | 0.99         |
|                                    |              | 1.00 (Class 1)         |            |              |
| KNN                                | 0.94         | 0.92 (Class 0)         | 1.00       | 0.96         |
|                                    |              | 1.00 (Class 1)         |            |              |

### Deep Learning Models with Graph Features
- Incorporating graph-based features into deep learning models showed improved performance, particularly in **ANN** and **DNN**. However, the **GNN** model did not perform as well compared to other models, as shown in Table 2.

| **Deep Learning Models with Graph Features** | **Accuracy** | **Precision**          | **Recall** | **F1-Score** |
|---------------------------------------------|--------------|------------------------|------------|--------------|
| ANN                                         | 0.97         | 0.98 (Class 0)         | 0.97       | 0.98         |
|                                              |              | 0.95 (Class 1)         |            |              |
| GNN                                         | 0.89         | 0.88 (Class 0)         | 0.95       | 0.92         |
|                                              |              | 0.91 (Class 1)         |            |              |
| DNN                                         | 0.96         | 0.96 (Class 0)         | 0.96       | 0.96         |
|                                              |              | 0.96 (Class 1)         |            |              |

- **SHAP/LIME** analysis offered valuable insights into the decision-making process of the DNN model. This helped in understanding which features influenced the predictions the most, providing a deeper interpretability layer for the model.

---

## Technologies Used
- **Programming Language**: Python  
- **Libraries and Tools**:  
  - PySpark (GraphX)  
  - TensorFlow/Keras  
  - SHAP and LIME  
  - Scikit-learn  
  - Pandas  
  - NumPy  
  - Matplotlib  
  - Apache Spark  

---

