---
title: Online Shopping Intention Analysis
emoji: 🛒
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# Online Shopping Intention Analysis 🛒

## 📌 Project Overview
This project focuses on **Unsupervised Learning** to analyze and group online shopping sessions. By clustering visitors based on their real-time behavior (time spent on various page types), we can identify different user "intents"—distinguishing between casual browsers and high-intent shoppers.

## 📊 Metadata
- **Dataset:** Online Shoppers Purchasing Intention Dataset
- **Size:** 12,330 Sessions
- **Field:** E-commerce / User Behavior Analytics
- **Language:** Python
- **Libraries:** Pandas, Scikit-Learn, Matplotlib, Seaborn

## ⚙️ Core Algorithm: K-Means Clustering
The project utilizes the **K-Means** algorithm to partition the sessions into distinct groups. 
- **Process:** It calculates the distance between data points and centroids to minimize within-cluster variance.
- **Features Used:** `Administrative_Duration`, `Informational_Duration`, and `ProductRelated_Duration`.

## 📏 Key Metric: Silhouette Score
To ensure the clusters are mathematically sound, we use the **Silhouette Score**.
- **Purpose:** It measures how similar an object is to its own cluster compared to other clusters.
- **Optimization:** We iterate through different values of $k$ to find where the "numbers stop changing" and the score is highest, ensuring clear separation between segments.

## 💡 Business Insights
The clustering results provide actionable intelligence for e-commerce platforms:
1. **Targeted Marketing:** Identify high-intent users who spend significant time on product pages to offer real-time discounts.
2. **UX Improvement:** Recognize "Browser" segments that spend time on informational pages and provide them with better educational content or FAQs.
3. **Conversion Optimization:** Understand the behavioral patterns that lead to a "Revenue" event without relying on pre-existing labels.

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the Jupyter Notebook or the Streamlit app: `streamlit run app.py`.