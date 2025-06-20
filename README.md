# Customer Segmentation using K-Means Clustering

This project uses the K-Means clustering algorithm to segment customers of a retail store based on their annual income and spending score. The goal is to identify distinct customer groups for targeted marketing strategies.

## 📊 Project Overview

The script performs the following steps:
1.  **Loads Data**: Reads customer data from `Mall_Customers.csv`.
2.  **Data Preparation**: Selects `Annual Income (k$)` and `Spending Score (1-100)` as the features for clustering and standardizes them.
3.  **Optimal Cluster Selection**: Uses the Elbow Method to determine the optimal number of clusters (K).
4.  **Clustering**: Applies the K-Means algorithm with the optimal K to group customers.
5.  **Visualization**: Generates a scatter plot to visualize the distinct customer segments.

## 🚀 Technologies Used

- Python
- pandas
- scikit-learn
- matplotlib

## ⚙️ Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lakx711/SCT_ML_2.git
    cd SCT_ML_2
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the script:**
    ```bash
    python main.py
    ```
    This will execute the script and display two plots: the Elbow Method graph followed by the customer cluster visualization.

## 📈 Results

The analysis identifies 5 distinct customer clusters, which are visualized in a scatter plot. This segmentation allows for a deeper understanding of the customer base, enabling the business to tailor its marketing efforts more effectively.

![Customer Segments](https://i.imgur.com/example.png)  <!-- You can replace this with a screenshot of your cluster plot --> 