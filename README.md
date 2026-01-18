# Customer Segmentation using Clustering

Unsupervised learning project to segment wholesale customers based on spending patterns across product categories.

## Dataset

Using the **Wholesale Customers** dataset from UCI Machine Learning Repository. Contains 440 customers with annual spending data across 6 product categories.

Source: https://archive.ics.uci.edu/ml/datasets/Wholesale+customers

### Features

| Feature | Description |
|---------|-------------|
| Channel | 1 = Hotel/Restaurant/Cafe, 2 = Retail |
| Region | 1 = Lisbon, 2 = Oporto, 3 = Other |
| Fresh | Annual spending on fresh products |
| Milk | Annual spending on milk products |
| Grocery | Annual spending on grocery |
| Frozen | Annual spending on frozen products |
| Detergents_Paper | Annual spending on detergents and paper |
| Delicassen | Annual spending on deli products |

## Clustering Algorithms

The notebook compares 6 different algorithms:

1. **K-Means** - Standard centroid-based clustering
2. **Agglomerative (Hierarchical)** - Bottom-up connectivity approach
3. **DBSCAN** - Density-based, can identify noise points
4. **Gaussian Mixture Model** - Probabilistic, soft clustering
5. **Mean Shift** - Automatically finds number of clusters
6. **Fuzzy C-Means** - Soft membership to multiple clusters

## Evaluation Metrics

Since this is unsupervised learning (no ground truth labels), we use internal metrics:

- **Silhouette Score**: Measures cluster cohesion vs separation. Range [-1, 1], higher is better.
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance. Higher is better.
- **Davies-Bouldin Index**: Average similarity between clusters. Lower is better.
- **Fuzzy Partition Coefficient (FCM only)**: Measures overlap between clusters.

## Finding Optimal K

Three methods used:
- Elbow method (inertia plot)
- Silhouette analysis
- Dendrogram visualization

## Key Results

Best performing: K-Means with K=3

Three customer segments identified:
- Fresh-focused customers (mostly HoReCa channel)
- Grocery/Household-focused customers (mostly Retail channel)
- Balanced spenders across categories

## Preprocessing Steps

1. Log transformation (data was heavily right-skewed)
2. StandardScaler normalization
3. Used only spending features for clustering (Channel/Region for validation)

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
scikit-fuzzy
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy scikit-fuzzy
```

## Usage

```bash
jupyter notebook ClusteringML.ipynb
```

The notebook loads data directly from UCI repository, so no local data files needed.

## Notes

- DBSCAN found many noise points with this dataset - not ideal for this use case
- Fuzzy C-Means useful for identifying customers with mixed purchasing patterns
- Log transform was important due to heavy skewness in spending data

## Project Structure

```
ClusteringML.ipynb   - Main analysis notebook
README.md            - This file
```
