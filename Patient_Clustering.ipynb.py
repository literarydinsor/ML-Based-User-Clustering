#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

# Define the features we want
n_samples = 1000
n_features = 5

# Generate a simulated dataset
data, _ = make_blobs(n_samples=n_samples, n_features=n_features, random_state=42)

# Create a DataFrame from the simulated data
df = pd.DataFrame(data, columns=['Age', 'Gender', 'Purchasing_Power', 'Frequency', 'Health_Risk_Score'])

# For simplicity, let's convert the continuous 'Gender' feature to binary (0 or 1)
df['Gender'] = (df['Gender'] > df['Gender'].median()).astype(int)

# Display the first few rows of the DataFrame
df.head()

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Pairplot to visualize the distribution of data and the relationship between features
sns.pairplot(df, hue='Gender')
plt.show()

# In[ ]:


from sklearn.preprocessing import StandardScaler

# Initialize a scaler
scaler = StandardScaler()

# Fit the scaler to the data and transform the data
df_scaled = scaler.fit_transform(df)

# Convert the scaled data into a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Display the first few rows of the scaled DataFrame
df_scaled.head()

# In[ ]:


from sklearn.cluster import KMeans

# Initialize a KMeans model
kmeans = KMeans(n_clusters=5, random_state=42)

# Fit the model to the scaled data
kmeans.fit(df_scaled)

# Get the cluster assignments for each data point
clusters = kmeans.labels_

# Add the cluster assignments to the original (unscaled) DataFrame
df['Cluster'] = clusters

# Display the first few rows of the DataFrame with the cluster assignments
df.head()

# In[ ]:


from sklearn.metrics import silhouette_score

# Calculate the silhouette score
silhouette = silhouette_score(df_scaled, clusters)

# Print the silhouette score
silhouette

# In[ ]:


# Calculate the mean values of the features for each cluster
cluster_profiles = df.groupby('Cluster').mean()

# Display the cluster profiles
cluster_profiles

# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot each cluster
for cluster in df['Cluster'].unique():
    # Select only the data observations with cluster number equal to the current
    tmp = df[df['Cluster'] == cluster]
    ax.scatter(tmp['Age'], tmp['Purchasing_Power'], tmp['Health_Risk_Score'], label=f'Cluster {cluster}')

# Set labels and legend
ax.set_xlabel('Age')
ax.set_ylabel('Purchasing Power')
ax.set_zlabel('Health Risk Score')
ax.legend()

# Show the plot
plt.show()

# In[ ]:


# Create a 2D plot for Age vs Purchasing Power colored by Cluster
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='Purchasing_Power', hue='Cluster', data=df, palette='viridis')
plt.title('Age vs Purchasing Power')
plt.show()

# Create a 2D plot for Age vs Health Risk Score colored by Cluster
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='Health_Risk_Score', hue='Cluster', data=df, palette='viridis')
plt.title('Age vs Health Risk Score')
plt.show()

# Create a 2D plot for Purchasing Power vs Health Risk Score colored by Cluster
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Purchasing_Power', y='Health_Risk_Score', hue='Cluster', data=df, palette='viridis')
plt.title('Purchasing Power vs Health Risk Score')
plt.show()

# In[ ]:


# Regenerate the simulated dataset
data, _ = make_blobs(n_samples=n_samples, n_features=n_features, random_state=42)

# Convert the data to a DataFrame
df = pd.DataFrame(data, columns=['Age', 'Gender', 'Purchasing_Power', 'Frequency', 'Health_Risk_Score'])

# Make sure Age is positive
df['Age'] = df['Age'].abs()

# Convert the continuous 'Gender' feature to binary (0 or 1)
df['Gender'] = (df['Gender'] > df['Gender'].median()).astype(int)

# Display the first few rows of the DataFrame
df.head()

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Pairplot to visualize the distribution of data and the relationship between features
sns.pairplot(df, hue='Gender')
plt.show()

# In[ ]:


from sklearn.preprocessing import StandardScaler

# Initialize a StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data and transform the data
df_scaled = scaler.fit_transform(df)

# Convert the scaled data back to a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Display the first few rows of the scaled DataFrame
df_scaled.head()

# In[ ]:


from sklearn.cluster import KMeans

# List to hold the SSE for each number of clusters
sse = []

# We will check the SSE for 1-10 clusters
for k in range(1, 11):
    # Initialize a KMeans object
    kmeans = KMeans(n_clusters=k, random_state=42)

    # Fit the KMeans object to the data
    kmeans.fit(df_scaled)

    # Append the SSE for this number of clusters to the list
    sse.append(kmeans.inertia_)

# Plot the SSE for each number of clusters
plt.figure(figsize=(6, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.show()

# In[ ]:


# Initialize a KMeans object with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the KMeans object to the data and predict the cluster for each data point
clusters = kmeans.fit_predict(df_scaled)

# Add the clusters to the DataFrame
df['Cluster'] = clusters

# Display the first few rows of the DataFrame with the clusters
df.head()

# In[ ]:


# Pairplot to visualize the clusters
sns.pairplot(df, hue='Cluster', palette='Dark2')
plt.show()

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D

# Initialize a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points in the 3D space
scatter = ax.scatter(df['Age'], df['Purchasing_Power'], df['Health_Risk_Score'], c=df['Cluster'], cmap='Dark2')

# Set the labels for the axes
ax.set_xlabel('Age')
ax.set_ylabel('Purchasing Power')
ax.set_zlabel('Health Risk Score')

# Set the title for the plot
plt.title('3D view of the clusters')

# Add a color bar
plt.colorbar(scatter)

# Show the plot
plt.show()

# In[ ]:


# Regenerate the simulated dataset
data, _ = make_blobs(n_samples=n_samples, n_features=n_features, random_state=42)

# Convert the data to a DataFrame
df = pd.DataFrame(data, columns=['Age', 'Gender', 'Purchasing_Power', 'Frequency', 'Health_Risk_Score'])

# Make sure Age and Frequency are positive and Age ranges from 0 to 80
df['Age'] = df['Age'].abs() * 4  # scale to 0-80
df['Frequency'] = df['Frequency'].abs()

# Convert the continuous 'Gender' feature to binary (0 or 1)
df['Gender'] = (df['Gender'] > df['Gender'].median()).astype(int)

# Display the first few rows of the DataFrame
df.head()

# In[ ]:


# Pairplot to visualize the distribution of data and the relationship between features
sns.pairplot(df, hue='Gender')
plt.show()

# In[ ]:


# Initialize a StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data and transform the data
df_scaled = scaler.fit_transform(df)

# Convert the scaled data back to a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Display the first few rows of the scaled DataFrame
df_scaled.head()

# In[ ]:


# List to hold the SSE for each number of clusters
sse = []

# We will check the SSE for 1-10 clusters
for k in range(1, 11):
    # Initialize a KMeans object
    kmeans = KMeans(n_clusters=k, random_state=42)

    # Fit the KMeans object to the data
    kmeans.fit(df_scaled)

    # Append the SSE for this number of clusters to the list
    sse.append(kmeans.inertia_)

# Plot the SSE for each number of clusters
plt.figure(figsize=(6, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.show()

# In[ ]:


# Initialize a KMeans object with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the KMeans object to the data and predict the cluster labels
clusters = kmeans.fit_predict(df_scaled)

# Add the cluster labels to the DataFrame
df['Cluster'] = clusters

# Display the first few rows of the DataFrame with the cluster labels
df.head()

# In[ ]:


# Initialize a 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot each cluster
for cluster in df['Cluster'].unique():
    # Get the data for this cluster
    cluster_data = df[df['Cluster'] == cluster]

    # Plot the data
    ax.scatter(cluster_data['Age'], cluster_data['Purchasing_Power'], cluster_data['Frequency'], label=f'Cluster {cluster}')

# Set the labels and title
ax.set_xlabel('Age')
ax.set_ylabel('Purchasing Power')
ax.set_zlabel('Frequency')
ax.set_title('Clusters')
ax.legend()

# Show the plot
plt.show()

# In[ ]:


# Pairplot to visualize the clusters in the context of all features
sns.pairplot(df, hue='Cluster')
plt.show()

# In[ ]:


# Initialize a 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot each cluster
for cluster in df['Cluster'].unique():
    # Get the data for this cluster
    cluster_data = df[df['Cluster'] == cluster]

    # Plot the data
    ax.scatter(cluster_data['Age'], cluster_data['Purchasing_Power'], cluster_data['Health_Risk_Score'], label=f'Cluster {cluster}')

# Set the labels and title
ax.set_xlabel('Age')
ax.set_ylabel('Purchasing Power')
ax.set_zlabel('Health Risk Score')
ax.set_title('Clusters')
ax.legend()

# Show the plot
plt.show()

# In[ ]:


# Pairplot to visualize the clusters in the context of all features
sns.pairplot(df, hue='Cluster')
plt.show()
