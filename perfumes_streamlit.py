import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# loading the dataset from csv
ulta_filtered = pd.read_csv('ulta_filtered.csv')

# Extract  brands and products
brands = np.sort(ulta_filtered['Brand'].unique())
products = np.sort(ulta_filtered['Product_Name'].unique())

# Building the pivot table for SVD
rating_crosstab = ulta_filtered.pivot_table(values='rating', index='user_loc', columns='Product_Name', fill_value=0)
X = rating_crosstab.values.T

# Apply SVD for dimensionality reduction
SVD = TruncatedSVD(n_components=60, random_state=17)
result_matrix_60 = SVD.fit_transform(X)

# perfume-perfume similarity matrix with cosine similarity
item_similarity_matrix = cosine_similarity(result_matrix_60)

# Get the perfume names
perfume_ids = rating_crosstab.T.index.tolist()

# Convert similarity matrix into df
item_similarity_df = pd.DataFrame(item_similarity_matrix, 
                                  index=perfume_ids, 
                                  columns=perfume_ids)

# Streamlit UI
st.title('Perfume Recommendation System')
st.subheader('Based on women perfumes and user ratings at ULTA')
st.header('If you love:')

# dropdown lists for user to select brand and product
selected_brand = st.selectbox('Select a brand:', brands)
filtered_products = ulta_filtered[ulta_filtered['Brand'] == selected_brand]['Product_Name'].unique()
selected_product = st.selectbox('Select a product:', filtered_products)


if selected_product:
    with st.spinner('Personalizing your recommendations...'):
        time.sleep(5)
        similarities = item_similarity_df[selected_product]
        top_similar = similarities.drop(index=selected_product).sort_values(ascending=False).head(5)

        
        st.subheader("You may want to try these 5 products:")
        for i, (product, similarity) in enumerate(top_similar.items(), 1):
            st.write(f"{i}. {product} (Similarity: {similarity:.2f})")
