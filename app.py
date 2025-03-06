import streamlit as st
import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 Load trained graph data & paper metadata
graph_data = torch.load("graph_data.pth")  # Load the graph structure
papers_df = pd.read_csv("papers.csv")  # Load paper metadata

# 🔹 Define GCN Model (same architecture as before)
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, 128)
        self.conv2 = pyg_nn.GCNConv(128, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# 🔹 Initialize model & load weights
model_gnn = GCN(in_channels=768, out_channels=128)
model_gnn.load_state_dict(torch.load("gnn_model.pth"))
model_gnn.eval()  # Set to evaluation mode

# 🔹 Function to recommend similar papers
def recommend_papers(paper_id, top_k=5, min_year=None, min_citations=0, venue_filter=None):
    """Finds the top-K most similar papers based on embeddings and applies filters."""
    paper_vec = graph_data.x[paper_id].detach().numpy().reshape(1, -1)
    similarities = cosine_similarity(paper_vec, graph_data.x.detach().numpy())
    top_papers = np.argsort(similarities[0])[::-1][1:]  # Exclude self

    recommended_papers = papers_df.loc[top_papers, ["title", "authors", "year", "venue", "citationCount"]]
    recommended_papers = recommended_papers.rename(columns={"title": "Title", "authors": "Authors", "year": "Year", "venue": "Venue", "citationCount": "Citations"})

    # Apply filters
    if min_year:
        recommended_papers = recommended_papers[recommended_papers["Year"] >= min_year]
    recommended_papers = recommended_papers[recommended_papers["Citations"] >= min_citations]
    if venue_filter and venue_filter != "All":
        recommended_papers = recommended_papers[recommended_papers["Venue"].str.contains(venue_filter, na=False, case=False)]

    return recommended_papers.head(top_k)  # Return top-k filtered results

# 🔹 Streamlit Web App UI
st.title("📚 AI-Powered Academic Paper Recommender")

# User input: Paper ID
paper_id = st.number_input("Enter Paper ID:", min_value=0, step=1)

# 🔹 Filters
st.sidebar.header("📌 Filter Options")
min_year = st.sidebar.number_input("Minimum Year", min_value=2000, max_value=2025, value=2015, step=1)
min_citations = st.sidebar.slider("Minimum Citations", min_value=0, max_value=5000, value=10, step=10)
venue_filter = st.sidebar.selectbox("Select Venue Type", ["All", "Conference", "Journal", "Workshop"])

if st.button("🔍 Get Recommendations"):
    recommended_papers = recommend_papers(paper_id, top_k=5, min_year=min_year, min_citations=min_citations, venue_filter=venue_filter)

    if recommended_papers.empty:
        st.warning("⚠️ No matching papers found with the selected filters!")
    else:
        st.write("### 📄 Recommended Papers:")
        st.dataframe(recommended_papers)
