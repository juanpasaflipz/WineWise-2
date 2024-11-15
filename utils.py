import pinecone
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import os

def initialize_pinecone():
    """Initialize Pinecone client with error handling"""
    try:
        # Initialize Pinecone using the new method
        pinecone_client = pinecone.Pinecone(
            api_key=os.environ['PINECONE_API_KEY']
        )
        
        # List available indexes to debug
        try:
            indexes = pinecone_client.list_indexes()
            if not indexes:
                st.error("No indexes found in your Pinecone account. Please create an index first.")
                return None
            
            # Convert IndexModel objects to a list of index names
            index_names = [index.name for index in indexes]
            st.write("Available indexes:", index_names)
            
            # Get the index if it exists
            target_index = 'wine-embeddings-1eowpvt'
            if target_index in index_names:
                index = pinecone_client.Index(target_index)
                return index
            else:
                st.error(f"Index '{target_index}' not found. Available indexes: {', '.join(index_names)}")
                return None
                
        except Exception as idx_error:
            st.error(f"Error accessing indexes: {str(idx_error)}")
            return None
            
    except Exception as e:
        st.error(f"Failed to connect to Pinecone: {str(e)}")
        if 'PINECONE_API_KEY' not in os.environ:
            st.error("PINECONE_API_KEY environment variable is not set")
        return None

def query_similar_wines(index, wine_id, top_k=5):
    """Query Pinecone for similar wines"""
    try:
        query_response = index.query(
            id=wine_id,
            top_k=top_k,
            include_metadata=True
        )
        return query_response
    except Exception as e:
        st.error(f"Error querying similar wines: {str(e)}")
        return None

def create_similarity_plot(similarities):
    """Create a radar chart for wine similarities"""
    labels = [f"Wine {i+1}" for i in range(len(similarities))]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=similarities,
        theta=labels,
        fill='toself',
        line_color='#722F37'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

def format_wine_details(metadata):
    """Format wine metadata for display"""
    return {
        "Name": metadata.get("name", "Unknown"),
        "Variety": metadata.get("variety", "Unknown"),
        "Winery": metadata.get("winery", "Unknown"),
        "Region": metadata.get("region", "Unknown"),
        "Country": metadata.get("country", "Unknown"),
        "Price": f"${metadata.get('price', 'N/A')}",
        "Points": metadata.get("points", "N/A")
    }
