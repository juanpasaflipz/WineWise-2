import pinecone
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import os

def initialize_pinecone():
    """Initialize Pinecone client with error handling"""
    try:
        # Debug: Print if API key exists (but not the key itself)
        st.write("Checking Pinecone API key...")
        if 'PINECONE_API_KEY' not in os.environ:
            st.error("PINECONE_API_KEY environment variable is not set")
            return None
        
        st.write("Initializing Pinecone client...")
        # Initialize Pinecone using the new method
        pinecone_client = pinecone.Pinecone(
            api_key=os.environ['PINECONE_API_KEY']
        )
        
        # List available indexes to debug
        try:
            st.write("Fetching available indexes...")
            indexes = pinecone_client.list_indexes()
            st.write("Raw indexes response:", type(indexes))  # Debug line
            
            if not indexes:
                st.error("No indexes found in your Pinecone account. Please create an index first.")
                return None
            
            # Convert IndexModel objects to a list of index names
            index_names = [index.name for index in indexes]
            st.write("Available indexes:", index_names)
            
            # Get the index if it exists
            target_index = 'wine-embeddings'
            st.write(f"Attempting to connect to index: {target_index}")
            
            if target_index in index_names:
                index = pinecone_client.Index(target_index)
                # Test index connection
                try:
                    # Attempt to describe index statistics
                    stats = index.describe_index_stats()
                    st.write("Index statistics:", stats)
                    st.success("Successfully connected to Pinecone index!")
                    return index
                except Exception as stats_error:
                    st.error(f"Error getting index statistics: {str(stats_error)}")
                    return None
            else:
                st.error(f"Index '{target_index}' not found. Available indexes: {', '.join(index_names)}")
                return None
                
        except Exception as idx_error:
            st.error(f"Error accessing indexes: {str(idx_error)}")
            st.write("Error details:", str(idx_error.__class__.__name__))
            return None
            
    except Exception as e:
        st.error(f"Failed to connect to Pinecone: {str(e)}")
        st.write("Error details:", str(e.__class__.__name__))
        return None

def query_by_metadata(index, metadata_filters, top_k=5):
    """Query wines by metadata using the exact field names from the upload script"""
    try:
        # Filter out empty values and prepare filters
        filter_conditions = {}
        
        if metadata_filters.get("wine_name"):
            filter_conditions["wine_name"] = {"$eq": metadata_filters["wine_name"].lower()}
            
        if metadata_filters.get("region"):
            filter_conditions["region"] = {"$eq": metadata_filters["region"].lower()}
        if metadata_filters.get("country"):
            filter_conditions["country"] = {"$eq": metadata_filters["country"].lower()}
        if metadata_filters.get("type"):
            filter_conditions["type"] = {"$eq": metadata_filters["type"].lower()}
        if metadata_filters.get("color"):
            filter_conditions["color"] = {"$eq": metadata_filters["color"].lower()}
            
        st.write("Debug: Applying metadata filters:", filter_conditions)
        
        if not filter_conditions:
            st.warning("Please enter at least one search criterion")
            return None
            
        # Query with metadata filtering
        query_response = index.query(
            vector=[0] * 384,
            top_k=top_k,
            include_metadata=True,
            filter=filter_conditions
        )
        
        st.write("Debug: Query Response Structure:")
        st.write("Number of matches:", len(query_response.matches))
        if query_response.matches:
            st.write("First match metadata:", query_response.matches[0].metadata)
        
        return query_response
        
    except Exception as e:
        st.error(f"Error querying wines: {str(e)}")
        st.write("Error details:", str(e.__class__.__name__))
        st.write("Debug - filters:", filter_conditions)
        return None

def create_similarity_plot(similarities):
    """Create a radar chart for wine similarities"""
    try:
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
    except Exception as e:
        st.error(f"Error creating similarity plot: {str(e)}")
        return None

def format_wine_details(metadata):
    """Format wine details using the exact field names from the upload script"""
    try:
        return {
            "Name": metadata.get("wine_name", "Unknown"),
            "Producer": metadata.get("producer", "Unknown"),
            "Type": metadata.get("type", "Unknown"),
            "Color": metadata.get("color", "Unknown"),
            "Region": metadata.get("region", "Unknown"),
            "Country": metadata.get("country", "Unknown")
        }
    except Exception as e:
        st.error(f"Error formatting wine details: {str(e)}")
        return {}
