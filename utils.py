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
    try:
        # Filter out empty values and prepare filters
        filters = {}
        
        # Handle text-based fields with exact matching
        if metadata_filters.get("wine_name"):
            # Simple exact match on the DISPLAY_NAME field
            filters["DISPLAY_NAME"] = metadata_filters["wine_name"]
            
        # Handle exact match fields
        if metadata_filters.get("region"):
            filters["REGION"] = metadata_filters["region"]
        if metadata_filters.get("country"):
            filters["COUNTRY"] = metadata_filters["country"]
        if metadata_filters.get("type"):
            filters["TYPE"] = metadata_filters["type"]
        if metadata_filters.get("color"):
            filters["COLOUR"] = metadata_filters["color"]
            
        st.write("Debug: Applying metadata filters:", filters)
        
        if not filters:
            st.warning("Please enter at least one search criterion")
            return None
            
        # Query with metadata filtering
        query_response = index.query(
            vector=[0] * 384,  # Dummy vector since we're filtering by metadata
            top_k=top_k,
            include_metadata=True,
            filter=filters
        )
        
        st.write("Debug: Query Response Structure:")
        st.write("Number of matches:", len(query_response.matches))
        if query_response.matches:
            st.write("First match metadata:", query_response.matches[0].metadata)
        
        return query_response
        
    except Exception as e:
        st.error(f"Error querying wines: {str(e)}")
        st.write("Error details:", str(e.__class__.__name__))
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
    try:
        return {
            "Name": metadata.get("DISPLAY_NAME", "Unknown"),
            "Producer": metadata.get("PRODUCER_NAME", "Unknown"),
            "Type": metadata.get("TYPE", "Unknown"),
            "Color": metadata.get("COLOUR", "Unknown"),
            "Region": metadata.get("REGION", "Unknown"),
            "Country": metadata.get("COUNTRY", "Unknown")
        }
    except Exception as e:
        st.error(f"Error formatting wine details: {str(e)}")
        return {}
