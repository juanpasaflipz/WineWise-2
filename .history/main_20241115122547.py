import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
from utils import (
    initialize_pinecone,
    query_by_metadata,
    create_similarity_plot,
    format_wine_details
)
from styles import apply_custom_styles

def main():
    # Page configuration must be the first Streamlit command
    st.set_page_config(
        page_title="Wine Recommendations",
        page_icon="🍷",
        layout="wide"
    )
    
    # Apply custom styles after page configuration
    apply_custom_styles()
    
    st.title("🍷 Wine Recommendation System")
    
    # Debug section
    st.write("Initializing application...")
    
    # Initialize Pinecone
    with st.spinner("Connecting to Pinecone..."):
        index = initialize_pinecone()
        if not index:
            st.error("Failed to initialize Pinecone. Please check the error messages above.")
            st.stop()
    
    # Sidebar for search options
    with st.sidebar:
        st.header("Search Options")
        st.write("Search for wines using any combination of criteria below.")
        
        wine_name = st.text_input(
            "Wine Name",
            placeholder="Enter wine name",
            help="Enter full or partial wine name"
        )
        
        region = st.text_input(
            "Region",
            placeholder="Enter wine region",
            help="Enter the wine region"
        )
        
        country = st.text_input(
            "Country",
            placeholder="Enter country",
            help="Enter the country of origin"
        )
        
        wine_type = st.selectbox(
            "Type",
            options=["", "Red", "White", "Rosé", "Sparkling"],
            help="Select the type of wine"
        )
        
        wine_color = st.selectbox(
            "Color",
            options=["", "Red", "White", "Rosé", "Gold"],
            help="Select the color of wine"
        )
        
        num_recommendations = st.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=10,
            value=5
        )
    
    # Main content
    # Create metadata filters
    metadata_filters = {
        "wine_name": wine_name,
        "region": region,
        "country": country,
        "type": wine_type,
        "color": wine_color
    }
    
    # Check if any search criteria are provided
    if any(metadata_filters.values()):
        with st.spinner("Finding matching wines..."):
            results = query_by_metadata(index, metadata_filters, num_recommendations)
            
            if results and results.matches:
                st.success(f"Found {len(results.matches)} matching wines!")
                
                # Create two columns for layout
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Matching Wines")
                    
                    # Display matching wines
                    for match in results.matches:
                        with st.container():
                            st.markdown(
                                f"""
                                <div class="wine-card">
                                    <h3>{match.metadata.get('wine_name', 'Unknown Wine')}</h3>
                                    <p class="similarity-score">Match Score: {match.score:.2f}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            details = format_wine_details(match.metadata)
                            for key, value in details.items():
                                st.write(f"**{key}:** {value}")
                            
                            st.divider()
                
                with col2:
                    st.subheader("Match Score Visualization")
                    # Create and display similarity plot
                    similarities = [match.score for match in results.matches]
                    fig = create_similarity_plot(similarities)
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning("No matching wines found. Please try different search criteria.")
    
    else:
        st.info("👈 Enter search criteria in the sidebar to find wines!")
        
        # Example section
        with st.expander("Need help getting started?"):
            st.write("""
                Try searching with these example criteria:
                - Wine Type: Red
                - Country: France
                - Region: Burgundy
                
                Or just enter a wine name to search by name only!
            """)
    
    # Footer
    st.markdown("""
        ---
        Made with ❤️ for wine enthusiasts
    """)

if __name__ == "__main__":
    main()
