import streamlit as st
import pandas as pd
from utils import (
    initialize_pinecone,
    query_similar_wines,
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
        st.write("Enter a wine ID to find similar wines.")
        
        wine_id = st.text_input(
            "Enter Wine ID",
            placeholder="e.g., wine_123",
            help="Enter the ID of the wine you want to find similar wines for"
        )
        
        num_recommendations = st.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=10,
            value=5
        )
    
    # Main content
    if wine_id:
        with st.spinner("Finding similar wines..."):
            results = query_similar_wines(index, wine_id, num_recommendations)
            
            if results and results.matches:
                st.success(f"Found {len(results.matches)} similar wines!")
                
                # Create two columns for layout
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Similar Wines")
                    
                    # Display similar wines
                    for match in results.matches:
                        with st.container():
                            st.markdown(
                                f"""
                                <div class="wine-card">
                                    <h3>{match.metadata.get('name', 'Unknown Wine')}</h3>
                                    <p class="similarity-score">Similarity: {match.score:.2f}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            details = format_wine_details(match.metadata)
                            for key, value in details.items():
                                st.write(f"**{key}:** {value}")
                            
                            st.divider()
                
                with col2:
                    st.subheader("Similarity Visualization")
                    # Create and display similarity plot
                    similarities = [match.score for match in results.matches]
                    fig = create_similarity_plot(similarities)
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning("No similar wines found. Please try a different wine ID.")
    
    else:
        st.info("👈 Enter a wine ID in the sidebar to get started!")
        
        # Example section
        with st.expander("Need help getting started?"):
            st.write("""
                Try using one of these example wine IDs:
                - wine_123
                - wine_456
                - wine_789
            """)
    
    # Footer
    st.markdown("""
        ---
        Made with ❤️ for wine enthusiasts
    """)

if __name__ == "__main__":
    main()
