"""
Streamlit UI for Enterprise RAG System
"""

import streamlit as st
import requests
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .source-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-high { color: #4CAF50; }
    .confidence-medium { color: #FF9800; }
    .confidence-low { color: #F44336; }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8000"


def query_rag_system(
    query: str,
    collection: str,
    top_k: int,
) -> Optional[dict]:
    """Send query to RAG API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/query",
            json={
                "query": query,
                "collection": collection,
                "top_k": top_k,
                "include_sources": True,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error querying API: {str(e)}")
        return None


# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=RAG+System", width=150)
    st.markdown("---")

    st.subheader("⚙️ Settings")
    collection = st.selectbox(
        "Document Collection",
        ["hr-policies", "technical-docs", "product-specs", "default"],
        index=0,
    )

    top_k = st.slider(
        "Number of Sources",
        min_value=1,
        max_value=10,
        value=5,
    )

    st.markdown("---")
    st.markdown("### 📊 System Status")
    st.success("✅ API Connected")
    st.info("📚 3 Collections Available")

# Main content
st.markdown('<p class="main-header">🎯 Enterprise RAG System</p>', unsafe_allow_html=True)
st.markdown("Ask questions about your enterprise knowledge base")

# Query input
query = st.text_input(
    "Enter your question:",
    placeholder="What is our company policy on remote work?",
    key="query_input",
)

# Search button
if st.button("🔍 Search", type="primary") or query:
    if query:
        with st.spinner("Searching knowledge base..."):
            result = query_rag_system(query, collection, top_k)

        if result:
            # Display answer
            st.markdown("### 💡 Answer")
            st.markdown(result.get("answer", "No answer available"))

            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                confidence = result.get("confidence", 0)
                confidence_class = (
                    "confidence-high" if confidence > 0.8
                    else "confidence-medium" if confidence > 0.5
                    else "confidence-low"
                )
                st.metric("Confidence", f"{confidence:.0%}")
            with col2:
                st.metric("Sources", len(result.get("sources", [])))

            # Display sources
            if result.get("sources"):
                st.markdown("### 📚 Sources")
                for i, source in enumerate(result.get("sources", []), 1):
                    with st.expander(
                        f"Source {i}: {source.get('document', 'Unknown')} "
                        f"(Relevance: {source.get('relevance_score', 0):.2f})"
                    ):
                        st.markdown(f"**Excerpt:**")
                        st.markdown(f"> {source.get('text', 'No text available')}")
    else:
        st.warning("Please enter a question")

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using LangChain, Pinecone, and GPT-4 | "
    "[GitHub](https://github.com/jinno-ai/enterprise-rag-system)"
)
