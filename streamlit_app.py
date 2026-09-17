import streamlit as st
from src.catalog_service import CatalogResult, answer_catalog_question


st.set_page_config(
    page_title="CatalogIQ - Hybrid RAG E-Commerce Support",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp > header {
        background-color: rgba(255, 255, 255, 0);
    }
    .catalog-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .catalog-subtitle {
        color: #52606d;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .scope-badge {
        display: inline-block;
        padding: 0.4rem 0.85rem;
        border-radius: 999px;
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        color: #1b5e20;
        font-size: 0.85rem;
        font-weight: 650;
        margin-bottom: 1rem;
        border: 1px solid #a5d6a7;
    }
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1976d2;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ff9800;
    }
    .example-btn {
        text-align: left;
        white-space: normal;
    }
    .stChatMessage {
        border-radius: 12px;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"] p:contains("out of scope")) {
        background-color: #fff3e0;
    }
    .stExpander {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_details" not in st.session_state:
    st.session_state.show_details = False
if "last_out_of_scope" not in st.session_state:
    st.session_state.last_out_of_scope = False


EXAMPLE_CATEGORIES = {
    "👕 Apparel & Catalog Products": [
        "Tell me about the 11 Degrees Core Pull Over Hoodie",
        "What are the available sizes and material details for the pullover hoodie?",
        "What colors does the 11 Degrees hoodie come in?",
        "Is the 11 Degrees hoodie machine washable?",
    ],
    "🔧 Specs & Technical Parts": [
        "What is the torque spec for screw S-4012-SCRW?",
        "Tell me about part S-4012-SCRW",
        "Which part number is the gas lift cylinder for CHAIR-ERG-X99?",
        "What are the dimensions of the CHAIR-ERG-X99?",
        "What is the weight capacity of DESK-STD-MOTO?",
        "Battery life of HEADSET-WL-PRO",
        "Actuation force for keyboard switch SW-2201-SWT",
    ],
    "🚨 Troubleshooting & Repairs": [
        "What does error E02 mean on DESK-STD-MOTO?",
        "Standing desk won't move up or down, shows error E01",
        "How do I replace the armrest bracket on the ErgoFlex chair?",
        "My chair squeaks when I lean back, how do I fix it?",
        "Left side of desk lags behind right when raising",
        "Monitor arm MON-ARM-DUAL won't hold position",
        "How to reset the standing desk controller CB-2201-CTL?",
        "Mouse cursor jumps around on glass desk",
    ],
    "⚖️ Comparisons": [
        "Compare CHAIR-ERG-X99 and DESK-STD-MOTO",
        "Which is better: CHAIR-ERG-X99 vs HEADSET-WL-PRO for long sessions?",
        "Difference between KB-ERGO-SPLIT and MOUSE-VERT-PRO",
    ],
    "📋 Warranty & Support": [
        "What is the warranty on CHAIR-ERG-X99?",
        "Warranty coverage for DESK-STD-MOTO",
        "How to claim warranty for HEADSET-WL-PRO?",
    ],
    "🚫 Out-of-Scope (Test Rejection)": [
        "How do I cook a turkey?",
        "What is the weather forecast for tomorrow?",
        "Who won the game last night?",
        "What's the best recipe for pasta?",
    ],
}


def render_result(result: CatalogResult) -> None:
    if result.error:
        st.error(f"⚠️ Service Warning: {result.error}")

    if result.out_of_scope:
        st.warning("🚫 **Query Out of Scope**")
        st.markdown(result.answer)
        if result.scope_reason:
            st.caption(f"💡 Reason: {result.scope_reason}")
        st.markdown(
            """
            <div class="warning-box">
            <strong>💡 Tip:</strong> I can only answer questions about products, parts, specifications, 
            troubleshooting, assembly, and warranty information in the catalog.<br><br>
            <strong>Try asking about:</strong>
            <ul>
                <li>Product details (e.g., "Tell me about CHAIR-ERG-X99")</li>
                <li>Specifications (e.g., "Torque spec for S-4012-SCRW")</li>
                <li>Troubleshooting (e.g., "Error E01 on standing desk")</li>
                <li>Comparisons (e.g., "Compare CHAIR-ERG-X99 vs DESK-STD-MOTO")</li>
                <li>Warranty info (e.g., "Warranty on CHAIR-ERG-X99")</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(result.answer)

    if not st.session_state.show_details:
        return

    timing = result.timing_ms
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔍 Search", f"{timing.get('sparse_dense_search_ms', 0):.0f} ms")
    with col2:
        st.metric("🎯 Rerank", f"{timing.get('rerank_ms', 0):.0f} ms")
    with col3:
        st.metric("🤖 Generation", f"{timing.get('generation_ms', 0):.0f} ms")
    with col4:
        st.metric("⏱️ Total", f"{timing.get('total_ms', 0):.0f} ms")

    if result.sources:
        with st.expander(f"📚 Retrieved Sources ({len(result.sources)})", expanded=False):
            st.dataframe(
                result.sources,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "doc_id": st.column_config.TextColumn("Document ID", width="medium"),
                    "sku": st.column_config.TextColumn("SKU", width="small"),
                    "product_name": st.column_config.TextColumn("Product", width="large"),
                }
            )

    citation = result.citation_check
    if citation:
        with st.expander("🔗 Citation Check", expanded=False):
            st.write({
                "✅ Valid Citations": citation.get("valid_citations", []),
                "❌ Invalid Citations": citation.get("invalid_citations", []),
                "✓ All Valid": citation.get("all_valid", False),
            })


def submit_query(query: str) -> None:
    query = query.strip()
    if not query:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching catalog..."):
            result = answer_catalog_question(query)
            render_result(result)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "result": result,
        }
    )


# Header
st.markdown(
    '<div class="catalog-title">CatalogIQ</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="catalog-subtitle">Enterprise technical support grounded in official product documentation, parts catalogs, and support tickets.</div>',
    unsafe_allow_html=True,
)
st.markdown('<span class="scope-badge">✅ Grounded mode enabled • Hybrid RAG (BM25 + Dense + Reranker)</span>', unsafe_allow_html=True)

pending_query = None

# Sidebar
with st.sidebar:
    st.markdown("## 📚 Catalog Scope")
    st.markdown(
        '<div class="info-box">'
        '<strong>CatalogIQ</strong> answers questions about supported products, part numbers, '
        'specifications, assembly guidance, troubleshooting, and warranty information.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### ✅ What You Can Ask")
    st.markdown(
        """
        - **Product overviews** — `Tell me about CHAIR-ERG-X99`
        - **Exact SKUs/Part IDs** — `S-4012-SCRW`, `CHAIR-ERG-X99`
        - **Specifications** — `Torque spec`, `Dimensions`, `Weight`, `Battery life`
        - **Troubleshooting** — `Error E01`, `Won't move`, `Squeaking`, `How to fix`
        - **Repairs/Replacements** — `How to replace armrest`, `Reset controller`
        - **Comparisons** — `Compare CHAIR-ERG-X99 vs DESK-STD-MOTO`
        - **Warranty** — `Warranty on CHAIR-ERG-X99`
        - **Assembly/Installation** — `How to assemble`, `Installation instructions`
        """
    )

    st.markdown("### ❌ What I Cannot Answer")
    st.markdown(
        """
        - General knowledge (cooking, weather, sports)
        - Personal advice or opinions
        - Products not in the catalog
        - Unrelated technical topics
        """
    )

    st.markdown("### 🎯 Example Queries")
    for category, examples in EXAMPLE_CATEGORIES.items():
        with st.expander(category, expanded=False):
            for example in examples:
                if st.button(
                    example,
                    key=f"btn_{hash(example)}",
                    use_container_width=True,
                    help=f"Click to ask: {example}",
                ):
                    pending_query = example

    st.divider()

    st.session_state.show_details = st.toggle(
        "🔧 Show retrieval details (timing, sources, citations)",
        value=st.session_state.show_details,
    )

    if st.button("🗑️ Clear Conversation", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.last_out_of_scope = False
        st.rerun()

    st.divider()
    st.caption("💡 **Pro tip:** Use specific SKUs (e.g., `S-4012-SCRW`) or product names (e.g., `CHAIR-ERG-X99`) for best results.")

# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            result = message.get("result")
            if result:
                render_result(result)
            else:
                st.markdown(message["content"])

# Show helpful hint if last query was out of scope
if st.session_state.last_out_of_scope and len(st.session_state.messages) > 0:
    last_msg = st.session_state.messages[-1]
    if last_msg.get("role") == "assistant":
        result = last_msg.get("result")
        if result and result.out_of_scope:
            st.info("💡 **Need help?** Try one of the example queries in the sidebar, or ask about a specific product, part number, error code, or troubleshooting issue.")

user_input = st.chat_input(
    "Ask about a product, part, specification, troubleshooting, or warranty...",
    key="chat_input"
)

query_to_run = pending_query or user_input
if query_to_run:
    submit_query(query_to_run)
    # Reset pending query after execution
    if pending_query:
        pending_query = None