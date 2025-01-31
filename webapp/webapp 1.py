import streamlit as st
import requests
from streamlit_chat import message

# 🎯 Define API URL
API_URL = "http://127.0.0.1:8000"

# ✅ Initialize session state for past recommendations, wishlist, and messages
if "past_recommendations" not in st.session_state:
    st.session_state.past_recommendations = []
if "wishlist" not in st.session_state:
    st.session_state.wishlist = []
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "wishlist_added" not in st.session_state:
    st.session_state["wishlist_added"] = set()
if "pending_wishlist" not in st.session_state:
    st.session_state["pending_wishlist"] = []
if "all_recommended_movies" not in st.session_state:
    st.session_state["all_recommended_movies"] = set()

# 🎭 Sidebar Navigation
st.sidebar.title("📖 Navigation")
page = st.sidebar.radio("Choose Recommendation Type:", ["Chat Mode 🎤", "Past Recommendations 📜", "Wishlist ❤️"])

# ✅ Function to check if the query is plot-based
def is_plot_query(user_query):
    plot_keywords = ["story", "plot", "who is", "can you predict", "about"]
    return any(keyword in user_query.lower() for keyword in plot_keywords)

# -------------------------
# 🗣 Chatbot Mode
# -------------------------
if page == "Chat Mode 🎤":
    st.title("🎬 Welcome to Talk2Rec: Interactive Movie Recommender")
    st.markdown("#### I am your personal movie recommender. How may I help you?")
    
    # Clear Chat Button
    if st.button("🆕 New Chat"):
        st.session_state["messages"] = []
        st.session_state["pending_wishlist"] = []
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state["messages"]):
            is_user = msg["role"] == "user"
            message(msg["content"], is_user=is_user, key=str(i))
    
    # Input area at the bottom
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("", key="input_query", label_visibility="collapsed")
    with col2:
        submit_button = st.button("➡", key="submit_button")
    
    if submit_button and query:
        st.session_state["messages"].append({"role": "user", "content": query})
        message(query, is_user=True)
        
        response = requests.get(f"{API_URL}/recommendations/content/", params={"query": query, "top_n": 5}).json()
        
        if "detail" in response:
            bot_response = "❌ Error fetching recommendations."
        else:
            recommended_movies = [movie['title'] for movie in response]
            st.session_state["pending_wishlist"] = recommended_movies  # Store for wishlist section
            st.session_state["all_recommended_movies"].update(recommended_movies)  # Store all recommended movies
            
            if is_plot_query(query):
                best_match = recommended_movies[0]
                bot_response = f"✅ The best match is: 🎬 {best_match}"
                st.session_state.past_recommendations.insert(0, {"query": query, "movies": [best_match]})
            else:
                bot_response = "✅ Here are your recommended movies:\n" + "\n".join([f"🎬 {movie}" for movie in recommended_movies])
                st.session_state.past_recommendations.insert(0, {"query": query, "movies": recommended_movies})
            
            st.session_state["messages"].append({"role": "assistant", "content": bot_response})
            message(bot_response, is_user=False)
    
    # Separate Wishlist Section in Chat Mode (Placed at Bottom)
    if st.session_state["pending_wishlist"]:
        with st.expander("💖 Add Movies to Wishlist"):
            for movie in st.session_state["pending_wishlist"]:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"🎬 **{movie}**")
                with col2:
                    if st.button(f"❤️ Add to Wishlist", key=f"wishlist_chat_{movie}"):
                        if movie not in st.session_state.wishlist:
                            st.session_state.wishlist.append(movie)
                            st.session_state.wishlist_added.add(movie)
                        st.success(f"✅ {movie} added to your wishlist!")
                        st.rerun()

# -------------------------
# 📜 Past Recommendations 
# -------------------------
elif page == "Past Recommendations 📜":
    st.title("📜 Past Recommendations")
    if not st.session_state.past_recommendations:
        st.info("🕰️ No past recommendations available.")
    else:
        for rec in st.session_state.past_recommendations:
            st.markdown(f"### 🎤 Query: *{rec['query']}*")
            for movie in rec["movies"]:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"🎬 **{movie}**")
                with col2:
                    if movie not in st.session_state.wishlist:
                        if st.button("❤️ Add to Wishlist", key=f"wishlist_past_{movie}"):
                            st.session_state.wishlist.append(movie)
                            st.session_state.wishlist_added.add(movie)
                            st.success(f"✅ {movie} added to your wishlist!")
                            st.rerun()
            st.markdown("---")

# -------------------------
# ❤️ Wishlist 
# -------------------------
elif page == "Wishlist ❤️":
    st.title("❤️ Your Wishlist")
    if not st.session_state.wishlist:
        st.info("No movies in wishlist yet!")
    else:
        for movie in st.session_state.wishlist:
            st.markdown(f"🎬 **{movie}**")
