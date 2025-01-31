import streamlit as st
import requests
from streamlit_chat import message

# 🎯 Define API URL
API_URL = "http://127.0.0.1:8000"

# ✅ Initialize session state for user authentication and data storage
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "user_data" not in st.session_state:
    st.session_state.user_data = {}  # Stores past recommendations & watchlist per user
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "pending_watchlist" not in st.session_state:
    st.session_state["pending_watchlist"] = []
if "all_recommended_movies" not in st.session_state:
    st.session_state["all_recommended_movies"] = set()

# -------------------------
# 🔐 Login Page
# -------------------------
def login_page():
    st.title("🔐 Login to Talk2Rec")
    username = st.text_input("Username", label_visibility="visible")
    password = st.text_input("Password", type="password", label_visibility="visible")
    
    if st.button("Login"):
        if username and password:
            st.session_state.logged_in = True
            st.session_state.current_user = username
            if username not in st.session_state.user_data:
                st.session_state.user_data[username] = {"past_recommendations": [], "watchlist": []}
            st.session_state["messages"] = []  # Clear messages on new login
            st.session_state["pending_watchlist"] = []  # Reset pending watchlist
            st.rerun()
        else:
            st.error("Invalid credentials! Try again.")
    
    st.stop()

# -------------------------
# 🏠 Main App (After Login)
# -------------------------
if not st.session_state.logged_in:
    login_page()

st.sidebar.title(f"👤 Welcome, {st.session_state.current_user}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state["messages"] = []  # Clear messages on logout
    st.session_state["pending_watchlist"] = []
    st.rerun()

# Get current user data
user = st.session_state.current_user
if user not in st.session_state.user_data:
    st.session_state.user_data[user] = {"past_recommendations": [], "watchlist": []}
user_data = st.session_state.user_data[user]

# 🎭 Sidebar Navigation
page = st.sidebar.radio("Choose Recommendation Type:", ["Chat Mode 🎤", "Past Recommendations 📜", "Watch List ❤️"])

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
        st.session_state["pending_watchlist"] = []
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state["messages"]):
            message(msg["content"], is_user=(msg["role"] == "user"), key=str(i))
    
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
            st.session_state["pending_watchlist"] = recommended_movies.copy()  # Store for watchlist section
            st.session_state["all_recommended_movies"].update(recommended_movies)  # Store all recommended movies
            
            if is_plot_query(query):
                best_match = recommended_movies[0]
                bot_response = f"✅ The best match is: 🎬 {best_match}"
                user_data["past_recommendations"].append({"query": query, "movies": [best_match]})
            else:
                bot_response = "✅ Here are your recommended movies:\n" + "\n".join([f"🎬 {movie}" for movie in recommended_movies])
                user_data["past_recommendations"].append({"query": query, "movies": recommended_movies})
            
            st.session_state["messages"].append({"role": "assistant", "content": bot_response})
            message(bot_response, is_user=False)

    # ✅ Section to add movies to Watchlist (Movies disappear once added)
    if st.session_state["pending_watchlist"]:
        st.subheader("❤️ Add to Watchlist")
        movies_to_remove = []
        for movie in st.session_state["pending_watchlist"]:
            if movie not in user_data["watchlist"]:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"🎬 **{movie}**")
                with col2:
                    if st.button(f"❤️ Add {movie} to Watchlist", key=f"watchlist_chat_{movie}_{user}"):
                        user_data["watchlist"].append(movie)
                        movies_to_remove.append(movie)
                        st.success(f"✅ {movie} added to your watchlist!")
        
        # ✅ Remove movies **AFTER** loop to prevent infinite reruns
        if movies_to_remove:
            st.session_state["pending_watchlist"] = [
                movie for movie in st.session_state["pending_watchlist"] if movie not in movies_to_remove
            ]
            st.rerun()

# -------------------------
# 📜 Past Recommendations
# -------------------------
elif page == "Past Recommendations 📜":
    st.title("📜 Past Recommendations")
    if not user_data["past_recommendations"]:
        st.info("🕰️ No past recommendations available.")
    else:
        for rec in user_data["past_recommendations"]:
            st.markdown(f"### 🎤 Query: *{rec['query']}*")
            for movie in rec["movies"]:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"🎬 **{movie}**")
                with col2:
                    if movie not in user_data["watchlist"]:
                        if st.button("❤️ Add to Watchlist", key=f"watchlist_past_{movie}_{user}"):
                            user_data["watchlist"].append(movie)
                            st.success(f"✅ {movie} added to your watchlist!")
                            st.rerun()
            st.markdown("---")

# -------------------------
# ❤️ Watch List
# -------------------------
elif page == "Watch List ❤️":
    st.title("❤️ Your Watch List")
    if not user_data["watchlist"]:
        st.info("No movies in watchlist yet!")
    else:
        for movie in user_data["watchlist"]:
            st.markdown(f"🎬 **{movie}**")
