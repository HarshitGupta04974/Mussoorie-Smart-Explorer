import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Mussoorie Vibe Engine",
    layout="wide",
    page_icon="🏔️"
)

# =====================================================
# PREMIUM CSS
# =====================================================
st.markdown("""
<style>
.main { background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; }
.stSidebar { background: linear-gradient(180deg, #141E30, #243B55); }
.premium-card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    margin-bottom: 20px;
}
.recommend-card {
    background: rgba(255,255,255,0.12);
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA LOADING
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("location_Data.csv")

locations_df = load_data()

# =====================================================
# WEIGHTED KNN ENGINE
# =====================================================
class MussoorieVibeEngine:
    def __init__(self, df):
        self.df = df.copy()
        self.features = ['Thrill', 'Seclusion', 'Driving Diff.', 'Crowd Density']

        self.scaler = StandardScaler()
        self.df_scaled = self.df.copy()
        self.df_scaled[self.features] = self.scaler.fit_transform(self.df[self.features])

        # Geographic Sister Hub Mapping
        self.sister_map = {
            "Library": ["Hathipaon", "Barlowganj"],
            "Hathipaon": ["Library"],
            "Picture Palace": ["Landour", "Barlowganj"],
            "Landour": ["Picture Palace", "Dhanaulti"],
            "Dhanaulti": ["Landour"],
            "Barlowganj": ["Library", "Picture Palace"]
        }

    def predict_vibe(self, user_hub, user_vibes, weights):

        # ✅ Use DataFrame to avoid sklearn warning
        user_input = pd.DataFrame(
            [[user_vibes[f] for f in self.features]],
            columns=self.features
        )

        user_scaled = self.scaler.transform(user_input)[0]

        weight_vector = np.array([weights[f] for f in self.features])

        weighted_data = self.df_scaled[self.features].values * weight_vector
        weighted_input = user_scaled * weight_vector

        distances = np.linalg.norm(weighted_data - weighted_input, axis=1)

        result_df = self.df.copy()
        result_df["match_score"] = (1 / (1 + distances)) * 100

        def get_priority(hub):
            if hub == user_hub:
                return 1
            elif hub in self.sister_map.get(user_hub, []):
                return 2
            else:
                return 3

        result_df["priority"] = result_df["Optimized Hub"].apply(get_priority)

        return result_df.sort_values(
            by=["priority", "match_score"],
            ascending=[True, False]
        )

# =====================================================
# RADAR CHART
# =====================================================
def create_radar_chart(row, user_prefs):
    categories = ['Thrill', 'Seclusion', 'Driving Diff.', 'Crowd Density']
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[user_prefs[c] for c in categories],
        theta=categories,
        fill='toself',
        name='Target',
        line_color='rgba(0,180,255,0.7)'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[row[c] for c in categories],
        theta=categories,
        fill='toself',
        name='Location',
        line_color='rgba(255,99,71,0.8)'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,10])),
        showlegend=False,
        height=250,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font_color="white"
    )

    return fig

# =====================================================
# SQLITE
# =====================================================
@st.cache_resource
def get_connection():
    conn = sqlite3.connect("reviews.db", check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT,
            rating INTEGER,
            review TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return conn

conn = get_connection()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🏔️ Mussoorie Engine")
page = st.sidebar.radio("Navigate", [
    "Homepage",
    "Location Discovery",
    "Smart Vibe Engine",
    "Community Reviews"
])

# =====================================================
# HOMEPAGE
# =====================================================
if page == "Homepage":
    st.title("🏔️ Mussoorie Vibe Engine")

    col1, col2 = st.columns(2)
    col1.metric("Total Locations", len(locations_df))
    col2.metric("Reviews", conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0])

    st.markdown("""
    <div class='premium-card'>
        <h3>🌄 Discover Mussoorie Intelligently</h3>
        <p>Geo-aware weighted recommendation system powered by a custom KNN engine.</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# LOCATION DISCOVERY
# =====================================================
elif page == "Location Discovery":
    st.title("📍 Explore by Hub")

    selected_hub = st.selectbox("Select Hub", locations_df["Optimized Hub"].unique())

    hub_data = locations_df[locations_df["Optimized Hub"] == selected_hub]

    for _, row in hub_data.iterrows():
        st.markdown(f"""
        <div class='premium-card'>
            <h3>{row['Name']}</h3>
            <p>⭐ {row['Initial Rating']} / 5</p>
            <p>🏷️ {row['Expert Tags']}</p>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# SMART VIBE ENGINE
# =====================================================
elif page == "Smart Vibe Engine":

    st.title("🌲 Smart Vibe Engine")

    with st.expander("🎯 Customize Your Vibe", expanded=True):
        c1, c2, c3, c4 = st.columns(4)

        u_t = c1.slider("Thrill", 1, 10, 5)
        u_s = c2.slider("Seclusion", 1, 10, 5)
        u_d = c3.slider("Driving Difficulty", 1, 10, 5)
        u_c = c4.slider("Crowd Density", 1, 10, 5)

        w_t = c1.number_input("Thrill Weight", 0.5, 5.0, 2.5)
        w_s = c2.number_input("Seclusion Weight", 0.5, 5.0, 2.0)

    current_hub = st.selectbox("Your Current Hub", locations_df["Optimized Hub"].unique())

    engine = MussoorieVibeEngine(locations_df)

    results = engine.predict_vibe(
        current_hub,
        {
            "Thrill": u_t,
            "Seclusion": u_s,
            "Driving Diff.": u_d,
            "Crowd Density": u_c
        },
        {
            "Thrill": w_t,
            "Seclusion": w_s,
            "Driving Diff.": 1.0,
            "Crowd Density": 1.5
        }
    )

    tiers = [("📍 Right Here", 1),
             ("🚗 Nearby", 2),
             ("🌄 Worth the Trip", 3)]

    cols = st.columns(3)

    for i, (label, priority) in enumerate(tiers):
        with cols[i]:
            st.markdown(f"### {label}")

            tier_data = results[results["priority"] == priority].head(2)

            for _, row in tier_data.iterrows():
                st.markdown(f"""
                <div class='recommend-card'>
                    <h4>{row['Name']} ({row['match_score']:.1f}%)</h4>
                </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(
                    create_radar_chart(
                        row,
                        {
                            "Thrill": u_t,
                            "Seclusion": u_s,
                            "Driving Diff.": u_d,
                            "Crowd Density": u_c
                        }
                    ),
                    width="stretch"
                )

# =====================================================
# COMMUNITY REVIEWS
# =====================================================
elif page == "Community Reviews":

    st.title("⭐ Community Lounge")

    with st.form("review_form"):
        location = st.selectbox("Location", locations_df["Name"].unique())
        rating = st.slider("Rating", 1, 5, 4)
        review = st.text_area("Write Review")

        if st.form_submit_button("Submit"):
            conn.execute(
                "INSERT INTO reviews (location, rating, review) VALUES (?, ?, ?)",
                (location, rating, review)
            )
            conn.commit()
            st.success("Review Submitted!")

    st.markdown("---")

    browse_location = st.selectbox("Browse Reviews", locations_df["Name"].unique())

    reviews = conn.execute(
        "SELECT rating, review, created_at FROM reviews WHERE location=?",
        (browse_location,)
    ).fetchall()

    for r in reversed(reviews):
        st.markdown(f"""
        <div class='premium-card'>
            <h4>⭐ {r[0]} / 5</h4>
            <p>{r[1]}</p>
            <small>{r[2]}</small>
        </div>
        """, unsafe_allow_html=True)