import streamlit as st
from content_based_filtering import recommend
from scipy.sparse import load_npz
import pandas as pd

from hybrid_recommender_system.utils.logger import get_logger


logger = get_logger(__name__)


# Transformed data path
TRANSFORMED_DATA_PATH = "data/transformed_data.npz"

# Cleaned data path
CLEANED_DATA_PATH = "data/cleaned_data.csv"


def load_recommender_data():
    """Load cleaned and transformed recommendation data."""

    try:
        logger.info("Loading cleaned data...")
        print("Loading cleaned data...")

        data = pd.read_csv(CLEANED_DATA_PATH)

        logger.info(f"Cleaned data loaded successfully. Shape: {data.shape}")
        print(f"Cleaned data loaded successfully. Shape: {data.shape}")

        logger.info("Loading transformed data...")
        print("Loading transformed data...")

        transformed_data = load_npz(TRANSFORMED_DATA_PATH)

        logger.info(
            f"Transformed data loaded successfully. Shape: {transformed_data.shape}"
        )
        print(
            f"Transformed data loaded successfully. Shape: {transformed_data.shape}"
        )

        return data, transformed_data

    except Exception:
        logger.exception("Failed to load recommender data")
        print("Failed to load recommender data.")
        raise


def main():

    try:
        logger.info("Starting Streamlit Spotify recommender application")
        print("Starting Streamlit Spotify recommender application")

        data, transformed_data = load_recommender_data()

        # Title
        st.title("Welcome to the Spotify Song Recommender!")

        # Subheader
        st.write(
            "### Enter the name of a song and the recommender "
            "will suggest similar songs 🎵🎧"
        )

        # Text Input
        song_name = st.text_input("Enter a song name:")
        st.write("You entered:", song_name)

        # Lowercase the input
        song_name = song_name.lower()

        # Number of recommendations
        k = st.selectbox(
            "How many recommendations do you want?",
            [5, 10, 15, 20],
            index=1
        )

        # Button
        if st.button("Get Recommendations"):

            logger.info(
                f"Recommendation requested for song='{song_name}', k={k}"
            )
            print(
                f"Recommendation requested for song='{song_name}', k={k}"
            )

            if (data["name"] == song_name).any():

                st.write("Recommendations for", f"**{song_name}**")

                recommendations = recommend(
                    song_name,
                    data,
                    transformed_data,
                    k
                )

                logger.info(
                    f"Generated {len(recommendations)} recommendations "
                    f"for '{song_name}'"
                )
                print(
                    f"Generated {len(recommendations)} recommendations "
                    f"for '{song_name}'"
                )

                # Display Recommendations
                for ind, recommendation in recommendations.iterrows():

                    recommended_song = recommendation["name"].title()
                    artist_name = recommendation["artist"].title()

                    if ind == 0:

                        st.markdown("## Currently Playing")
                        st.markdown(
                            f"#### **{recommended_song}** by **{artist_name}**"
                        )

                        st.audio(
                            recommendation["spotify_preview_url"]
                        )

                        st.write("---")

                    elif ind == 1:

                        st.markdown("### Next Up 🎵")
                        st.markdown(
                            f"#### {ind}. **{recommended_song}** "
                            f"by **{artist_name}**"
                        )

                        st.audio(
                            recommendation["spotify_preview_url"]
                        )

                        st.write("---")

                    else:

                        st.markdown(
                            f"#### {ind}. **{recommended_song}** "
                            f"by **{artist_name}**"
                        )

                        st.audio(
                            recommendation["spotify_preview_url"]
                        )

                        st.write("---")

            else:

                logger.warning(
                    f"Song not found in database: '{song_name}'"
                )
                print(
                    f"Song not found in database: '{song_name}'"
                )

                st.write(
                    f"Sorry, we couldn't find {song_name} in our database. "
                    "Please try another song."
                )

        logger.info("Streamlit Spotify recommender application is ready")

    except Exception:
        logger.exception("Streamlit recommender application failed")
        print("Streamlit recommender application failed.")
        raise


if __name__ == "__main__":
    main()