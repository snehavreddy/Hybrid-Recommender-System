import streamlit as st
import pandas as pd

from numpy import load
from scipy.sparse import load_npz

from content_based_filtering import content_recommendation
from hybrid_recommendations import HybridRecommenderSystem
from hybrid_recommender_system.utils.logger import get_logger


# Logger
logger = get_logger(__name__)


# Data paths
CLEANED_DATA_PATH = "data/cleaned_data.csv"
TRANSFORMED_DATA_PATH = "data/transformed_data.npz"
TRACK_IDS_PATH = "data/track_ids.npy"
FILTERED_DATA_PATH = "data/collab_filtered_data.csv"
INTERACTION_MATRIX_PATH = "data/interaction_matrix.npz"
TRANSFORMED_HYBRID_DATA_PATH = "data/transformed_hybrid_data.npz"


def load_recommender_data():
    """
    Load all datasets and transformed matrices required by the
    recommendation application.

    Returns
    -------
    tuple
        A tuple containing songs data, transformed content data,
        track IDs, collaborative-filtering song data, interaction
        matrix, and transformed hybrid data.
    """

    print("Loading recommender data...")
    logger.info("Loading recommender data")

    # Load cleaned song data.
    print(f"Loading cleaned data from: {CLEANED_DATA_PATH}")
    logger.info(
        "Loading cleaned data from: %s",
        CLEANED_DATA_PATH,
    )

    songs_data = pd.read_csv(CLEANED_DATA_PATH)

    # Load content-based transformed data.
    print(
        f"Loading transformed content data from: "
        f"{TRANSFORMED_DATA_PATH}"
    )
    logger.info(
        "Loading transformed content data from: %s",
        TRANSFORMED_DATA_PATH,
    )

    transformed_data = load_npz(
        TRANSFORMED_DATA_PATH
    )

    # Load track IDs.
    print(f"Loading track IDs from: {TRACK_IDS_PATH}")
    logger.info(
        "Loading track IDs from: %s",
        TRACK_IDS_PATH,
    )

    track_ids = load(
        TRACK_IDS_PATH,
        allow_pickle=True,
    )

    # Load collaborative-filtering song data.
    print(
        f"Loading collaborative-filtering data from: "
        f"{FILTERED_DATA_PATH}"
    )
    logger.info(
        "Loading collaborative-filtering data from: %s",
        FILTERED_DATA_PATH,
    )

    filtered_data = pd.read_csv(
        FILTERED_DATA_PATH
    )

    # Load interaction matrix.
    print(
        f"Loading interaction matrix from: "
        f"{INTERACTION_MATRIX_PATH}"
    )
    logger.info(
        "Loading interaction matrix from: %s",
        INTERACTION_MATRIX_PATH,
    )

    interaction_matrix = load_npz(
        INTERACTION_MATRIX_PATH
    )

    # Load transformed hybrid data.
    print(
        f"Loading transformed hybrid data from: "
        f"{TRANSFORMED_HYBRID_DATA_PATH}"
    )
    logger.info(
        "Loading transformed hybrid data from: %s",
        TRANSFORMED_HYBRID_DATA_PATH,
    )

    transformed_hybrid_data = load_npz(
        TRANSFORMED_HYBRID_DATA_PATH
    )

    print("All recommender data loaded successfully.")

    logger.info(
        "All recommender data loaded successfully"
    )

    return (
        songs_data,
        transformed_data,
        track_ids,
        filtered_data,
        interaction_matrix,
        transformed_hybrid_data,
    )


def display_recommendations(recommendations):
    """
    Display recommended songs in the Streamlit application.

    The first recommendation is displayed as the currently
    playing song, while the remaining recommendations are
    displayed as the next songs in the list.

    Parameters
    ----------
    recommendations : pandas.DataFrame
        DataFrame containing recommended songs.

    Returns
    -------
    None
    """

    print(
        f"Displaying {len(recommendations)} recommendations."
    )

    logger.info(
        "Displaying %s recommendations",
        len(recommendations),
    )

    for ind, recommendation in recommendations.iterrows():

        display_song_name = (
            recommendation["name"].title()
        )

        display_artist_name = (
            recommendation["artist"].title()
        )

        if ind == 0:

            st.markdown("## Currently Playing")

            st.markdown(
                f"#### **{display_song_name}** "
                f"by **{display_artist_name}**"
            )

            st.audio(
                recommendation[
                    "spotify_preview_url"
                ]
            )

            st.write("---")

        elif ind == 1:

            st.markdown("### Next Up 🎵")

            st.markdown(
                f"#### {ind}. **{display_song_name}** "
                f"by **{display_artist_name}**"
            )

            st.audio(
                recommendation[
                    "spotify_preview_url"
                ]
            )

            st.write("---")

        else:

            st.markdown(
                f"#### {ind}. **{display_song_name}** "
                f"by **{display_artist_name}**"
            )

            st.audio(
                recommendation[
                    "spotify_preview_url"
                ]
            )

            st.write("---")


def main():
    """
    Run the Streamlit Spotify recommendation application.

    The application supports two recommendation modes:

    1. Content-Based Filtering
       Used when the requested song is not available in the
       collaborative-filtering dataset.

    2. Hybrid Recommender System
       Used when the requested song is available in the
       collaborative-filtering dataset.

    Returns
    -------
    None
    """

    try:
        print(
            "Starting Spotify recommendation application..."
        )

        logger.info(
            "Starting Spotify recommendation application"
        )

        (
            songs_data,
            transformed_data,
            track_ids,
            filtered_data,
            interaction_matrix,
            transformed_hybrid_data,
        ) = load_recommender_data()

        # Title.
        st.title(
            "Welcome to the Spotify Song Recommender!"
        )

        # Subheader.
        st.write(
            "### Enter the name of a song and the "
            "recommender will suggest similar songs 🎵🎧"
        )

        # Song input.
        song_name = st.text_input(
            "Enter a song name:"
        )

        st.write(
            "You entered:",
            song_name,
        )

        # Artist input.
        artist_name = st.text_input(
            "Enter the artist name:"
        )

        st.write(
            "You entered:",
            artist_name,
        )

        # Convert inputs to lowercase.
        song_name = song_name.lower()
        artist_name = artist_name.lower()

        # Number of recommendations.
        k = st.selectbox(
            "How many recommendations do you want?",
            [5, 10, 15, 20],
            index=1,
        )

        # Check whether the song is available for
        # collaborative filtering.
        hybrid_available = (
            (
                filtered_data["name"]
                == song_name
            )
            & (
                filtered_data["artist"]
                == artist_name
            )
        ).any()

        if hybrid_available:

            filtering_type = (
                "Hybrid Recommender System"
            )

            print(
                "Song is available for hybrid "
                "recommendations."
            )

            logger.info(
                "Hybrid recommendation available for "
                "%s by %s",
                song_name,
                artist_name,
            )

            # Diversity slider.
            diversity = st.slider(
                label="Diversity in Recommendations",
                min_value=1,
                max_value=9,
                value=5,
                step=1,
            )

            content_based_weight = (
                1 - (diversity / 10)
            )

            # Display recommendation ratio.
            chart_data = pd.DataFrame(
                {
                    "type": [
                        "Personalized",
                        "Diverse",
                    ],
                    "ratio": [
                        10 - diversity,
                        diversity,
                    ],
                }
            )

            st.bar_chart(
                chart_data,
                x="type",
                y="ratio",
            )

        else:

            filtering_type = (
                "Content-Based Filtering"
            )

            print(
                "Song is not available for hybrid "
                "recommendations. Using content-based filtering."
            )

            logger.info(
                "Using content-based filtering for "
                "%s by %s",
                song_name,
                artist_name,
            )

        # Content-based filtering.
        if (
            filtering_type
            == "Content-Based Filtering"
        ):

            if st.button(
                "Get Recommendations"
            ):

                song_exists = (
                    (
                        songs_data["name"]
                        == song_name
                    )
                    & (
                        songs_data["artist"]
                        == artist_name
                    )
                ).any()

                if song_exists:

                    print(
                        "Generating content-based "
                        "recommendations..."
                    )

                    logger.info(
                        "Generating content-based "
                        "recommendations for %s by %s",
                        song_name,
                        artist_name,
                    )

                    st.write(
                        "Recommendations for",
                        f"**{song_name}** by "
                        f"**{artist_name}**",
                    )

                    recommendations = (
                        content_recommendation(
                            song_name=song_name,
                            artist_name=artist_name,
                            songs_data=songs_data,
                            transformed_data=transformed_data,
                            k=k,
                        )
                    )

                    display_recommendations(
                        recommendations
                    )

                else:

                    logger.warning(
                        "Song not found in database: "
                        "%s by %s",
                        song_name,
                        artist_name,
                    )

                    st.write(
                        f"Sorry, we couldn't find "
                        f"{song_name} in our database. "
                        "Please try another song."
                    )

        # Hybrid recommendation system.
        elif (
            filtering_type
            == "Hybrid Recommender System"
        ):

            if st.button(
                "Get Recommendations"
            ):

                print(
                    "Generating hybrid recommendations..."
                )

                logger.info(
                    "Generating hybrid recommendations "
                    "for %s by %s",
                    song_name,
                    artist_name,
                )

                st.write(
                    "Recommendations for",
                    f"**{song_name}** by "
                    f"**{artist_name}**",
                )

                recommender = (
                    HybridRecommenderSystem(
                        number_of_recommendations=k,
                        weight_content_based=(
                            content_based_weight
                        ),
                    )
                )

                recommendations = (
                    recommender.give_recommendations(
                        song_name=song_name,
                        artist_name=artist_name,
                        songs_data=filtered_data,
                        transformed_matrix=(
                            transformed_hybrid_data
                        ),
                        track_ids=track_ids,
                        interaction_matrix=(
                            interaction_matrix
                        ),
                    )
                )

                display_recommendations(
                    recommendations
                )

                logger.info(
                    "Hybrid recommendations displayed "
                    "successfully"
                )

    except Exception:
        print(
            "Spotify recommendation application failed."
        )

        logger.exception(
            "Spotify recommendation application failed"
        )

        st.error(
            "Something went wrong while generating "
            "recommendations. Please check the logs."
        )

        raise


if __name__ == "__main__":
    main()