import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz

from data_cleaning import data_for_content_filtering
from hybrid_recommender_system.utils.logger import get_logger


# Logger
logger = get_logger(__name__)


# Cleaned Data Path
CLEANED_DATA_PATH = "data/cleaned_data.csv"

# Columns to transform
frequency_encode_cols = ["year"]
ohe_cols = ["artist", "time_signature", "key"]
tfidf_col = "tags"
standard_scale_cols = ["duration_ms", "loudness", "tempo"]
min_max_scale_cols = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]


def train_transformer(data):
    """Create, fit and save the feature transformer."""

    print("Starting transformer training...")
    logger.info("Starting transformer training")

    transformer = ColumnTransformer(
        transformers=[
            (
                "frequency_encode",
                CountEncoder(normalize=True, return_df=True),
                frequency_encode_cols,
            ),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore"),
                ohe_cols,
            ),
            (
                "tfidf",
                TfidfVectorizer(max_features=85),
                tfidf_col,
            ),
            (
                "standard_scale",
                StandardScaler(),
                standard_scale_cols,
            ),
            (
                "min_max_scale",
                MinMaxScaler(),
                min_max_scale_cols,
            ),
        ],
        remainder="passthrough",
        n_jobs=-1,
    )

    print("Fitting transformer...")
    logger.info("Fitting transformer")

    transformer.fit(data)

    print("Transformer fitted successfully.")
    logger.info("Transformer fitted successfully")

    print("Saving transformer...")
    logger.info("Saving transformer to transformer.joblib")

    joblib.dump(transformer, "transformer.joblib")

    print("Transformer saved successfully.")
    logger.info("Transformer saved successfully")


def transform_data(data):
    """Load the transformer and transform the data."""

    print("Loading transformer...")
    logger.info("Loading transformer from transformer.joblib")

    transformer = joblib.load("transformer.joblib")

    print("Transforming data...")
    logger.info("Transforming data")

    transformed_data = transformer.transform(data)

    print(
        f"Data transformation completed. Shape: {transformed_data.shape}"
    )
    logger.info(
        "Data transformation completed. Shape: %s",
        transformed_data.shape,
    )

    return transformed_data


def save_transformed_data(transformed_data, save_path):
    """Save transformed sparse data."""

    print(f"Saving transformed data to {save_path}...")
    logger.info("Saving transformed data to %s", save_path)

    save_npz(save_path, transformed_data)

    print("Transformed data saved successfully.")
    logger.info("Transformed data saved successfully")


def calculate_similarity_scores(input_vector, data):
    """Calculate cosine similarity between input vector and dataset."""

    print("Calculating similarity scores...")
    logger.info("Calculating similarity scores")

    similarity_scores = cosine_similarity(input_vector, data)

    print("Similarity scores calculated successfully.")
    logger.info("Similarity scores calculated successfully")

    return similarity_scores


def recommend(song_name, songs_data, transformed_data, k=10):
    """Generate content-based song recommendations."""

    print(f"Generating recommendations for: {song_name}")
    logger.info("Generating recommendations for: %s", song_name)

    song_name = song_name.lower()

    song_row = songs_data.loc[songs_data["name"] == song_name]

    if song_row.empty:
        print(f"Song not found: {song_name}")
        logger.warning("Song not found: %s", song_name)
        return pd.DataFrame()

    song_index = song_row.index[0]

    input_vector = transformed_data[song_index].reshape(1, -1)

    similarity_scores = calculate_similarity_scores(
        input_vector,
        transformed_data,
    )

    top_k_songs_indexes = np.argsort(
        similarity_scores.ravel()
    )[-k - 1:][::-1]

    top_k_songs_names = songs_data.iloc[top_k_songs_indexes]

    top_k_list = top_k_songs_names[
        ["name", "artist", "spotify_preview_url"]
    ].reset_index(drop=True)

    print(f"Generated {len(top_k_list)} recommendations.")
    logger.info(
        "Generated %s recommendations for %s",
        len(top_k_list),
        song_name,
    )

    return top_k_list


def main(data_path, song_name, k=10):
    """Run the content-based recommendation pipeline."""

    try:
        print("Starting content-based recommendation pipeline...")
        logger.info("Starting content-based recommendation pipeline")

        song_name = song_name.lower()

        print(f"Reading data from: {data_path}")
        logger.info("Reading data from: %s", data_path)

        data = pd.read_csv(data_path)

        print(f"Data loaded successfully. Shape: {data.shape}")
        logger.info("Data loaded successfully. Shape: %s", data.shape)

        print("Preparing data for content filtering...")
        logger.info("Preparing data for content filtering")

        data_content_filtering = data_for_content_filtering(data)

        print(
            "Content filtering data prepared. "
            f"Shape: {data_content_filtering.shape}"
        )
        logger.info(
            "Content filtering data prepared. Shape: %s",
            data_content_filtering.shape,
        )

        train_transformer(data_content_filtering)

        transformed_data = transform_data(
            data_content_filtering
        )

        save_transformed_data(
            transformed_data,
            "data/transformed_data.npz",
        )

        print(f"Finding song: {song_name}")
        logger.info("Finding song: %s", song_name)

        song_row = data.loc[data["name"] == song_name]

        if song_row.empty:
            print(f"Song not found: {song_name}")
            logger.warning("Song not found: %s", song_name)
            return

        song_index = song_row.index[0]

        input_vector = transformed_data[
            song_index
        ].reshape(1, -1)

        similarity_scores = calculate_similarity_scores(
            input_vector,
            transformed_data,
        )

        top_k_songs_indexes = np.argsort(
            similarity_scores.ravel()
        )[-k - 1:-1][::-1]

        top_k_songs_names = data.iloc[
            top_k_songs_indexes
        ]

        print("\nTop recommendations:")
        print(top_k_songs_names)

        logger.info(
            "Generated %s recommendations for %s",
            len(top_k_songs_names),
            song_name,
        )

        print("Content-based recommendation pipeline completed.")
        logger.info(
            "Content-based recommendation pipeline completed"
        )

    except Exception:
        print("Content-based recommendation pipeline failed.")
        logger.exception(
            "Content-based recommendation pipeline failed"
        )
        raise


if __name__ == "__main__":
    main(CLEANED_DATA_PATH, "Hips Don't Lie")