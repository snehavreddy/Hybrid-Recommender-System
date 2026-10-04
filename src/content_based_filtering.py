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


# Paths
CLEANED_DATA_PATH = "data/cleaned_data.csv"
TRANSFORMER_PATH = "transformer.joblib"
TRANSFORMED_DATA_PATH = "data/transformed_data.npz"


# Columns to transform
frequency_encode_cols = ["year"]

ohe_cols = [
    "artist",
    "time_signature",
    "key",
]

tfidf_col = "tags"

standard_scale_cols = [
    "duration_ms",
    "loudness",
    "tempo",
]

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
    """
    Create, fit, and save the feature transformation pipeline.

    The transformer applies different preprocessing techniques to
    different feature types:
    - Count encoding for the year column.
    - One-hot encoding for categorical columns.
    - TF-IDF vectorization for tags.
    - Standard scaling for selected numerical features.
    - Min-max scaling for audio features.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing the features required for transformation.

    Returns
    -------
    None
        The fitted transformer is saved to TRANSFORMER_PATH.
    """

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
    logger.info(
        "Saving transformer to %s",
        TRANSFORMER_PATH,
    )

    joblib.dump(transformer, TRANSFORMER_PATH)

    print("Transformer saved successfully.")
    logger.info("Transformer saved successfully")


def transform_data(data):
    """
    Load the saved transformer and transform the input data.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset to be transformed using the fitted transformer.

    Returns
    -------
    scipy.sparse.spmatrix
        Transformed feature matrix.
    """

    print("Loading transformer...")
    logger.info(
        "Loading transformer from %s",
        TRANSFORMER_PATH,
    )

    transformer = joblib.load(TRANSFORMER_PATH)

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
    """
    Save the transformed feature matrix as a sparse NPZ file.

    Parameters
    ----------
    transformed_data : scipy.sparse.spmatrix
        Sparse matrix containing the transformed feature data.

    save_path : str
        File path where the transformed data should be saved.

    Returns
    -------
    None
        The transformed data is saved to the specified path.
    """

    print(f"Saving transformed data to {save_path}...")
    logger.info(
        "Saving transformed data to %s",
        save_path,
    )

    save_npz(
        save_path,
        transformed_data,
    )

    print("Transformed data saved successfully.")
    logger.info("Transformed data saved successfully")


def calculate_similarity_scores(input_vector, data):
    """
    Calculate cosine similarity between an input song vector
    and all song vectors in the dataset.

    Parameters
    ----------
    input_vector : array-like
        Feature vector representing the selected song.

    data : scipy.sparse.spmatrix
        Transformed feature matrix containing all songs.

    Returns
    -------
    numpy.ndarray
        Array containing the cosine similarity score between
        the input song and every song in the dataset.
    """

    print("Calculating similarity scores...")
    logger.info("Calculating similarity scores")

    similarity_scores = cosine_similarity(
        input_vector,
        data,
    )

    print("Similarity scores calculated successfully.")
    logger.info("Similarity scores calculated successfully")

    return similarity_scores


def recommend(song_name, songs_data, transformed_data, k=10):
    """
    Generate content-based song recommendations.

    The function finds the requested song, obtains its transformed
    feature vector, calculates cosine similarity against all songs,
    and returns the top k most similar songs.

    The input song itself is excluded from the recommendations.

    Parameters
    ----------
    song_name : str
        Name of the song for which recommendations are required.

    songs_data : pandas.DataFrame
        Dataset containing song names, artists, and preview URLs.

    transformed_data : scipy.sparse.spmatrix
        Transformed feature matrix corresponding to songs_data.

    k : int, default=10
        Number of similar songs to recommend.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the recommended song names,
        artists, and Spotify preview URLs.

        If the requested song is not found, an empty DataFrame
        with the expected columns is returned.
    """

    print(f"Generating recommendations for: {song_name}")
    logger.info(
        "Generating recommendations for: %s",
        song_name,
    )

    song_name = song_name.lower().strip()

    song_row = songs_data.loc[
        songs_data["name"] == song_name
    ]

    if song_row.empty:
        print(f"Song not found: {song_name}")

        logger.warning(
            "Song not found: %s",
            song_name,
        )

        return pd.DataFrame(
            columns=[
                "name",
                "artist",
                "spotify_preview_url",
            ]
        )

    song_index = song_row.index[0]

    input_vector = transformed_data[
        song_index
    ].reshape(1, -1)

    similarity_scores = calculate_similarity_scores(
        input_vector,
        transformed_data,
    )

    # Exclude the input song itself.
    top_k_songs_indexes = np.argsort(
        similarity_scores.ravel()
    )[-k - 1:-1][::-1]

    top_k_songs = songs_data.iloc[
        top_k_songs_indexes
    ]

    recommendations = top_k_songs[
        [
            "name",
            "artist",
            "spotify_preview_url",
        ]
    ].reset_index(drop=True)

    print(
        f"Generated {len(recommendations)} recommendations."
    )

    logger.info(
        "Generated %s recommendations for %s",
        len(recommendations),
        song_name,
    )

    return recommendations


def main(data_path, song_name, k=10):
    """
    Run the complete content-based recommendation pipeline.

    The pipeline loads the cleaned dataset, prepares the data
    for content-based filtering, trains the feature transformer,
    transforms the data, saves the transformed features, and
    generates recommendations for the requested song.

    Parameters
    ----------
    data_path : str
        Path to the cleaned music dataset.

    song_name : str
        Name of the song for which recommendations should
        be generated.

    k : int, default=10
        Number of recommendations to generate.

    Returns
    -------
    None
        The pipeline generates and prints recommendations.
    """

    try:
        print(
            "Starting content-based recommendation pipeline..."
        )

        logger.info(
            "Starting content-based recommendation pipeline"
        )

        song_name = song_name.lower().strip()

        print(f"Reading data from: {data_path}")

        logger.info(
            "Reading data from: %s",
            data_path,
        )

        data = pd.read_csv(data_path)

        print(
            f"Data loaded successfully. Shape: {data.shape}"
        )

        logger.info(
            "Data loaded successfully. Shape: %s",
            data.shape,
        )

        print(
            "Preparing data for content filtering..."
        )

        logger.info(
            "Preparing data for content filtering"
        )

        data_content_filtering = (
            data_for_content_filtering(data)
        )

        print(
            "Content filtering data prepared. "
            f"Shape: {data_content_filtering.shape}"
        )

        logger.info(
            "Content filtering data prepared. Shape: %s",
            data_content_filtering.shape,
        )

        train_transformer(
            data_content_filtering
        )

        transformed_data = transform_data(
            data_content_filtering
        )

        save_transformed_data(
            transformed_data,
            TRANSFORMED_DATA_PATH,
        )

        print(f"Finding song: {song_name}")

        logger.info(
            "Finding song: %s",
            song_name,
        )

        recommendations = recommend(
            song_name,
            data,
            transformed_data,
            k,
        )

        if recommendations.empty:
            return

        print("\nTop recommendations:")
        print(recommendations)

        logger.info(
            "Content-based recommendation pipeline completed"
        )

        print(
            "Content-based recommendation pipeline completed."
        )

    except Exception:
        print(
            "Content-based recommendation pipeline failed."
        )

        logger.exception(
            "Content-based recommendation pipeline failed"
        )

        raise


if __name__ == "__main__":
    main(
        CLEANED_DATA_PATH,
        "Hips Don't Lie",
    )