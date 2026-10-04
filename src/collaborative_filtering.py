import pandas as pd
import dask.dataframe as dd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from hybrid_recommender_system.utils.logger import get_logger


# Logger
logger = get_logger(__name__)


# Output paths
track_ids_save_path = "data/track_ids.npy"
filtered_data_save_path = "data/collab_filtered_data.csv"
interaction_matrix_save_path = "data/interaction_matrix.npz"

# Input paths
songs_data_path = "data/cleaned_data.csv"
user_listening_history_data_path = "data/User Listening History.csv"


def filter_songs_data(
    songs_data: pd.DataFrame,
    track_ids: list,
    save_df_path: str,
) -> pd.DataFrame:
    """
    Filter the songs dataset using the provided track IDs.

    The filtered data is sorted by track ID, its index is reset,
    and the resulting DataFrame is saved as a CSV file.

    Parameters
    ----------
    songs_data : pandas.DataFrame
        Songs dataset containing track information.

    track_ids : list
        List of track IDs that should be retained.

    save_df_path : str
        File path where the filtered dataset will be saved.

    Returns
    -------
    pandas.DataFrame
        Filtered and sorted songs dataset.
    """

    print("Filtering songs data...")
    logger.info("Filtering songs data using track IDs")

    filtered_data = songs_data[
        songs_data["track_id"].isin(track_ids)
    ]

    print(
        f"Filtered songs data. Shape: {filtered_data.shape}"
    )
    logger.info(
        "Filtered songs data. Shape: %s",
        filtered_data.shape,
    )

    print("Sorting songs data by track ID...")
    logger.info("Sorting songs data by track ID")

    filtered_data.sort_values(
        by="track_id",
        inplace=True,
    )

    print("Resetting index...")
    logger.info("Resetting filtered data index")

    filtered_data.reset_index(
        drop=True,
        inplace=True,
    )

    save_pandas_data_to_csv(
        filtered_data,
        save_df_path,
    )

    return filtered_data


def save_pandas_data_to_csv(
    data: pd.DataFrame,
    file_path: str,
) -> None:
    """
    Save a pandas DataFrame to a CSV file.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be saved.

    Returns
    -------
    None
        The DataFrame is saved to the specified file path.
    """

    print(f"Saving data to {file_path}...")
    logger.info(
        "Saving DataFrame to %s",
        file_path,
    )

    data.to_csv(
        file_path,
        index=False,
    )

    print("Data saved successfully.")
    logger.info("Data saved successfully")


def save_sparse_matrix(
    matrix: csr_matrix,
    file_path: str,
) -> None:
    """
    Save a sparse matrix to an NPZ file.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Sparse matrix to be saved.

    Returns
    -------
    None
        The matrix is saved to the specified file path.
    """

    print(f"Saving sparse matrix to {file_path}...")
    logger.info(
        "Saving sparse matrix to %s",
        file_path,
    )

    save_npz(
        file_path,
        matrix,
    )

    print("Sparse matrix saved successfully.")
    logger.info("Sparse matrix saved successfully")


def create_interaction_matrix(
    history_data: dd.DataFrame,
    track_ids_save_path: str,
    save_matrix_path: str,
) -> csr_matrix:
    """
    Create a sparse user-track interaction matrix from listening history.

    The play counts are converted to float values, user and track IDs
    are converted to categorical values, and categorical codes are used
    to create numeric row and column indices for the sparse matrix.

    The track IDs are saved separately so that the matrix indices can
    later be mapped back to the original track IDs.

    Parameters
    ----------
    history_data : dask.dataframe.DataFrame
        User listening history dataset.

    track_ids_save_path : str
        File path where the track ID mapping will be saved.

    save_matrix_path : str
        File path where the interaction matrix will be saved.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse interaction matrix where rows represent tracks,
        columns represent users, and values represent play counts.
    """

    print("Creating interaction matrix...")
    logger.info("Starting interaction matrix creation")

    print("Copying listening history data...")
    logger.info("Copying listening history data")

    df = history_data.copy()

    print("Converting playcount to float...")
    logger.info("Converting playcount column to float64")

    df["playcount"] = df["playcount"].astype(
        np.float64
    )

    print("Converting user and track IDs to categorical values...")
    logger.info(
        "Converting user_id and track_id to categorical values"
    )

    df = df.categorize(
        columns=[
            "user_id",
            "track_id",
        ]
    )

    print("Creating numeric user and track indices...")
    logger.info(
        "Creating numeric indices for users and tracks"
    )

    user_mapping = df["user_id"].cat.codes
    track_mapping = df["track_id"].cat.codes

    print("Extracting track IDs...")
    logger.info("Extracting track ID categories")

    track_ids = df[
        "track_id"
    ].cat.categories.values

    print(
        f"Found {len(track_ids)} unique tracks."
    )
    logger.info(
        "Found %s unique tracks",
        len(track_ids),
    )

    print(
        f"Saving track IDs to {track_ids_save_path}..."
    )
    logger.info(
        "Saving track IDs to %s",
        track_ids_save_path,
    )

    np.save(
        track_ids_save_path,
        track_ids,
        allow_pickle=True,
    )

    print("Track IDs saved successfully.")
    logger.info("Track IDs saved successfully")

    print("Adding numeric indices to the data...")
    logger.info("Adding user_idx and track_idx columns")

    df = df.assign(
        user_idx=user_mapping,
        track_idx=track_mapping,
    )

    print("Aggregating play counts...")
    logger.info(
        "Aggregating play counts by track and user"
    )

    interaction_matrix = (
        df.groupby(
            [
                "track_idx",
                "user_idx",
            ]
        )["playcount"]
        .sum()
        .reset_index()
    )

    print("Computing aggregated interaction data...")
    logger.info("Computing Dask interaction data")

    interaction_matrix = interaction_matrix.compute()

    print(
        "Interaction data computed successfully. "
        f"Shape: {interaction_matrix.shape}"
    )

    logger.info(
        "Interaction data computed successfully. Shape: %s",
        interaction_matrix.shape,
    )

    print("Preparing sparse matrix indices...")
    logger.info("Preparing sparse matrix row and column indices")

    row_indices = interaction_matrix[
        "track_idx"
    ]

    col_indices = interaction_matrix[
        "user_idx"
    ]

    values = interaction_matrix[
        "playcount"
    ]

    n_tracks = row_indices.nunique()
    n_users = col_indices.nunique()

    print(
        f"Interaction matrix dimensions: "
        f"{n_tracks} tracks × {n_users} users"
    )

    logger.info(
        "Interaction matrix dimensions: %s tracks x %s users",
        n_tracks,
        n_users,
    )

    print("Creating sparse interaction matrix...")
    logger.info("Creating CSR interaction matrix")

    interaction_matrix = csr_matrix(
        (
            values,
            (
                row_indices,
                col_indices,
            ),
        ),
        shape=(
            n_tracks,
            n_users,
        ),
    )

    print(
        "Sparse interaction matrix created. "
        f"Shape: {interaction_matrix.shape}"
    )

    logger.info(
        "Sparse interaction matrix created. Shape: %s",
        interaction_matrix.shape,
    )

    save_sparse_matrix(
        interaction_matrix,
        save_matrix_path,
    )

    print("Interaction matrix creation completed.")
    logger.info(
        "Interaction matrix creation completed"
    )

    return interaction_matrix


def collaborative_recommendation(
    song_name,
    artist_name,
    track_ids,
    songs_data,
    interaction_matrix,
    k=5,
):
    """
    Generate collaborative-filtering song recommendations.

    The function finds the requested song using its name and artist,
    retrieves its interaction vector, calculates cosine similarity
    against all tracks, and returns the top k most similar tracks.
    """

    print(
        f"Generating collaborative recommendations for: "
        f"{song_name} by {artist_name}"
    )

    logger.info(
        "Generating collaborative recommendations for: "
        "%s by %s",
        song_name,
        artist_name,
    )

    song_name = song_name.lower()
    artist_name = artist_name.lower()

    print("Finding requested song...")
    logger.info(
        "Searching for song: %s by %s",
        song_name,
        artist_name,
    )

    song_row = songs_data.loc[
        (songs_data["name"] == song_name)
        & (songs_data["artist"] == artist_name)
    ]

    if song_row.empty:
        print(
            f"Song not found: {song_name} by {artist_name}"
        )

        logger.warning(
            "Song not found: %s by %s",
            song_name,
            artist_name,
        )

        return pd.DataFrame()

    input_track_id = song_row[
        "track_id"
    ].values.item()

    logger.info(
        "Input track ID: %s",
        input_track_id,
    )

    print("Finding interaction matrix index...")
    logger.info(
        "Finding matrix index for track ID: %s",
        input_track_id,
    )

    ind = np.where(
        track_ids == input_track_id
    )[0].item()

    print("Fetching input interaction vector...")
    logger.info("Fetching input interaction vector")

    input_array = interaction_matrix[ind]

    print("Calculating similarity scores...")
    logger.info(
        "Calculating collaborative similarity scores"
    )

    similarity_scores = cosine_similarity(
        input_array,
        interaction_matrix,
    )

    print("Similarity scores calculated successfully.")
    logger.info(
        "Similarity scores calculated successfully"
    )

    recommendation_indices = np.argsort(
        similarity_scores.ravel()
    )[-k - 1:][::-1]

    recommendation_track_ids = track_ids[
        recommendation_indices
    ]

    top_scores = np.sort(
        similarity_scores.ravel()
    )[-k - 1:][::-1]

    scores_df = pd.DataFrame(
        {
            "track_id": recommendation_track_ids.tolist(),
            "score": top_scores,
        }
    )

    print("Fetching recommended songs...")
    logger.info(
        "Fetching recommended songs using track IDs"
    )

    top_k_songs = (
        songs_data
        .loc[
            songs_data["track_id"].isin(
                recommendation_track_ids
            )
        ]
        .merge(
            scores_df,
            on="track_id",
        )
        .sort_values(
            by="score",
            ascending=False,
        )
        .drop(
            columns=[
                "track_id",
                "score",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"Generated {len(top_k_songs)} recommendations."
    )

    logger.info(
        "Generated %s collaborative recommendations",
        len(top_k_songs),
    )

    return top_k_songs


def main():
    """
    Run the collaborative-filtering data preparation pipeline.

    The pipeline loads the user listening history, extracts unique
    track IDs, filters the songs dataset using those IDs, and creates
    the sparse track-user interaction matrix.

    Returns
    -------
    None
        The generated datasets and interaction matrix are saved to
        their configured output paths.
    """

    try:
        print(
            "Starting collaborative filtering pipeline..."
        )
        logger.info(
            "Starting collaborative filtering pipeline"
        )

        print(
            f"Loading listening history from: "
            f"{user_listening_history_data_path}"
        )

        logger.info(
            "Loading listening history from: %s",
            user_listening_history_data_path,
        )

        user_data = dd.read_csv(
            user_listening_history_data_path
        )

        print("Listening history loaded successfully.")
        logger.info(
            "Listening history loaded successfully"
        )

        print("Finding unique track IDs...")
        logger.info("Finding unique track IDs")

        unique_track_ids = (
            user_data.loc[:, "track_id"]
            .unique()
            .compute()
        )

        unique_track_ids = unique_track_ids.tolist()

        print(
            f"Found {len(unique_track_ids)} unique track IDs."
        )

        logger.info(
            "Found %s unique track IDs",
            len(unique_track_ids),
        )

        print(
            f"Loading songs data from: "
            f"{songs_data_path}"
        )

        logger.info(
            "Loading songs data from: %s",
            songs_data_path,
        )

        songs_data = pd.read_csv(
            songs_data_path
        )

        print(
            f"Songs data loaded successfully. "
            f"Shape: {songs_data.shape}"
        )

        logger.info(
            "Songs data loaded successfully. Shape: %s",
            songs_data.shape,
        )

        print("Filtering songs data...")
        logger.info("Filtering songs data")

        filtered_songs_data = filter_songs_data(
            songs_data,
            unique_track_ids,
            filtered_data_save_path,
        )

        # Keep only listening-history records for tracks
        # that exist in the cleaned songs dataset.
        #
        # We use an explicit Dask merge here instead of isin()
        # so that the listening history and songs dataset have
        # exactly the same track universe.
        valid_track_ids = filtered_songs_data[
            ["track_id"]
        ]

        user_data = user_data.merge(
            dd.from_pandas(
                valid_track_ids,
                npartitions=1,
            ),
            on="track_id",
            how="inner",
        )

        print(
            "Filtered listening history to tracks "
            "available in cleaned songs data."
        )

        logger.info(
            "Filtered listening history to tracks "
            "available in cleaned songs data."
        )

        print("Creating interaction matrix...")
        logger.info("Creating interaction matrix")

        create_interaction_matrix(
            user_data,
            track_ids_save_path,
            interaction_matrix_save_path,
        )

        print(
            "Collaborative filtering pipeline completed."
        )

        logger.info(
            "Collaborative filtering pipeline completed"
        )

    except Exception:
        print(
            "Collaborative filtering pipeline failed."
        )

        logger.exception(
            "Collaborative filtering pipeline failed"
        )

        raise


if __name__ == "__main__":
    main()