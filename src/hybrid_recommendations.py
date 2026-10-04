import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

from hybrid_recommender_system.utils.logger import get_logger


logger = get_logger(__name__)


class HybridRecommenderSystem:
    """
    Hybrid recommendation system combining content-based and
    collaborative-filtering similarity scores.

    The final recommendation score is calculated using a weighted
    combination of content-based and collaborative-filtering scores.

    Parameters
    ----------
    number_of_recommendations : int
        Number of songs to return as recommendations.

    weight_content_based : float
        Weight assigned to content-based similarity scores.
        The collaborative-filtering weight is calculated as
        1 - weight_content_based.
    """

    def __init__(
        self,
        number_of_recommendations: int,
        weight_content_based: float,
    ):
        """
        Initialize the hybrid recommender system.

        Parameters
        ----------
        number_of_recommendations : int
            Number of recommendations to generate.

        weight_content_based : float
            Weight assigned to the content-based recommender.
            Must be between 0 and 1.
        """

        print("Initializing hybrid recommender system...")
        logger.info(
            "Initializing hybrid recommender system"
        )

        self.number_of_recommendations = (
            number_of_recommendations
        )

        self.weight_content_based = (
            weight_content_based
        )

        self.weight_collaborative = (
            1 - weight_content_based
        )

        print(
            "Hybrid recommender initialized successfully."
        )

        logger.info(
            "Hybrid recommender initialized. "
            "Content weight: %s, Collaborative weight: %s",
            self.weight_content_based,
            self.weight_collaborative,
        )

    def __calculate_content_based_similarities(
        self,
        song_name,
        artist_name,
        songs_data,
        transformed_matrix,
    ):
        """
        Calculate content-based similarity scores for a song.

        The selected song's transformed feature vector is compared
        with all song feature vectors using cosine similarity.

        Parameters
        ----------
        song_name : str
            Name of the input song.

        artist_name : str
            Artist of the input song.

        songs_data : pandas.DataFrame
            Song metadata containing song names and artists.

        transformed_matrix : scipy.sparse matrix
            Transformed content-based feature matrix.

        Returns
        -------
        numpy.ndarray
            Cosine similarity scores between the input song and
            all songs.
        """

        print(
            f"Calculating content-based similarities for "
            f"{song_name} by {artist_name}..."
        )

        logger.info(
            "Calculating content-based similarities for %s by %s",
            song_name,
            artist_name,
        )

        song_row = songs_data.loc[
            (songs_data["name"] == song_name)
            & (songs_data["artist"] == artist_name)
        ]

        if song_row.empty:
            logger.warning(
                "Song not found for content-based similarity: "
                "%s by %s",
                song_name,
                artist_name,
            )

            raise ValueError(
                f"Song not found: {song_name} by {artist_name}"
            )

        song_index = song_row.index[0]

        input_vector = transformed_matrix[
            song_index
        ].reshape(1, -1)

        content_similarity_scores = cosine_similarity(
            input_vector,
            transformed_matrix,
        )

        print(
            "Content-based similarity calculation completed."
        )

        logger.info(
            "Content-based similarity calculation completed"
        )

        return content_similarity_scores

    def __calculate_collaborative_filtering_similarities(
        self,
        song_name,
        artist_name,
        track_ids,
        songs_data,
        interaction_matrix,
    ):
        """
        Calculate collaborative-filtering similarity scores for a song.

        The input song is mapped to its track ID and corresponding
        interaction-matrix row. Cosine similarity is then calculated
        against all tracks.

        Parameters
        ----------
        song_name : str
            Name of the input song.

        artist_name : str
            Artist of the input song.

        track_ids : numpy.ndarray
            Array mapping interaction-matrix indices to track IDs.

        songs_data : pandas.DataFrame
            Song metadata containing song names, artists, and track IDs.

        interaction_matrix : scipy.sparse matrix
            Track-user interaction matrix.

        Returns
        -------
        numpy.ndarray
            Cosine similarity scores between the input song and
            all songs.
        """

        print(
            "Calculating collaborative-filtering similarities..."
        )

        logger.info(
            "Calculating collaborative-filtering similarities "
            "for %s by %s",
            song_name,
            artist_name,
        )

        song_row = songs_data.loc[
            (songs_data["name"] == song_name)
            & (songs_data["artist"] == artist_name)
        ]

        if song_row.empty:
            logger.warning(
                "Song not found for collaborative filtering: "
                "%s by %s",
                song_name,
                artist_name,
            )

            raise ValueError(
                f"Song not found: {song_name} by {artist_name}"
            )

        input_track_id = song_row[
            "track_id"
        ].values.item()

        logger.info(
            "Input track ID: %s",
            input_track_id,
        )

        matching_indices = np.where(
            track_ids == input_track_id
        )[0]

        if len(matching_indices) == 0:
            logger.warning(
                "Track ID %s not found in interaction matrix",
                input_track_id,
            )

            raise ValueError(
                f"Track ID not found in interaction matrix: "
                f"{input_track_id}"
            )

        ind = matching_indices[0]

        input_array = interaction_matrix[ind]

        collaborative_similarity_scores = (
            cosine_similarity(
                input_array,
                interaction_matrix,
            )
        )

        print(
            "Collaborative-filtering similarity "
            "calculation completed."
        )

        logger.info(
            "Collaborative-filtering similarity calculation "
            "completed"
        )

        return collaborative_similarity_scores

    def __normalize_similarities(
        self,
        similarity_scores,
    ):
        """
        Normalize similarity scores using min-max normalization.

        The scores are transformed to a range between 0 and 1.

        Parameters
        ----------
        similarity_scores : numpy.ndarray
            Similarity scores to normalize.

        Returns
        -------
        numpy.ndarray
            Normalized similarity scores.
        """

        print("Normalizing similarity scores...")
        logger.info("Normalizing similarity scores")

        minimum = np.min(similarity_scores)
        maximum = np.max(similarity_scores)

        if maximum == minimum:
            logger.warning(
                "Similarity scores have identical minimum "
                "and maximum values"
            )

            return np.zeros_like(
                similarity_scores
            )

        normalized_scores = (
            (similarity_scores - minimum)
            / (maximum - minimum)
        )

        print(
            "Similarity score normalization completed."
        )

        logger.info(
            "Similarity score normalization completed"
        )

        return normalized_scores

    def __weighted_combination(
        self,
        content_based_scores,
        collaborative_filtering_scores,
    ):
        """
        Combine content-based and collaborative-filtering scores.

        The two normalized score sets are combined according to
        the configured weights.

        Parameters
        ----------
        content_based_scores : numpy.ndarray
            Normalized content-based similarity scores.

        collaborative_filtering_scores : numpy.ndarray
            Normalized collaborative-filtering similarity scores.

        Returns
        -------
        numpy.ndarray
            Weighted hybrid recommendation scores.
        """

        print(
            "Combining content-based and "
            "collaborative-filtering scores..."
        )

        logger.info(
            "Combining similarity scores using weights. "
            "Content: %s, Collaborative: %s",
            self.weight_content_based,
            self.weight_collaborative,
        )

        weighted_scores = (
            self.weight_content_based
            * content_based_scores
        ) + (
            self.weight_collaborative
            * collaborative_filtering_scores
        )

        print(
            "Weighted score combination completed."
        )

        logger.info(
            "Weighted score combination completed"
        )

        return weighted_scores

    def give_recommendations(
        self,
        song_name,
        artist_name,
        songs_data,
        track_ids,
        transformed_matrix,
        interaction_matrix,
    ):
        """
        Generate recommendations using the hybrid approach.

        Content-based and collaborative-filtering similarities are
        calculated independently, normalized, and combined using
        the configured weights. The songs with the highest final
        weighted scores are returned.

        Parameters
        ----------
        song_name : str
            Name of the song for which recommendations are required.

        artist_name : str
            Artist of the input song.

        songs_data : pandas.DataFrame
            Song metadata containing song names, artists, and track IDs.

        track_ids : numpy.ndarray
            Track IDs corresponding to interaction-matrix rows.

        transformed_matrix : scipy.sparse matrix
            Transformed content-based feature matrix.

        interaction_matrix : scipy.sparse matrix
            Track-user interaction matrix.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the recommended songs.
        """

        try:
            print(
                f"Generating hybrid recommendations for "
                f"{song_name} by {artist_name}..."
            )

            logger.info(
                "Generating hybrid recommendations for %s by %s",
                song_name,
                artist_name,
            )

            # Calculate content-based similarities.
            content_based_similarities = (
                self.__calculate_content_based_similarities(
                    song_name=song_name,
                    artist_name=artist_name,
                    songs_data=songs_data,
                    transformed_matrix=transformed_matrix,
                )
            )

            # Calculate collaborative-filtering similarities.
            collaborative_filtering_similarities = (
                self.__calculate_collaborative_filtering_similarities(
                    song_name=song_name,
                    artist_name=artist_name,
                    track_ids=track_ids,
                    songs_data=songs_data,
                    interaction_matrix=interaction_matrix,
                )
            )

            # Normalize content-based similarities.
            normalized_content_based_similarities = (
                self.__normalize_similarities(
                    content_based_similarities
                )
            )

            # Normalize collaborative-filtering similarities.
            normalized_collaborative_filtering_similarities = (
                self.__normalize_similarities(
                    collaborative_filtering_similarities
                )
            )

            # Combine the normalized scores.
            weighted_scores = (
                self.__weighted_combination(
                    content_based_scores=(
                        normalized_content_based_similarities
                    ),
                    collaborative_filtering_scores=(
                        normalized_collaborative_filtering_similarities
                    ),
                )
            )

            print(
                "Finding top hybrid recommendations..."
            )

            logger.info(
                "Finding top %s hybrid recommendations",
                self.number_of_recommendations,
            )

            # Get recommendation indices.
            recommendation_indices = np.argsort(
                weighted_scores.ravel()
            )[
                -self.number_of_recommendations - 1:
            ][::-1]

            # Get recommendation track IDs.
            recommendation_track_ids = track_ids[
                recommendation_indices
            ]

            # Get recommendation scores.
            top_scores = np.sort(
                weighted_scores.ravel()
            )[
                -self.number_of_recommendations - 1:
            ][::-1]

            scores_df = pd.DataFrame(
                {
                    "track_id": (
                        recommendation_track_ids.tolist()
                    ),
                    "score": top_scores,
                }
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
                f"Generated {len(top_k_songs)} "
                "hybrid recommendations."
            )

            logger.info(
                "Generated %s hybrid recommendations for %s by %s",
                len(top_k_songs),
                song_name,
                artist_name,
            )

            return top_k_songs

        except Exception:
            print(
                "Hybrid recommendation generation failed."
            )

            logger.exception(
                "Hybrid recommendation generation failed "
                "for %s by %s",
                song_name,
                artist_name,
            )

            raise