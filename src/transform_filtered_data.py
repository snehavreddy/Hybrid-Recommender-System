import pandas as pd

from data_cleaning import data_for_content_filtering
from content_based_filtering import (
    transform_data,
    save_transformed_data,
)
from hybrid_recommender_system.utils.logger import get_logger


# Logger
logger = get_logger(__name__)


# Path of collaborative-filtering filtered data
FILTERED_DATA_PATH = "data/collab_filtered_data.csv"

# Path for transformed hybrid data
SAVE_PATH = "data/transformed_hybrid_data.npz"


def main(data_path, save_path):
    """
    Prepare collaborative-filtering song data for the hybrid recommender.

    The function loads the filtered collaborative-filtering dataset,
    removes columns that are not required for content-based feature
    transformation, transforms the remaining features using the
    previously trained transformer, and saves the resulting sparse
    matrix.

    Parameters
    ----------
    data_path : str
        Path to the filtered collaborative-filtering CSV file.

    save_path : str
        Path where the transformed hybrid data will be saved.

    Returns
    -------
    None
        The transformed data is saved to the specified output path.
    """

    try:
        print("Starting hybrid data transformation pipeline...")
        logger.info(
            "Starting hybrid data transformation pipeline"
        )

        # Load the filtered data.
        print(f"Reading filtered data from: {data_path}")
        logger.info(
            "Reading filtered data from: %s",
            data_path,
        )

        filtered_data = pd.read_csv(data_path)

        print(
            f"Filtered data loaded successfully. "
            f"Shape: {filtered_data.shape}"
        )
        logger.info(
            "Filtered data loaded successfully. Shape: %s",
            filtered_data.shape,
        )

        # Prepare data for content-based transformation.
        print(
            "Preparing filtered data for content-based transformation..."
        )
        logger.info(
            "Preparing filtered data for content-based transformation"
        )

        filtered_data_cleaned = data_for_content_filtering(
            filtered_data
        )

        print(
            "Filtered data prepared successfully. "
            f"Shape: {filtered_data_cleaned.shape}"
        )
        logger.info(
            "Filtered data prepared successfully. Shape: %s",
            filtered_data_cleaned.shape,
        )

        # Transform the data using the trained transformer.
        print("Transforming hybrid data...")
        logger.info("Transforming hybrid data")

        transformed_data = transform_data(
            filtered_data_cleaned
        )

        print(
            f"Hybrid data transformation completed. "
            f"Shape: {transformed_data.shape}"
        )
        logger.info(
            "Hybrid data transformation completed. Shape: %s",
            transformed_data.shape,
        )

        # Save the transformed data.
        print(
            f"Saving transformed hybrid data to: {save_path}"
        )
        logger.info(
            "Saving transformed hybrid data to: %s",
            save_path,
        )

        save_transformed_data(
            transformed_data,
            save_path,
        )

        print(
            "Transformed hybrid data saved successfully."
        )
        logger.info(
            "Transformed hybrid data saved successfully"
        )

        print(
            "Hybrid data transformation pipeline completed."
        )
        logger.info(
            "Hybrid data transformation pipeline completed"
        )

    except Exception:
        print(
            "Hybrid data transformation pipeline failed."
        )
        logger.exception(
            "Hybrid data transformation pipeline failed"
        )
        raise


if __name__ == "__main__":
    main(
        FILTERED_DATA_PATH,
        SAVE_PATH,
    )