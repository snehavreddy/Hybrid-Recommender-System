import pandas as pd

from hybrid_recommender_system.utils.logger import get_logger


DATA_PATH = "data/Music Info.csv"
OUTPUT_PATH = "data/cleaned_data.csv"

logger = get_logger(__name__)


def clean_data(data):
    """Clean the raw music dataset."""

    print("Starting data cleaning...")
    logger.info("Starting data cleaning")

    cleaned_data = (
        data
        .drop_duplicates(subset="spotify_id")
        .drop(columns=["genre", "spotify_id"])
        .fillna({"tags": "no_tags"})
        .assign(
            name=lambda x: x["name"].str.lower(),
            artist=lambda x: x["artist"].str.lower(),
            tags=lambda x: x["tags"].str.lower()
        )
    )

    print(f"Data cleaning completed. Shape: {cleaned_data.shape}")
    logger.info(
        "Data cleaning completed. Shape: %s",
        cleaned_data.shape
    )

    return cleaned_data


def data_for_content_filtering(data):
    """Prepare data for content-based filtering."""

    print("Preparing data for content filtering...")
    logger.info("Preparing data for content filtering")

    return data.drop(
        columns=["track_id", "name", "spotify_preview_url"]
    )


def main(data_path):
    """Load, clean and save the dataset."""

    try:
        print("Starting data cleaning pipeline...")
        logger.info("Starting data cleaning pipeline")

        print(f"Reading data from: {data_path}")
        logger.info("Reading data from: %s", data_path)

        data = pd.read_csv(data_path)

        print(f"Raw data loaded successfully. Shape: {data.shape}")
        logger.info(
            "Raw data loaded successfully. Shape: %s",
            data.shape
        )

        cleaned_data = clean_data(data)

        print(f"Saving cleaned data to: {OUTPUT_PATH}")
        logger.info("Saving cleaned data to: %s", OUTPUT_PATH)

        cleaned_data.to_csv(OUTPUT_PATH, index=False)

        print("Cleaned data saved successfully.")
        logger.info("Cleaned data saved successfully")

        print("Data cleaning pipeline completed.")
        logger.info("Data cleaning pipeline completed")

    except Exception:
        print("Data cleaning pipeline failed.")
        logger.exception("Data cleaning pipeline failed")
        raise


if __name__ == "__main__":
    main(DATA_PATH)