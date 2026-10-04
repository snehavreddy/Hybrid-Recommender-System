import time

import requests


APP_URL = "http://localhost:8000"


def get_app_status(url: str) -> int:
    """
    Send a GET request to the application and return its status code.

    Parameters
    ----------
    url : str
        URL of the application.

    Returns
    -------
    int
        HTTP status code returned by the application.
    """

    response = requests.get(url)
    return response.status_code


def test_app_loading() -> None:
    """
    Test whether the Streamlit application loads successfully.

    The test waits for the application to start, sends a request
    to the application URL, and verifies that the response status
    code is 200.
    """

    print("Waiting for Streamlit application to start...")
    time.sleep(60)

    status_code = get_app_status(APP_URL)

    print(f"Streamlit application returned status code: {status_code}")

    assert status_code == 200, "Unable to load Streamlit App"

    print("Streamlit App Loaded Successfully")