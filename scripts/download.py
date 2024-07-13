import argparse
import os
import os.path as op
import warnings

import requests
from loguru import logger
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Unverified HTTPS request")


def download_data(url_file, out_folder):
    # Define the username and password
    if "smplx" in url_file:
        flag = "SMPLX"
    elif "mano" in url_file:
        flag = "MANO"
    #elif "arctic" in url_file:
        #flag = "ARCTIC"
    else:
        flag = "HOLD"

    username = os.environ[f"{flag}_USERNAME"]
    password = os.environ[f"{flag}_PASSWORD"]
    password_fake = "*" * len(password)

    logger.info(f"Username: {username}")
    logger.info(f"Password: {password_fake}")

    post_data = {"username": username, "password": password}
    # Read the URLs from the file
    with open(url_file, "r") as f:
        urls = f.readlines()

    # Strip newline characters from the URLs
    urls = [url.strip() for url in urls]

    # Loop through the URLs and download the files
    logger.info(f"Start downloading from {url_file}")
    pbar = tqdm(urls)
    for url in pbar:
        pbar.set_description(f"Downloading {url[-40:]}")
        # Make a POST request with the username and password
        response = requests.post(
            url,
            data=post_data,
            stream=True,
            verify=False,
            allow_redirects=True,
        )

        if response.status_code == 401:
            logger.warning(
                f"Authentication failed for URLs in {url_file}. Username/password correct?"
            )
            break

        # Get the filename from the URL
        filename = url.split("/")[-1]
        if "models_smplx_v1_1" in url:
            filename = "models_smplx_v1_1.zip"
        elif "mano_v1_2" in url:
            filename = "mano_v1_2.zip"
        elif "image" in url:
            filename = "/".join(url.split("/")[-2:])

        # Write the contents of the response to a file
        out_p = op.join(out_folder, filename)
        os.makedirs(op.dirname(out_p), exist_ok=True)
        with open(out_p, "wb") as f:
            f.write(response.content)

    logger.info("Done")


def main():
    parser = argparse.ArgumentParser(description="Download files from a list of URLs")
    parser.add_argument(
        "--url_file",
        type=str,
        help="Path to file containing list of URLs",
        required=True,
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        help="Path to folder to store downloaded files",
        required=True,
    )
    args = parser.parse_args()
    download_data(args.url_file, args.out_folder)


if __name__ == "__main__":
    main()
