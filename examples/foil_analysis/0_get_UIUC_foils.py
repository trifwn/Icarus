import os
import re

import requests

from ICARUS import INSTALL_DIR
from ICARUS.database import Database

database_folder = os.path.join(INSTALL_DIR, "Data")
DB = Database(database_folder)
DB2D = DB.DB2D

# Target URL with airfoil links
url = "https://m-selig.ae.illinois.edu/ads/coord_database.html"
# Base URL for the airfoil files
base_url = "https://m-selig.ae.illinois.edu/ads/"

# Download the webpage content
response = requests.get(url)
if response.status_code == 200:
    # Find all lines containing .dat filenames
    lines = response.text.split("\n")
    filenames = []
    for line in lines:
        match = re.search(r'href="(.*?)\.dat"', line)
        if match:
            filenames.append(f"{match.group(1)}.dat")

    for filename in filenames:
        download_url = base_url + filename

        # Get the Airfoil name from the filename
        airfoil_name = filename.split(".")[0].split("/")[-1]
        # Download the file (handle potential errors)
        try:
            response = requests.get(download_url)
            if response.status_code == 200:
                # Remove the .dat extension
                filename = airfoil_name.lower()
                # Save the downloaded data locally with the filename
                dirname = airfoil_name.upper()

                os.makedirs(os.path.join(DB2D, dirname), exist_ok=True)
                filename = os.path.join(DB2D, dirname, filename)
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded: {filename} from {download_url}")
            else:
                print(f"Error downloading {filename} (status: {response.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
else:
    print(f"Failed to retrieve content from {url} (status: {response.status_code})")
