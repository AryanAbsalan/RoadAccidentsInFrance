import os
import zipfile
from src.entity import DataIngestionConfig
from custom_logger import logger

import urllib.request as request
from urllib.error import HTTPError
import os
from custom_logger import logger
import zipfile

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def download_file(self):
        try:
            if not os.path.exists(self.config.local_data_file):
                filename, headers = request.urlretrieve(url = self.config.source_url,
                                                    filename = self.config.local_data_file)

                logger.info(f"{filename} download! With following info: \n{headers}")
        
            else:
                logger.info(f"File already exists.")
            
        except HTTPError as e:
            print(f"HTTPError: Could not download the file. URL may be incorrect or unavailable. Error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        # Check if the file exists before attempting to open it
        if not os.path.isfile(self.config.local_data_file):
            print(f"Error: The file {self.config.local_data_file} does not exist.")
            return

        # Check if the file is a valid zip file
        try:
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                # Attempt to read the file to ensure it's a valid ZIP archive
                zip_ref.testzip()  # Will raise an exception if the zip is invalid
        except zipfile.BadZipFile:
            print(f"Error: {self.config.local_data_file} is not a valid zip file.")
            return
        except Exception as e:
            print(f"Error: {str(e)}")
            return

        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)