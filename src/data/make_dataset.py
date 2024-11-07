import os
import shutil
from datetime import datetime
from zipfile import ZipFile
import pandas as pd

class DataMover:
    def __init__(self, accidents_data_path, combined_data_path,zip_file_path, base_directory='data/raw'):
        self.accidents_data_path = accidents_data_path
        self.combined_data_path = combined_data_path
        self.zip_file_path=zip_file_path
        self.base_directory = base_directory

    def move_and_rename_data(self):
        """Move the accidents_data.csv to a dated folder and rename it."""
        print("Starting the process to move and rename data.")

        # Get today's date in the desired format
        today = datetime.now()
        date_folder_name = today.strftime("%d_%m_%Y_%H_%M")
        new_folder_path = os.path.join(self.base_directory, date_folder_name)

        # Create the new directory
        os.makedirs(new_folder_path, exist_ok=True)
        print(f"Created directory: {new_folder_path}")

        # Move accidents_data.csv to the new directory and rename it
        new_raw_data_path = os.path.join(new_folder_path, "accidents_data.csv")
        if os.path.exists(self.accidents_data_path):
            try:
                shutil.move(self.accidents_data_path, new_raw_data_path)
                print(f"Moved {self.accidents_data_path} to {new_raw_data_path}")
            except Exception as e:
                print(f"Error moving file: {e}")
        else:
            print(f"Error: The file {self.accidents_data_path} does not exist.")
    
    def process_and_zip_data(self):
        # Read the combined_data.csv
        try:
            print(f"Reading combined data from: {self.combined_data_path}")
            df = pd.read_csv(self.combined_data_path)
            print("Data read successfully")
        except FileNotFoundError:
            print(f"Error: The file {self.combined_data_path} does not exist.")
            return
        except Exception as e:
            print(f"Error reading the file: {e}")
            return
        
        # Save it as accidents_data.csv in the raw directory
        try:
            os.makedirs(os.path.dirname(self.accidents_data_path), exist_ok=True)  # Create directory if not exists
            df.to_csv(self.accidents_data_path, index=False)
            print(f"Saved the data to {self.accidents_data_path}")
        except Exception as e:
            print(f"Error saving the file: {e}")
            return
        
        # Create a zip file of accidents_data.csv
        try:
            with ZipFile(self.zip_file_path, 'w') as zipf:
                zipf.write(accidents_data_path, os.path.basename(accidents_data_path))  # Write the file to the zip
            print(f"Created zip file at: {self.zip_file_path}")
            print("Combining new generated data and the existing dataset process completed successfully.")
        except Exception as e:
            print(f"Error creating the zip file: {e}")
            return
        
if __name__ == "__main__":
    # Set base_dir to the root of the project
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    
    # Define paths using forward slashes for cross-platform compatibility
    accidents_data_path = os.path.join(base_dir, "RoadAccidentsInFrance", "data", "raw", "accidents_data.csv")
    combined_data_path = os.path.join(base_dir, "RoadAccidentsInFrance", "notebooks", "src", "data", "final", "combined_data.csv")

    zip_file_path = os.path.join(base_dir, "RoadAccidentsInFrance", "data", "raw", "accidents_data.zip")
        

    # Create the DataMover instance and run the process
    mover = DataMover(accidents_data_path=accidents_data_path, 
                      combined_data_path=combined_data_path,
                      zip_file_path=zip_file_path)

    # Move the accidents_data.csv to a dated folder and rename it
    mover.move_and_rename_data()
    # Create a zip file from combined_data.csv as accidents_data.csv
    mover.process_and_zip_data()

