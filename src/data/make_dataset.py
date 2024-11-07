import os
import shutil
from datetime import datetime


class DataMover:
    def __init__(self, raw_data_path, combined_data_path, base_directory='data/raw'):
        self.raw_data_path = raw_data_path
        self.combined_data_path = combined_data_path
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
        try:
            shutil.move(self.raw_data_path, new_raw_data_path)
            print(f"Moved {self.raw_data_path} to {new_raw_data_path}")
        except Exception as e:
            print(f"Error moving file: {e}")

        # Also, move the combined_data.csv to data/raw and rename it
        try:
            shutil.move(self.combined_data_path, os.path.join(self.base_directory, "accidents_data.csv"))
            print(f"Moved {self.combined_data_path} to {os.path.join(self.base_directory, 'accidents_data.csv')}")
        except Exception as e:
            print(f"Error moving combined data file: {e}")

        print("Data move and rename process completed.")

if __name__ == "__main__":
    # Set base_dir to the root of the project
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

    # Define paths based on the project root directory
    raw_data_path = os.path.join(base_dir, "data\\raw\\accidents_data.csv")
    combined_data_path = os.path.join(base_dir, "notebooks\\src\\data\\final\\combined_data.csv")
    print(base_dir)
    print(raw_data_path)
    print(combined_data_path)


    # Create the DataMover instance and run the process
    mover = DataMover(raw_data_path=raw_data_path, combined_data_path=combined_data_path)
    mover.move_and_rename_data()

