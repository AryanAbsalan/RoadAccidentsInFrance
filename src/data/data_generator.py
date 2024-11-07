import pandas as pd
import numpy as np
import os

class DataProcessor:
    def __init__(self, original_data_path, generated_data_path, output_data_path, size=400000):
        self.original_data_path = original_data_path
        self.generated_data_path = generated_data_path
        self.output_data_path = output_data_path
        self.size = size
        self.columns_to_process = [
            "Severity","Driver_Age", "Safety_Equipment", "Department_Code", "Mobile_Obstacle",
            "Vehicle_Category", "Position_In_Vehicle", "Collision_Type",
            "Number_of_Lanes", "Time_of_Day", "Journey_Type", "Obstacle_Hit",
            "Road_Category", "Gender", "User_Category", "Intersection_Type"
        ]
    
    def load_original_data(self):
        """Load the original dataset."""
        print("Loading the original dataset.")
        self.original_data = pd.read_csv(self.original_data_path)
        print(f"Original dataset loaded with {self.original_data.shape[0]} rows.")
    
    def generate_random_data(self):
        """Generate random data based on unique values in each column."""
        print("Generating random data.")
        random_data = {}

        for column in self.columns_to_process:
            unique_values = self.original_data[column].unique()  # Get unique values
            random_data[column] = np.random.choice(unique_values, size=self.size, replace=True)

        self.generated_data = pd.DataFrame(random_data)
        print(f"Random data generated with {self.generated_data.shape[0]} rows.")
        
        # Save the generated data if needed
        self.generated_data.to_csv(self.generated_data_path, index=False)
        print(f"Generated data saved at {self.generated_data_path}.")
    
    def combine_and_save_datasets(self):
        """Combine the original and generated datasets and save to output path."""
        print("Combining the original and generated datasets.")
        combined_data = pd.concat([self.original_data, self.generated_data], ignore_index=True)
        print(f"Combined dataset created with {combined_data.shape[0]} rows.")

        # Save the combined dataset
        combined_data.to_csv(self.output_data_path, index=False)
        print(f"Combined data saved at {self.output_data_path}.")
    
    def run_pipeline(self):
        """Run all steps of the data processing pipeline."""
        print("Starting data processing pipeline.")
        
        self.load_original_data()
        self.generate_random_data()
        self.combine_and_save_datasets()
        
        print("Data processing pipeline completed.")


# Define main function
if __name__ == "__main__":
    # Set base_dir to the root of the project
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    

    # Paths to your data files, built relative to the current script's directory
    original_data_path = os.path.join(base_dir, "notebooks\\src\\data\\final\\data_final.csv")
    generated_data_path = os.path.join(base_dir, "notebooks\\src\\data\\final\\generated_data.csv")
    output_data_path = os.path.join(base_dir, "notebooks\\src\\data\\final\\combined_data.csv")
    size = 400000
    print(base_dir)
    print(original_data_path)

    # Initialize the DataProcessor class
    processor = DataProcessor(
        original_data_path=original_data_path,
        generated_data_path=generated_data_path,
        output_data_path=output_data_path,
        size=size
    )

    # Run the pipeline
    processor.run_pipeline()


"""
Unique values for the column: 
Driver_Age:
 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 
 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 
 105, 106, 107, 108, 109]

Safety_Equipment: [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43, 90, 91, 92, 93]

Department_Code:
 [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 201, 202, 210, 220, 230, 240, 250, 260, 270, 280,
  290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550,
  560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820,
  830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 971, 972, 973, 974, 976]

Mobile_Obstacle: [0, 1, 2, 4, 5, 6, 9]
Vehicle_Category: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 99]
Position_In_Vehicle: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Collision_Type: [1, 2, 3, 4, 5, 6, 7]
Number_of_Lanes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Time_of_Day: [0, 1, 2, 3]
Journey_Type: [0, 1, 2, 3, 4, 5, 9]
Obstacle_Hit: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
Road_Category: [1, 2, 3, 4, 5, 6, 9]
Gender: [1, 2]
User_Category: [1, 2, 3, 4]
Intersection_Type: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

"""