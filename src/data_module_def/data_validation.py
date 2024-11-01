import os
import pandas as pd
from src.config_manager import DataValidationConfig
from custom_logger import logger


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = None
            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()
        
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status =  True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
            
            logger.info(f">>>>> validation_status {validation_status} completed <<<<<\n\nx=======x")

            # If validation passes, save the validated data
            if validation_status:
                validated_data_path = self.config.validated_data_path
                
                # Ensure the directory exists
                directory = os.path.dirname(validated_data_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)  # Create the directory if it does not exist

                
                data.to_csv(validated_data_path, index=False)  # Save the validated data

                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation status: {validation_status}\n")
            return validation_status
        except Exception as e:
            raise e
