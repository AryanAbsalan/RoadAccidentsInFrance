✅ Project Name : Road Accidents In France

This project is a starting Pack for MLOps projects based on the subject "Road Accidents In France". 
The Road Accidents In France project is focused on the analysis and prediction of road accidents in France, based on various factors and features related to the accidents.
The project aims to understand the impact of different variables on accident severity and build models to predict accident outcomes.

✅ Project Organization

├── .github/  # GitHub Actions workflows for automation.
│   ├── workflows/  # Contains GitHub Actions workflow files.
│   │   ├── dvc_repro_daily.yml  # Workflow to run the DVC pipeline daily.
│   │   ├── python-app.yml  # Workflow to set up and run Python-based applications.
│   │   ├── run_data_generator_dvc_pipeline.yml  # Workflow to trigger the data generation step in DVC pipeline.
├── custom_logger.py  # Custom logging utility for the project.
├── data/  # Directory containing datasets and data processing files.
│   ├── processed/  # Processed data ready for model training.
│   │   ├── X_test.csv  # Features for testing.
│   │   ├── X_train.csv  # Features for training.
│   │   ├── y_test.csv  # Target variable for testing.
│   │   ├── y_train.csv  # Target variable for training.
│   ├── raw/  # Raw data files directly from the source.
│   │   ├── accidents_data.csv  # Raw accident data in CSV format.
│   │   ├── accidents_data.zip  # Raw accident data in compressed format.
│   ├── status.txt  # A file to log the status of the data pipeline or project.
│   ├── transformed/  # Transformed data ready for analysis.
│   │   ├── accidents_data_transformed.csv  # Transformed version of raw data.
│   ├── validated/  # Data that has passed validation checks.
│   │   ├── accidents_data_validated.csv  # Validated dataset.
├── docker-compose.yml  # Configuration for Docker to set up multi-container applications.
├── Dockerfile  # Dockerfile to build the container image for the project.
├── dvc.lock  # DVC lock file that defines dependencies and pipeline stages.
├── dvc.yaml  # DVC file that defines the data pipeline stages and steps.
├── LICENSE  # Open-source license for the project.
├── logs/  # Folder containing log files generated during the project.
│   ├── logs.log  # Log file storing runtime messages and errors.
├── metrics/  # Folder containing performance metrics related to the model.
│   ├── metrics.json  # JSON file containing model evaluation metrics.
├── models/  # Folder containing machine learning models and preprocessing files.
│   ├── .gitkeep  # Ensures the folder is tracked by Git (empty).
│   ├── DecisionTreeClassifier_model.pkl  # Pickled Decision Tree classifier model.
│   ├── preprocessor.joblib  # Joblib file for data preprocessing steps.
├── notebooks/  # Jupyter notebooks for analysis, exploration, and model development.
│   ├── .gitkeep  # Ensures the folder is tracked by Git.
│   ├── 01_road_accident_france_download_datasets.ipynb  # Notebook to download accident datasets.
│   ├── 02_road_accident_france_preprocess_datasets.ipynb  # Notebook for preprocessing accident datasets.
│   ├── 03_road_accident_france_feature_extraction.ipynb  # Notebook for feature extraction from the data.
│   ├── 04_road_accident_france_modelling.ipynb  # Notebook for building machine learning models.
│   ├── 05_road_accident_france_risk_areas.ipynb  # Notebook for analyzing accident risk areas.
│   ├── src/  # Source code for data processing and output generation within notebooks.
│   │   ├── data/  # Data-related files within the notebook project.
│   │   │   ├── clean/  # Cleaned datasets.
│   │   │   │   ├── data_clean.zip  # Compressed cleaned data.
│   │   │   ├── contour-des-departements.geojson  # Geospatial data for French departments.
│   │   │   ├── final/  # Final processed datasets.
│   │   │   │   ├── data_final.zip  # Compressed final data.
├── README.md  # Main README file containing project description and instructions.
├── references/  # Folder for reference materials such as research papers or links.
│   ├── .gitkeep  # Ensures the folder is tracked by Git.
├── reports/  # Folder for storing final reports and generated figures.
│   ├── .gitkeep  # Ensures the folder is tracked by Git.
│   ├── figures/  # Folder containing figures related to the project.
│   │   ├── .gitkeep  # Ensures the folder is tracked by Git.
│   │   ├── Implementation_Scheme.png  # Diagram illustrating the implementation scheme.
│   ├── RoadAccidentsInFrance_FinalReport.pdf  # Final project report.
├── requirements.txt  # Lists Python dependencies required for the project.
├── src/  # Main application source code and scripts.
│   ├── app/  # Core application files.
│   │   ├── main.py  # Main entry point for the application.
│   │   ├── model_handler.py  # Code for handling model inference and predictions.
│   ├── common_utils.py  # Utility functions used across the project.
│   ├── config/  # Configuration files for the application.
│   │   ├── config.py  # Python configuration file.
│   │   ├── config.yaml  # YAML configuration file.
│   │   ├── config_manager.py  # Script to manage and load configurations.
│   ├── data/  # Data processing scripts.
│   │   ├── data_generator.py  # Script for generating synthetic data.
│   │   ├── make_dataset.py  # Script to create datasets.
│   │   ├── __init__.py  # Makes the folder a Python package.
│   ├── data_module_def/  # Modules for data ingestion, transformation, and validation.
│   │   ├── data_ingestion.py  # Script for data ingestion.
│   │   ├── data_transformation.py  # Script for data transformation.
│   │   ├── data_validation.py  # Script for data validation.
│   │   ├── schema.yaml  # Defines the data schema.
│   │   ├── __init__.py  # Makes the folder a Python package.
│   ├── entity.py  # Defines the data models or classes representing the entities.
│   ├── models/  # Files related to model training and prediction.
│   │   ├── predict_model.py  # Script for model prediction.
│   │   ├── train_model.py  # Script for model training.
│   │   ├── __init__.py  # Makes the folder a Python package.
│   ├── models_module_def/  # Modules for model evaluation and training.
│   │   ├── model_evaluation.py  # Script for model evaluation.
│   │   ├── model_trainer.py  # Script for training models.
│   │   ├── params.yaml  # Parameters for model training and evaluation.
│   │   ├── __init__.py  # Makes the folder a Python package.
│   ├── pipeline_steps/  # Steps in the data science pipeline.
│   │   ├── stage01_data_ingestion.py  # Data ingestion step.
│   │   ├── stage02_data_validation.py  # Data validation step.
│   │   ├── stage03_data_transformation.py  # Data transformation step.
│   │   ├── stage04_model_trainer.py  # Model training step.
│   │   ├── stage05_model_evaluation.py  # Model evaluation step.
│   │   ├── __init__.py  # Makes the folder a Python package.
│   ├── tests/  # Unit tests for the application.
│   │   ├── test_main.py  # Tests for the main application.
│   ├── __init__.py  # Makes the folder a Python package.
├── views/  # Contains the view-related code for rendering outputs or results.
│   ├── views.py  # Handles rendering and visualization.
├── __init__.py  # Marks the directory as a Python package.
├── _alert.rules.yaml  # Configuration files for alerting or monitoring.
├── _alertmanageryaml  # Configuration for alert management system.
├── _docker-compose_pro.yaml  # Docker Compose configuration for the production environment.
├── _docker-compose.yaml  # Alternate Docker Compose configuration for development or other environments.
├── _Dockerfile  # Alternate Dockerfile for the project.
├── _Readme.md  # An alternate or supplementary README file.
├── _requirements.txt  # An alternate or environment-specific requirements file.

--------


Below is an overview of the key components of the dataset, data models, and the relationships used in the project.
✅ Data Model Description:

The core dataset contains several attributes representing different aspects of road accidents, drivers, and the environment. These features are used to predict the severity of accidents and understand the underlying factors contributing to accidents.
Here is a brief explanation of the fields:
Accident Data Fields
    id: A unique identifier for each record in the dataset.
    Driver_Age: Age of the driver involved in the accident (integer).
    Safety_Equipment: An integer representing whether the driver used safety equipment such as seatbelts or helmets.
    Department_Code: A code representing the region (department) in France where the accident occurred.
    Mobile_Obstacle: An indicator for whether there was a moving obstacle involved in the accident (such as another vehicle, animal, etc.).
    Vehicle_Category: A numeric value representing the type of vehicle involved in the accident (e.g., car, motorcycle, bicycle).
    Position_In_Vehicle: Indicates the position of the individual within the vehicle (e.g., driver, front seat, rear seat).
    Collision_Type: A numeric code representing the type of collision (e.g., frontal, rear-end).
    Number_of_Lanes: The number of lanes on the road where the accident occurred.
    Time_of_Day: Encoded time segment during the day (e.g., morning, afternoon, night).
    Journey_Type: The purpose of the trip (e.g., commute, leisure).
    Obstacle_Hit: Indicates whether an obstacle was hit during the accident (e.g., no obstacle, obstacle on the road, or off the road).
    Road_Category: The type of road where the accident occurred (e.g., highway, local road).
    Gender: Gender of the individual involved in the accident (coded as male or female).
    User_Category: Describes the role of the individual involved in the accident (e.g., driver, passenger, pedestrian).
    Intersection_Type: Type of intersection where the accident took place (e.g., roundabout, crossroad).
    Predicted Severity
    Predicted_Severity: The predicted severity of the accident (e.g., minor, severe, fatal). This is the target variable in the prediction model.
    Request Time
    request_time: The timestamp when the prediction was made. This is used to track the timing of the predictions and requests.


✅ SQLModel Definition for User and Prediction
    The project also includes a User table that stores information about the users who interact with the prediction system. This model supports user authentication and authorization, as well as the ability to link a user to the predictions they make.
    Relationship with Prediction Model:
    The UserModel has a one-to-many relationship with the PredictionModel. Each user can have multiple predictions associated with them. This enables tracking of who requested each prediction and maintaining an organized structure for the predictions.


✅ Authentication and Utility Functions
    verify_password
        Utility function to check if the entered plain text password matches the hashed password stored in the database.
        Parameters:
        plain_password: The password provided by the user in plain text.
        hashed_password: The hashed password retrieved from the database.
        Returns:
        True if the passwords match, False otherwise.
    get_password_hash
        Utility function to hash a plain text password before storing it in the database.
        Parameters:
        password: The plain text password to be hashed.
        Returns:
        The hashed version of the password for secure storage.
    get_db
        Dependency function to establish a database session.
        Creates a session with the database, yields it for use in other functions, and ensures the session is properly closed after.
        Useful for functions that need to interact with the database.

Database and User Retrieval Functions
    get_user
        Fetches a user from the database using their username.
        Parameters:
        db: The active database session.
        username: The username of the user to retrieve.
        Returns:
        The UserModel object if the user is found; None if no user matches the username.
    authenticate_user
        Verifies user credentials by checking if the username exists and if the password is correct.
        Parameters:
        db: The active database session.
        username: The username of the user to authenticate.
        password: The plain text password provided by the user.
        Returns:
        The UserModel object if authentication succeeds; False if authentication fails.

Token Generation and Authentication Functions
    create_access_token
        Generates a JWT access token to authenticate and authorize users.
        Parameters:
        data: A dictionary containing user information (e.g., username).
        expires_delta: Optional parameter to set a custom token expiration time. Defaults to 15 minutes if not provided.
        Returns:
        A JWT token string with embedded user information and expiration time.

Dependencies for User Authentication and Validation
    get_current_user
        Retrieves the current user based on the provided JWT token.
        Parameters:
        token: The JWT token passed by the user for authentication.
        db: The active database session.
        Returns:
        The UserModel object representing the current user if the token is valid. Raises an exception if the token is invalid or the user is not found.
    get_current_active_user
        Ensures that the current user is active (not disabled).
        Parameters:
        current_user: The user object obtained from get_current_user.
        Returns:
        The UserModel object if the user is active. Raises an exception if the user is inactive.


✅ API Endpoints

    / (Home Page)
    Displays the homepage content.
    
    /register/ (User Registration)
    Registers a new user by checking for existing usernames and securely storing hashed passwords.
    
    /token (User Login)
    Authenticates a user, generating and returning a JWT access token for access to secure routes.
    
    /users/me/ (Current User Details)
    Returns details of the currently authenticated user, requiring them to be active.
    
    /predict/ (Accident Severity Prediction)
    Generates a prediction for accident severity using input characteristics, saves the prediction with user and timestamp info, and returns the prediction result.
    
    /metrics (Metrics Monitoring)
    Returns application metrics for monitoring in the Prometheus format.

✅ Key Interactions

1. Clone the Repository
```bash
   git clone <git@github.com:AryanAbsalan/RoadAccidentsInFrance.git>
   cd <repository_name>
```

2. Create and Activate a Virtual Environment
   It's recommended to use a virtual environment to manage dependencies.
```bash
   # Create a virtual environment
   python -m venv venv

   # Activate the virtual environment
   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
```

3. Install Dependencies
   Install the project requirements using `requirements.txt`.
```bash
   pip install -r requirements.txt
```

4. Install DVC (Data Version Control) (Optional)
   If your project uses DVC for data management and pipeline versioning:
```bash
   pip install dvc
```

5. Build and Run with Docker (If Applicable)
   If you are using Docker, you can build and run the application in a Docker container:
```bash
   # Build the Docker image
   docker build -t roadaccidentinfrance .

   # Run the Docker container
   docker run -p 8000:8000 roadaccidentinfrance
```

6. Start Services with `docker-compose`
   For multi-container applications, use `docker-compose`.
```bash
   docker-compose up --build
```
    Please concider the collowing comand :

    # Check the DVC status to view pending changes
    dvc status
    # Commit the changes to a specific data file in DVC
    dvc commit data/raw/accidents_data.csv
    # Add the dvc.lock file to git
    git add dvc.lock
    # Commit changes, including updates to dvc.lock
    git commit -m "Updated DVC"
    # Reproduce the DVC pipeline to ensure data and models are up to date
    dvc repro

    ✅ Step 1: Define Configuration Classes 
    We start by writing the configuration objects in `src/entity.py`. 
    These configurations will help in managing the settings and parameters required for each stage in a clean and organized manner. 
    
    Configuration objects:
        * DataIngestionConfig
        * DataValidationConfig
        * DataTransformationConfig
        * ModelTrainerConfig
        * ModelEvaluationConfig

    ✅ Step 2: Configuration Manager 
    We create the class `ConfigurationManager` in `src/config_manager.py` 
    
    This class will:
        * Read paths from `config.yaml`.
        * Read hyperparameters from `params.yaml`.
        * Read the data types from `schema.yaml`.
        * Create configuration objects for each of the stages through the help of the objects defined on the step before: DataIngestionConfig, DataValidationConfig, ModelTrainerConfig and ModelEvaluationConfig.
        * Create necessary folders.

    ⚠️ Pay attention to the `mlflow_uri` on the `get_model_evaluation_config`, make sure you adapt it with your own dagshub credentials. 

    ✅ Step 3: Data module definition and model module definition.
    Files of the  `src/data_module_def` folder, create:

        1. Data Ingestion module 
        This class will:
        * Download the dataset into the appropriate folder.
        * Unzip the dataset into the appropriate folder.

        2. Data Validation module 
        This class will:
            * Validate columns against the schema. Optional: you can also verify the informatic type.
            * Issue a text file saying if the data is valid.

        3. Data Transformation module 
        This class will:
            * Split the data into training and test sets.
            * Save the corresponding csv files into the appropriate folder.

        Similarly, in the corresponding files of the `src/models_module_def` folder, create:

        1. Model trainer module 
        This class will:
            * Train the model using the hyperparameters specified in `params.yaml`.
            * Save the trained model into the appropriate folder.

        2. Model Evaluation module 
        This class will
            * Evaluate the model and log metrics using MLFlow

    ✅ Step 4: Pipeline Steps 
    In `src/pipeline_steps` we create scripts for each stage of the pipeline to instantiate and run the processes:

        * stage01_data_ingestion.py
        * stage02_data_validation.py
        * stage03_data_transformation.py
        * stage04_model_trainer.py
        * stage05_model_evaluation.py

    ✅ Step 5: Use DVC to connect the different stages of your pipeline 
        
    Start by setting DagsHub as your distant storage through DVC.

```bash
    dvc remote add -f origin s3://dvc
    dvc remote modify origin endpointurl https://dagshub.com/your_username/your_repo.s3 
    dvc remote default origin
```
    
    Start by setting DagsHub as your distant storage through DVC.

    For Example:
```bash
        dvc init
        dvc remote add -f origin s3://dvc
        dvc remote modify origin endpointurl https://dagshub.com/aryan.absalan/RoadAccidentsInFrance.s3 
        dvc remote default origin
```

    Use dvc to connect the different steps of your pipeline.

    The command for addind all of the steps of the pipeline are: 

```bash
        dvc stage add -n data_ingestion -d src/pipeline_steps/stage01_data_ingestion.py -d src/config.yaml -o data/raw/accidents_data_red.csv python src/pipeline_steps/stage01_data_ingestion.py

        dvc stage add -n data_validation -d src/pipeline_steps/stage02_data_validation.py -d data/raw/accidents_data_red.csv -o data/validated/accidents_data_validated.csv python src/pipeline_steps/stage02_data_validation.py

        dvc stage add -n data_transformation -d src/pipeline_steps/stage03_data_transformation.py -d data/validated/accidents_data_validated.csv -o data/transformed/accidents_data_transformed.csv python src/pipeline_steps/stage03_data_transformation.py


        dvc stage add -n model_training -d src/pipeline_steps/stage04_model_trainer.py -d data/transformed/accidents_data_transformed.csv -o models/DecisionTreeClassifier_model.pkl python src/pipeline_steps/stage04_model_trainer.py

        dvc stage add -n model_evaluation -d src/pipeline_steps/stage05_model_evaluation.py -d models/DecisionTreeClassifier_model.pkl -d data/validated/accidents_data_validated.csv -o reports/model_evaluation_report.txt python src/pipeline_steps/stage05_model_evaluation.py

```
    You can run the pipeline through the command `dvc repro`.
```bash
        dvc add data/raw/accidents_data.csv
        dvc add data/validated/accidents_data_validated.csv
        dvc add data/transformed/accidents_data_transformed.csv
        dvc add models/DecisionTreeClassifier_model.pkl
        dvc add reports/decision_tree_evaluation_report.txt

        dvc commit data/raw/accidents_data.csv
        dvc push
```

    To track the changes with git, run: git add dvc.lock 'reports\.gitignore'
    To enable auto staging, run: dvc config core.autostage true
    Use `dvc push` to send your updates to remote storage.

    ⚠️ If you're unable to download the dataset from Google Drive due to authentication or security issues,
    you can alternatively download the dataset from the GitHub repository from:

    "https://github.com/AryanAbsalan/RoadAccidentsInFrance/blob/master/notebooks/src/data/final/data_final.zip"
    
    After downloading, rename the file to accidents_data.zip and extract it to the data/raw directory. 
    Ensure that the extracted accidents_data.csv is also in the same directory. 
    Once the files are correctly placed in data/raw/, you can proceed to run the DVC pipeline with the command dvc repro.
    This will allow you to continue with the data processing and model training steps without issues.

7. Running the Application Locally (Without Docker)
   To start the application directly without Docker, use the following command (assuming FastAPI):
```bash
   # Run the app (FastAPI/UVicorn)
   uvicorn src.app.main:app --reload
```

8. Verify Installation with Tests
   Run tests to ensure everything is set up correctly.
```bash
   # Run tests (using pytest)
   pytest src/tests
```

9. Monitor Metrics (Prometheus)
    To access metrics, navigate to the `/metrics` endpoint in your browser or monitoring tool:
```bash
    http://localhost:8000/metrics
```

✅ Docker Compose for Container Management

    Start Containers with docker-compose
        You can start the containers defined in your docker-compose_pro.yml file using the following
```bash 
    docker-compose -f docker-compose_pro.yml up
```
    This command will start all the services defined in the docker-compose_pro.yml file. 
    If you want to run it in detached mode (in the background), add the -d flag:
```bash 
    docker-compose -f docker-compose_pro.yml up -d
```
    Stop Containers with docker-compose
    To stop the containers, use the following command:
```bash 
    docker-compose -f docker-compose_pro.yml down
```

✅ Containers Overview and Interaction

Here's a short explanation of each container and how you can interact with them:

✅ Web Application Container (web)
    Description: This container runs the FastAPI web application that serves as the backend of your project. It handles requests, serves endpoints, and processes business logic.
    You can interact with the API by making HTTP requests to the exposed ports (usually port 8000).
    Example: http://localhost:8000/ to view the homepage or interact with other endpoints.

✅ Database Container (db)
    Description: This container runs a relational database ( MySQL). It stores the application's data.
    You can interact with the database through SQL commands and connected to the container’s exposed database port.

✅ model Container (model)
    The container loads the pre-trained DecisionTree model using the modelhandler class.

✅ Prometheus Container (prometheus)
    Description: Prometheus collects and stores metrics from your containers and services. It helps monitor the performance and health of your application.
    Access the Prometheus UI through http://localhost:9090/ (default).
    Use it to query metrics and analyze the performance of your services.
    Example: View metrics for the web application http://localhost:8000/.

✅ Grafana Container (grafana)
    Description: Grafana visualizes the metrics collected by Prometheus. It helps create dashboards and visual representations of your application’s performance and health.
    Access the Grafana UI via http://localhost:3000/.
    You can connect Grafana to Prometheus to visualize real-time data and monitor your containers and infrastructure.
    Login with default credentials (admin/admin) to get started with creating dashboards.


✅ Thank you...