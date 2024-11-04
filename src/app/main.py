import os
from datetime import datetime, timedelta, timezone
from typing import List, Annotated, Optional

from fastapi.responses import HTMLResponse
import jwt

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext

from sqlmodel import SQLModel, Field, create_engine, Session, select,Relationship
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from pydantic import BaseModel
from src.app.model_handler import ModelHandler
import joblib
from dotenv import load_dotenv

from views.views import homepage 

# Load environment variables from .env file
load_dotenv()

# Read model path from environment variables and initialize model_handler
model_path = os.getenv("MODEL_PATH", "models/decision_tree_model.joblib")
preprocessor_path = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.joblib")

# Load the model and preprocessor
model_handler = ModelHandler()
model_handler.model = joblib.load(model_path )
model_handler.preprocessor = joblib.load(preprocessor_path)

# Read database configuration from environment variables
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "rootpass")
MYSQL_HOST = os.getenv("MYSQL_HOST", "db") # for testing with docker container
# MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost") # for testing locally
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "roadaccidentsinfrance")

# Testing correct loading of .env
print(f"MYSQL_HOST after loading .env: {MYSQL_HOST}")  # Add this line

# Construct the database URL
DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
# Create an SQLAlchemy engine using the constructed DATABASE_URL
engine = create_engine(DATABASE_URL,echo=True) # Set echo=True to log all SQL statement
# Create a session factory for interacting with the database
SessionLocal = sessionmaker(autocommit=False,autoflush=False, bind=engine)


SECRET_KEY = "799a3869fefa62efea1e0bdbc49e38b32f64f3aca92c7e95bfd9db7f66d00d8d"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
# Set up password hashing with passlib
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# OAuth2 configuration
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Description for the API
description = """
Road Accidents In France API helps you predict the severity of road accidents based on various parameters. 

You can **predict the severity of an accident** by providing the necessary parameters.

#### Request Body
- **Driver_Age**: Age of the driver.
- **Safety_Equipment**: Level of safety equipment used.
- **Department_Code**: Code of the department involved.
- **Mobile_Obstacle**: Presence of mobile obstacles.
- **Vehicle_Category**: Category of the vehicle.
- **Position_In_Vehicle**: Position of the driver in the vehicle.
- **Collision_Type**: Type of collision.
- **Number_of_Lanes**: Number of lanes involved in the accident.
- **Time_of_Day**: Time of day when the accident occurred.
- **Journey_Type**: Type of journey (e.g., work, leisure).
- **Obstacle_Hit**: Whether an obstacle was hit.
- **Road_Category**: Category of the road.
- **Gender**: Gender of the driver.
- **User_Category**: Category of the user.
- **Intersection_Type**: Type of intersection.

### Example Usage
```json
POST /predict/
{
    "Driver_Age": 33,
    "Safety_Equipment": 10,
    "Department_Code": 590,
    "Mobile_Obstacle": 2,
    "Vehicle_Category": 3,
    "Position_In_Vehicle": 1,
    "Collision_Type": 2,
    "Number_of_Lanes": 4,
    "Time_of_Day": 1,
    "Journey_Type": 1,
    "Obstacle_Hit": 1,
    "Road_Category": 2,
    "Gender": 1,
    "User_Category": 2,
    "Intersection_Type": 1
}
"""

# FastAPI instance
app = FastAPI(
    title="Road Accidents In France API",
    description=description,
    summary="Predict the severity of an accident",
    version="0.0.1",
    contact={
        "name": "Aryan Absalan",
        "github": "https://github.com/AryanAbsalan",
        "email": "Aryan.absalan@gmail.com",

        "name": "Jannis Zeelen",
        "github": "https://github.com/JannisZeelen",
        "email": "Jannis.Zeelen@gmail.com",

        "name": "Tanja Schroeder",
        "github": "https://github.com/tanjaldir",
        "email": "",
    },
    license_info={
        "name": "Apache 2.0 - MIT License",
        "identifier": "MIT",
    },
)


# Model for creating tokens
class Token(BaseModel):
    access_token: str
    token_type: str

# Model for holding token data
class TokenData(BaseModel):
    username: Optional[str] = None

# Define the Users table using SQLModel
class UserModel(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True, nullable=False)
    hashed_password: str
    disabled: Optional[bool] = Field(default=False)
    # Relationship with PredictionModel
    predictions: List["PredictionModel"] = Relationship(back_populates="user")

# Model for creating a new user (without the ID)
class UserCreate(BaseModel):
    username: str
    hashed_password: str
    disabled: Optional[bool] = False

# Define the 'predictions' table
class PredictionModel(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    Driver_Age: int  # Age of the driver involved in the accident.
    Safety_Equipment: int  # Representing safety equipment used (e.g., seatbelt, helmet).
    Department_Code: int  # Indicating the regional department where the accident occurred.
    Mobile_Obstacle: int  # Indicate a moving obstacle like another vehicle, animal, etc.
    Vehicle_Category: int  # Representing the type of vehicle (e.g., car, motorcycle, bicycle).
    Position_In_Vehicle: int  # Refers to the position of the individual in the vehicle (driver, front seat, rear seat).
    Collision_Type: int  # Indicating the type of collision (e.g., frontal, rear-end).
    Number_of_Lanes: int  # Number of lanes on the road where the accident occurred.
    Time_of_Day: int  # Encoded time segment during the day (e.g., morning, afternoon, night).
    Journey_Type: int  # Representing the purpose of the trip (e.g., commute, leisure).
    Obstacle_Hit: int  # Obstacle hit by the vehicle (no obstacle, obstacle on the road, obstacle off the road).
    Road_Category: int  # Indicating the type of road (e.g., highway, local road).
    Gender: int  # Gender of the individual (male and female).
    User_Category: int  # Describing the role of the user involved (e.g., driver, passenger, pedestrian).
    Intersection_Type: int  # Type of intersection where the accident took place (e.g., roundabout, crossroad).
    Predicted_Severity: int  # Representing the predicted severity of the accident (e.g., minor, severe, fatal).
    request_time: datetime = Field(default_factory=datetime.utcnow)

    # Foreign key relationship to UserModel
    user_id: Optional[int] = Field(default=None, foreign_key="usermodel.id")
    user: Optional["UserModel"] = Relationship(back_populates="predictions")

# Pydantic model for request validation
class PredictionInput(BaseModel):
    Driver_Age:int
    Safety_Equipment:int
    Department_Code:int 
    Mobile_Obstacle:int
    Vehicle_Category:int
    Position_In_Vehicle:int
    Collision_Type:int 
    Number_of_Lanes:int 
    Time_of_Day:int 
    Journey_Type:int 
    Obstacle_Hit:int 
    Road_Category:int 
    Gender: int
    User_Category:int
    Intersection_Type:int

# Utility function to verify password
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# Utility function to hash the password
def get_password_hash(password):
    return pwd_context.hash(password)

# Dependency: Create a database session
def get_db():
    with Session(engine) as session:
        yield session

# Function to fetch a user from the database
def get_user(db: Session, username: str):
    statement = select(UserModel).where(UserModel.username == username)
    result = db.execute(statement)  
    user = result.scalar_one_or_none()  # Fetch a single user or None
    return user

# Function to authenticate user by checking the password
def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

# Function to create a JWT access token
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Dependency: Retrieve the current user using the token
async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        user = get_user(db, username=username)
        if user is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    return user

# Dependency: Ensure the user is active (not disabled)
async def get_current_active_user(current_user: Annotated[UserModel, Depends(get_current_user)]):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.on_event("startup")
def startup():
    """
    Initializes the application by ensuring the required database and tables exist.
    Steps:
        1. Connect to MySQL server without specifying the database to check if it exists.
        2. If the target database does not exist, create it.
        3. Verify the connection to the target database.
        4. Create all required tables as defined in the SQLModel models.
        5. Insert a default admin user if not already present.
    """
    # Connect to MySQL server without specifying a database first (so we can create the database if needed)
    engine_without_db = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}", echo=True)

    with engine_without_db.connect() as conn:
        # Create the database if it doesn't exist
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}`;"))
        print(f"\t **** Database '{MYSQL_DATABASE}' is available.")
    
    # Now, modify engine to use the database
    engine_with_db = create_engine(DATABASE_URL, echo=True)

    # Test if the connection to the roadaccidentsinfrance database was works
    try:
        with engine_with_db.connect() as connection:
            print(f"\t **** Connection to '{MYSQL_DATABASE}' database was successful!")
    except Exception as e:
        print("Connection failed:", e)

    # Create all tables defined by SQLModel if they do not exist
    SQLModel.metadata.create_all(bind=engine_with_db)  # This will create the users table if it doesn't exist

    # Insert admin user
    admin_user = UserModel(
        username="admin",
        hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # Hashed password for "secret"
        disabled=False
    )
    
    with Session(engine) as session:
        # Check if the user already exists
        existing_user = session.execute(select(UserModel).where(UserModel.username == admin_user.username)).first()
        if existing_user is None:
            # Add the admin user to the session and commit
            session.add(admin_user)
            session.commit()
            print(f"Inserted admin user: {admin_user.username}")
        else:
            print(f"User {admin_user.username} already exists.")

# Define the root endpoint
@app.get("/", response_class=HTMLResponse)
async def home():
    return homepage()
# Route to register a new user
@app.post("/register/", response_model=UserCreate)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """
    Registers a new user in the system.
    Steps:
        1. Check if a user with the given username already exists in the database.
            - If a user is found, raises a 400 Bad Request error with a message indicating that the username is already registered.
        2. Create a new `UserModel` instance from the provided input data.
            - The password is securely hashed before being stored in the database.
        3. Add the new user to the database session and commit the transaction to save the record.
            - This assigns a unique ID to the user and saves it permanently in the database.
        4. Refresh the user instance to include any database-generated values, such as the unique ID.
        5. Return the created user instance.
    """
    # Check if the user already exists
    db_user = get_user(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Create a UserModel instance from the user input
    user = UserModel(
        username=user.username,
        hashed_password= get_password_hash(user.hashed_password),
        disabled=user.disabled
    )

    # Create the user in the database
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

# Route to login and generate a JWT token
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: Session = Depends(get_db)):
    """
    Authenticates a user and provides a JWT access token upon successful login.
    Steps:
        1. Authenticate the user using provided credentials (username and password).
            - If authentication fails, raises a 401 Unauthorized error.
        2. Sets the expiration time for the access token.
        3. Creates a JWT token containing the username as the subject.
        4. Returns the token with a "bearer" type for use in authorized routes.
    """
    # Authenticate the user using provided credentials (username and password)
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Sets the expiration time for the access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    # Creates a JWT token containing the username as the subject
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    # Returns the token with a "bearer" type for use in authorized routes
    return Token(access_token=access_token, token_type="bearer")

# Route to get the current logged-in user's details
@app.get("/users/me/", response_model=UserModel)
async def read_users_me(current_user: Annotated[UserModel, Depends(get_current_active_user)]):
    """ Retrieves the details of the currently authenticated user. """
    return current_user

@app.post("/predict/")
def get_prediction(current_user: Annotated[UserModel, Depends(get_current_active_user)],  # Ensure the user is logged in
                   data: PredictionInput, 
                   db: Session = Depends(get_db)):
    """
    Generates a prediction for accident severity based on user input and saves it to the database.
    Steps:
        1. Validates that the user is logged in using the `get_current_active_user` dependency.
        2. Extracts input data from the `PredictionInput` model, which includes various characteristics of the accident scenario.
        3. Calls a prediction function to generate a predicted severity based on the input data.
            - This should be replaced with an actual prediction model in a production environment.
        4. Creates a new entry in the `PredictionModel` with the input data, predicted severity, the current timestamp, and the ID of the logged-in user.
        5. Adds the new prediction record to the database and commits the transaction.
        6. Refreshes the entry to obtain the updated object, including the generated ID.
        7. Returns a JSON response containing the predicted severity, the record ID, and the username of the current user.
    """
    # Extract input values
    features = [
        data.Driver_Age,
        data.Safety_Equipment,
        data.Department_Code,
        data.Mobile_Obstacle,
        data.Vehicle_Category,
        data.Position_In_Vehicle,
        data.Collision_Type,
        data.Number_of_Lanes,
        data.Time_of_Day,
        data.Journey_Type,
        data.Obstacle_Hit,
        data.Road_Category,
        data.Gender,
        data.User_Category,
        data.Intersection_Type
    ]
    
    # Get prediction from the model
    predicted_severity = model_handler.predict(features)

    print("\t\t\t ******* predicted_severity: " , predicted_severity)
    
    # Save to database
    prediction_entry = PredictionModel(
        Driver_Age=data.Driver_Age,
        Safety_Equipment=data.Safety_Equipment,
        Department_Code=data.Department_Code, 
        Mobile_Obstacle=data.Mobile_Obstacle,
        Vehicle_Category=data.Vehicle_Category,
        Position_In_Vehicle=data.Position_In_Vehicle,
        Collision_Type=data.Collision_Type,
        Number_of_Lanes=data.Number_of_Lanes, 
        Time_of_Day=data.Time_of_Day, 
        Journey_Type=data.Journey_Type,
        Obstacle_Hit=data.Obstacle_Hit, 
        Road_Category=data.Road_Category, 
        Gender=data.Gender,
        User_Category=data.User_Category,
        Intersection_Type=data.Intersection_Type,
        Predicted_Severity=predicted_severity,
        request_time=datetime.utcnow(),  # Time when the prediction is made
        user_id=current_user.id 
    )
    
    db.add(prediction_entry)  # Add the new prediction record
    db.commit()  # Commit the changes to save it in the database
    db.refresh(prediction_entry)  # Refresh to get the updated object (with the id)
    
    # Return the prediction along with the record id and the username of the current user 
    return {
        "predicted_severity": prediction_entry.Predicted_Severity,
        "id": prediction_entry.id,
        "user": current_user.username
    }
    