import os
from pathlib import Path
import logging


"""
Project Template Generator Script.

This script creates the directory and file structure for the 'us_visa' ML project.
It ensures all required folders exist and creates empty placeholder files where needed.
Useful for initializing a new project with a consistent layout.

Steps:
    1. Define the project name and required files.
    2. Create missing directories.
    3. Create empty files if they don't exist or are of size 0.
"""

# Configure logging for console output
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] : %(message)s :')

# Project name
project_name="us_visa"

# List of all required files and folders
list_of_files=[
    ".github/workflows/.gitkeep",
    
    f"src/{project_name}/__init__.py",
    
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",  
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/components/model_pusher.py",
    
    f"src/{project_name}/configuration/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/entity/artifact_entity.py",
    
    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/logger/__init__.py",
    
    f"src/{project_name}/pipline/__init__.py",
    f"src/{project_name}/pipline/training_pipeline.py",
    f"src/{project_name}/pipline/prediction_pipeline.py",
    
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/main_utils.py",
    
    "config/model.yaml",
    "config/schema.yaml",
    
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "demo.py",
    "setup.py",
    
]


# Iterate through the file list and create directories & files if missing
for filepath in list_of_files:

    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directories if they don't exist
    if filedir != "":

        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory : {filedir} for the file:{filename}")

    # Create empty file if missing or empty
    if not filepath.exists() or filepath.stat().st_size == 0:

        with open(filepath, "w") as f:
            logging.info(f"Creating empty file : {filepath}")
            pass

    else:
        logging.info(f"{filename} is already exists.")

    










