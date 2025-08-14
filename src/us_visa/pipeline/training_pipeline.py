import sys , shutil
from pathlib import Path

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.constants import *

from us_visa.entity.config_entity import (DataIngestionConfig,
                                          DataValidationConfig,
                                          DataTransformationConfig,
                                          ModelTrainerConfig,
                                          ModelEvaluationConfig,
                                          ModelPusherConfig)

from us_visa.entity.artifact_entity import (DataIngestionArtifact,
                                            DataValidationArtifact,
                                            DataTransformationArtifact,
                                            ModelTrainerArtifact,
                                            ModelEvaluationArtifact,
                                            ModelPusherArtifact)


from us_visa.components.data_ingestion import DataIngestion
from us_visa.components.data_validation import DataValidation
from us_visa.components.data_transformation import DataTransformation
from us_visa.components.model_trainer import ModelTrainer
from us_visa.components.model_evaluation import ModelEvaluation
from us_visa.components.model_pusher import ModelPusher





class TrainPipeline:
    
    """
    Orchestrates the entire ML training pipeline for the US Visa prediction project.
    
    This pipeline consists of:
        1. Data ingestion from MongoDB or another source.
        2. Data validation to ensure schema and quality checks.
        3. Data transformation (feature engineering, encoding, scaling, etc.).
        4. Model training with specified algorithms and hyperparameters.
        5. Model evaluation to compare with existing production models.
        6. Model pushing to AWS S3 and saving locally if the new model is better.
    
    Each stage consumes an Artifact from the previous stage and outputs its own Artifact.
    """
    
    def __init__(self):
        
        # Initialize configuration objects for each pipeline stage
        
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()
        
        
    def start_data_ingestion(self)->DataIngestionArtifact:
        
        """
        Runs the Data Ingestion component which retrieves the dataset,
        splits into training/testing sets, and stores them locally.
        """
        
        try:
            logging.info("Entered the start_data_ingestion method TrainPipeline class inside src/us_visa/pipline/training_pipeline.py")
            
            logging.info("Getting the data from mongodb")
            
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            
            logging.info("Got the train_data and test_data")
            logging.info("Exited the start_data_ingestion method TrainPipeline class inside src/us_visa/pipline/training_pipeline.py")
            
            return data_ingestion_artifact
        
        except Exception as e:
            raise USvisaException(e, sys) from e
        
    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        
        """
        Validates ingested data to ensure schema correctness and quality checks.
        """
        
        logging.info("Entered the start_data_validation method of TrainPipeline class inside src/us_visa/pipline/training_pipeline.py")

        try:
            # created the object of the class
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=self.data_validation_config
                                             )

            # Calling method
            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")

            logging.info("Exited the start_data_validation method of TrainPipeline class inside src/us_visa/pipline/training_pipeline.py")
            

            return data_validation_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e
        
    
    def start_data_transformation(self,
                                  data_ingestion_artifact: DataIngestionArtifact,
                                  data_validation_artifact: DataValidationArtifact
                                  ) -> DataTransformationArtifact:
        """
        Transforms validated data into a processed format ready for model training.
        """
        
        try:
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                     data_validation_artifact=data_validation_artifact,
                                                     data_transformation_config=self.data_transformation_config,
                                                     )
            
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            
            return data_transformation_artifact
        
        except Exception as e:
            raise USvisaException(e, sys) 
    
    
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
       
        """
        Trains a machine learning model using transformed data.
        """
        
        try:
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                         model_trainer_config=self.model_trainer_config
                                         )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact

        except Exception as e:
            raise USvisaException(e, sys)
    
    
    def start_model_evaluation(self, 
                               data_ingestion_artifact:DataIngestionArtifact,
                               data_transformation_artifact:DataTransformationArtifact,
                               model_trainer_artifact:ModelTrainerArtifact,
                               
                               ) -> ModelEvaluationArtifact:
        """
        Compares the newly trained model with the existing production model.
        """
        try:
            model_evaluation = ModelEvaluation(data_ingestion_artifact=data_ingestion_artifact,
                                               data_transformation_artifact=data_transformation_artifact,
                                               model_trainer_artifact=model_trainer_artifact,
                                               model_eval_config = self.model_evaluation_config,
                                               )
           
        
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        
        
        except Exception as e:
            raise USvisaException(e, sys)
    
    
    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        
        """
        Pushes the model to AWS S3 and/or saves locally for production use.
        """
        try:
            
            model_pusher = ModelPusher(model_evaluation_artifact=model_evaluation_artifact,
                                       model_pusher_config=self.model_pusher_config
                                       )
            
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            
            return model_pusher_artifact
        
        except Exception as e:
            raise USvisaException(e, sys)
    
    
    def run_pipeline(self) -> None:
        
        """
        Executes all stages of the ML pipeline sequentially.
        """
        try :
            # Step 1: Data ingestion
            data_ingestion_artifact=self.start_data_ingestion()
            # Step 2: Data validation
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            # Step 3: Data transformation
            data_transformation_artifact=self.start_data_transformation(
                                                        data_ingestion_artifact=data_ingestion_artifact,
                                                        data_validation_artifact=data_validation_artifact
                                                        )
            # Step 4: Model training
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            # Step 5: Model evaluation
            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                    data_transformation_artifact=data_transformation_artifact,
                                                                    model_trainer_artifact=model_trainer_artifact)
            
            # Step 6: Model pushing if improved
            if not model_evaluation_artifact.is_trained_model_accepted:
                
                print("Trained model is not better than the AWS S3 model.")
                print("Therefore, No need to push the Trained model to AWS S3 bucket for production use.")
                logging.info("Trained model is not better than the AWS S3 model. Skipping push.")
                  
            else:
                
                # Pushing the trained model to GCS bucket for production use
                model_pusher_artifacts=self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
                print("Pushed the trained model to AWS S3 bucket......")
                
                
                #trained_model_filepath = model_trainer_artifact.trained_model_filepath
                from_filepath = Path(model_trainer_artifact.trained_model_filepath)
                to_filepath = Path(f"{LOCAL_PRODUCTION_MODEL_DIR}/{Path(S3_PRODUCTION_MODEL_NAME)}")

                # Make sure the parent directory exists
                to_filepath.parent.mkdir(parents=True, exist_ok=True)

                # Copy the file
                shutil.copy(from_filepath, to_filepath)
                print(f"Saved the current trained model as Production model to {LOCAL_PRODUCTION_MODEL_DIR}/{S3_PRODUCTION_MODEL_NAME}")
                
    
            logging.info("Pipeline execution completed successfully.\nExited the run_pipline method of TrainPipeline class in src/us_visa/pipline/training_pipeline.py")

        
        except Exception as e:
            raise USvisaException(e, sys) from e
        
        
