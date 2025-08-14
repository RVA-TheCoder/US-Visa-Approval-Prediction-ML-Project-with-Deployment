import sys

from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from us_visa.entity.config_entity import ModelPusherConfig

from us_visa.cloud_storage.aws_storage import SimpleStorageService
from us_visa.entity.s3_estimator import USvisaEstimator



class ModelPusher:
    
    """
    Handles pushing the trained model to AWS S3 for deployment.

    This class is responsible for:
        1. Receiving the trained model path from the ModelEvaluation stage.
        2. Uploading the trained model to the specified S3 location.
        3. Creating a ModelPusherArtifact containing S3 storage details.
    """
    
    def __init__(self, 
                 model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        
        """
        Initialize ModelPusher with the evaluation results and config.

        Parameters : 
            (a) model_evaluation_artifact (ModelEvaluationArtifact): 
                        Contains details about the evaluated/trained model.
            (b) model_pusher_config (ModelPusherConfig): 
                        Configuration parameters for S3 bucket and model storage path.
        """
        
        self.s3 = SimpleStorageService()
        
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        
        #self.usvisa_estimator = USvisaEstimator(bucket_name=model_pusher_config.bucket_name,
        #                                        s3_prod_model_path=model_pusher_config.s3_model_key_path)
        
        self.usvisa_estimator = USvisaEstimator(bucket_name=model_pusher_config.bucket_name,
                                                #s3_prod_model_path=model_pusher_config.s3_model_key_path
                                                )

        #s3_prod_model_path=self.model_eval_config.s3_prod_model_key_path          


    def initiate_model_pusher(self) -> ModelPusherArtifact:
        
        """
        Upload the trained model to the configured S3 bucket.

        Steps:
            1. Upload model from local storage to S3.
            2. Return a ModelPusherArtifact containing upload details. (s3_model_path)

        Returns:
            ModelPusherArtifact: Contains S3 bucket name and model path.
        
        Raises:
            USvisaException: If the upload process fails.
        """
        
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            
            logging.info("Uploading artifacts folder to s3 bucket")
            
            self.usvisa_estimator.upload_model_to_s3(from_filepath=self.model_evaluation_artifact.trained_model_path,
                                                     remove=False
                                                     )


            model_pusher_artifact = ModelPusherArtifact(bucket_name=self.model_pusher_config.bucket_name,
                                                        s3_model_path=self.model_pusher_config.s3_model_key_path
                                                        )

            logging.info("Uploaded artifacts folder to s3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelTrainer class")
            
            return model_pusher_artifact
        
        except Exception as e:
            raise USvisaException(e, sys) from e
        
        
        
        
        
