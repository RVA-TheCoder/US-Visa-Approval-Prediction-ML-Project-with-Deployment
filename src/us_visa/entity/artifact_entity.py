from dataclasses import dataclass


"""
This module defines 'Artifact' dataclasses for the US Visa Approval project.

Artifacts are structured outputs from each stage of the ML pipeline.  
They store file paths, metrics, or statuses produced by a step, and are passed to the next step.  

For example:
    - DataIngestionArtifact stores the paths of ingested datasets.
    - ModelTrainerArtifact stores the trained model path and training metrics.
"""





@dataclass
class DataIngestionArtifact:
    
    """
    Artifact generated after the Data Ingestion stage.

    Attributes:
        (a) trained_data_filepath (str): Path to the processed training dataset.
        (b) test_data_filepath (str): Path to the processed testing dataset.
    """
    
    trained_data_filepath:str
    test_data_filepath:str
    


@dataclass
class DataValidationArtifact:
    
    """
    Artifact generated after the Data Validation stage.

    Attributes:
        (a) data_validation_status (bool): Whether the dataset passed validation checks.
        (b) data_validation_message (str): Summary or reason for validation result.
        (c) data_drift_report_filepath (str): Path to the data drift report file.
    """
    
    data_validation_status:bool
    data_validation_message:str
    data_drift_report_filepath:str


@dataclass
class DataTransformationArtifact:
    
    """
    Artifact generated after the Data Transformation stage.

    Attributes:
        (a) data_transformer_object_filepath (str): Path to the serialized transformer/preprocessor object.
        (b) transformed_train_data_filepath (str): Path to transformed training data (.npy).
        (c) transformed_test_data_filepath (str): Path to transformed testing data (.npy).
    """
    
    data_transformer_object_filepath:str 
    transformed_train_data_filepath:str
    transformed_test_data_filepath:str



@dataclass
class ClassificationMetricArtifact:
    
    """
    Stores classification performance metrics for both train and test sets.

    Attributes:
        (a) train_accuracy (float): Accuracy on the training set.
        (b) train_f1_score (float): F1-score on the training set.
        (c) train_precision_score (float): Precision score on the training set.
        (d) train_recall_score (float): Recall score on the training set.
        (e) test_accuracy (float): Accuracy on the test set.
        (f) test_f1_score (float): F1-score on the test set.
        (g) test_precision_score (float): Precision score on the test set.
        (h) test_recall_score (float): Recall score on the test set.
    """
    
    train_accuracy:float
    train_f1_score:float
    train_precision_score:float
    train_recall_score:float
    test_accuracy:float
    test_f1_score:float
    test_precision_score:float
    test_recall_score:float


@dataclass
class ModelTrainerArtifact:
    
    """
    Artifact generated after the Model Training stage.

    Attributes:
        (a) trained_model_filepath (str): Path to the serialized trained model.
        (b) metric_artifact (ClassificationMetricArtifact): Evaluation metrics from training/testing.
    """
    
    trained_model_filepath:str 
    metric_artifact:ClassificationMetricArtifact
    

@dataclass
class ModelEvaluationArtifact:
    
    """
    Artifact generated after the Model Evaluation stage.

    Attributes:
        (a) is_trained_model_accepted (bool): Whether the trained model meets acceptance criteria.
        (b) eval_metric_f1score_diff (float): Difference in F1-score between the new model and production model.
        (c) s3_prod_model_path (str): S3 path of the production model.
        (d) trained_model_path (str): Local path of the newly trained model.
    """
    
    is_trained_model_accepted:bool
    eval_metric_f1score_diff:float
    s3_prod_model_path:str 
    trained_model_path:str   
    


@dataclass
class ModelPusherArtifact:
    
    """
    Artifact generated after the Model Pushing stage.

    Attributes:
        (a) bucket_name (str): S3 bucket name where the model is pushed.
        (b) s3_model_path (str): Full S3 path of the pushed model.
    """
    
    bucket_name:str
    s3_model_path:str
