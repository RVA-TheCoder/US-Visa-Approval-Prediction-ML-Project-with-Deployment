from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    trained_filepath:str
    test_filepath:str
    

@dataclass
class DataValidationArtifact:
    data_validation_status:bool
    data_validation_message:str
    data_drift_report_filepath:str


@dataclass
class DataTransformationArtifact:
    
    transformed_object_filepath:str 
    transformed_train_filepath:str
    transformed_test_filepath:str



@dataclass
class ClassificationMetricArtifact:
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
    trained_model_filepath:str 
    metric_artifact:ClassificationMetricArtifact










