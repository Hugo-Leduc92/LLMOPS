from kfp.dsl import pipeline

from src.pipeline_components.data_transformation_component import data_transformation_component


@pipeline(name="yoda-data-processing-pipeline")
def model_training_pipeline(
    raw_dataset_uri: str,
) -> None:
    data_transformation_component(
        train_test_split_ratio=0.2,
        raw_dataset_uri=raw_dataset_uri,
    )  # type: ignore


