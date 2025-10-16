from kfp import dsl
from kfp.dsl import pipeline

from src.pipeline_components.data_transformation_component import data_transformation_component


@pipeline(name="yoda-data-processing-pipeline")
def model_training_pipeline(
    input_gcs_csv_uri: str,
    user_column: str = "sentence",
    assistant_column: str = "translation",
    assistant_fallback_column: str = "translation_extra",
    prefer_extra: bool = True,
    emit_both_variants: bool = False,
):
    step = data_transformation_component(
        input_gcs_csv_uri=input_gcs_csv_uri,
        user_column=user_column,
        assistant_column=assistant_column,
        assistant_fallback_column=assistant_fallback_column,
        prefer_extra=prefer_extra,
        emit_both_variants=emit_both_variants,
    )


