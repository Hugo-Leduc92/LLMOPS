import os
import sys
from google.cloud import aiplatform
from google.cloud.aiplatform import PipelineJob

# Ensure project root (which contains `src/`) is on PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.constants import get_gcp_config
from src.pipelines.model_training_pipeline import model_training_pipeline


def compile_pipeline(pipeline_spec_path: str) -> None:
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=model_training_pipeline,
        package_path=pipeline_spec_path,
    )


def submit_pipeline(pipeline_spec_path: str, raw_dataset_uri: str) -> None:
    gcp = get_gcp_config()
    aiplatform.init(project=gcp.project_id, location=gcp.region)

    job = PipelineJob(
        display_name="yoda-data-processing",
        template_path=pipeline_spec_path,
        pipeline_root=f"gs://{gcp.bucket_name}/vertex-pipelines",
        parameter_values={
            "raw_dataset_uri": raw_dataset_uri,
        },
    )
    job.run(sync=True)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/pipeline_runner.py gs://bucket/path/to/yoda_sentences.csv", file=sys.stderr)
        sys.exit(2)

    pipeline_spec_path = os.path.join(os.path.dirname(__file__), "../artifacts/yoda_data_processing.json")
    os.makedirs(os.path.dirname(pipeline_spec_path), exist_ok=True)

    compile_pipeline(pipeline_spec_path)
    submit_pipeline(pipeline_spec_path, raw_dataset_uri=sys.argv[1])


if __name__ == "__main__":
    main()


