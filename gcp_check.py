import os
import sys
from google.cloud import aiplatform
from google.cloud import storage


def require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        print(f"Missing required environment variable: {var_name}", file=sys.stderr)
        sys.exit(1)
    return value


def main() -> None:
    # Load .env if present to avoid shell parsing issues
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(dotenv_path):
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Strip surrounding quotes if any
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                if key and value and key not in os.environ:
                    os.environ[key] = value

    project_id = require_env("GCP_PROJECT_ID")
    region = require_env("GCP_REGION")
    bucket_name = require_env("GCP_BUCKET_NAME")

    print(f"Using project={project_id}, region={region}, bucket={bucket_name}")

    # Initialize Vertex AI client
    aiplatform.init(project=project_id, location=region)
    print("Vertex AI initialized successfully.")

    # Check GCS bucket accessibility
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    if not bucket.exists():
        print(f"Bucket does not exist or is not accessible: {bucket_name}", file=sys.stderr)
        sys.exit(2)
    print("GCS bucket is accessible.")


if __name__ == "__main__":
    main()
