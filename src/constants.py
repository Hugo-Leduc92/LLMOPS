import os
from dataclasses import dataclass


def _load_dotenv_if_present() -> None:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    dotenv_path = os.path.join(repo_root, ".env")
    if os.path.exists(dotenv_path):
        try:
            from dotenv import load_dotenv  # type: ignore
        except Exception:
            # Fallback to manual parsing if python-dotenv is unavailable at runtime
            with open(dotenv_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value and key not in os.environ:
                        os.environ[key] = value
        else:
            load_dotenv(dotenv_path)


_load_dotenv_if_present()


def require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return value


@dataclass(frozen=True)
class GcpConfig:
    project_id: str
    region: str
    bucket_name: str


def get_gcp_config() -> GcpConfig:
    return GcpConfig(
        project_id=require_env("GCP_PROJECT_ID"),
        region=require_env("GCP_REGION"),
        bucket_name=require_env("GCP_BUCKET_NAME"),
    )


