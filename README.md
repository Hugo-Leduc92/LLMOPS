# LLMOPS

## Environment variables

Create a `.env` file (not committed) alongside this README with:

```
GCP_PROJECT_ID=your-project-id
GCP_REGION=your-region
GCP_BUCKET_NAME=your-bucket-name
```

You can copy from `.env.example`.

## Session 2 pipeline

Code lives under `src/` and `scripts/` once scaffolded.

### Run locally

1. Install deps using uv or pip.
2. Ensure `.env` contains GCP settings.
3. Compile and submit pipeline:

```
python scripts/pipeline_runner.py gs://YOUR_BUCKET/path/to/yoda_sentences.csv
```