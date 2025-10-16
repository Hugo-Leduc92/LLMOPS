from kfp.dsl import OutputPath, component


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas>=2.3.2",
        "datasets==4.0.0",
        "gcsfs",
    ],
)
def data_transformation_component(
    input_gcs_csv_uri: str,
    user_column: str = "sentence",
    assistant_column: str = "translation",
    assistant_fallback_column: str = "translation_extra",
    prefer_extra: bool = True,
    emit_both_variants: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
    train_output_path: OutputPath(str) = "",
    test_output_path: OutputPath(str) = "",
) -> None:
    """Format and split Yoda Sentences for Phi-3 fine-tuning.

    - Chooses assistant text from `translation_extra` when `prefer_extra=True`, otherwise `translation`.
    - Optionally emits both variants per row when `emit_both_variants=True`.
    """
    import logging
    import json
    import pandas as pd
    from datasets import Dataset

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting data transformation process...")

    logger.info(f"Reading from {input_gcs_csv_uri}")
    df = pd.read_csv(input_gcs_csv_uri)

    # Validate columns
    expected = {user_column}
    if assistant_column:
        expected.add(assistant_column)
    if assistant_fallback_column:
        expected.add(assistant_fallback_column)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Build conversations using selected assistant column
    conversations = []
    for _, row in df.iterrows():
        user_text = str(row[user_column])
        primary_ok = assistant_column in df.columns and pd.notna(row[assistant_column])
        extra_ok = assistant_fallback_column in df.columns and pd.notna(row[assistant_fallback_column])

        if prefer_extra and extra_ok:
            assistant_text = str(row[assistant_fallback_column])
        elif primary_ok:
            assistant_text = str(row[assistant_column])
        elif extra_ok:
            assistant_text = str(row[assistant_fallback_column])
        else:
            assistant_text = ""

        conversations.append([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ])

        if emit_both_variants and primary_ok and extra_ok:
            other_text = str(row[assistant_column]) if prefer_extra else str(row[assistant_fallback_column])
            conversations.append([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": other_text},
            ])

    ds = Dataset.from_dict({"messages": [json.dumps(conv) for conv in conversations]})
    logger.info("Splitting dataset: test_size=%.2f seed=%d", test_size, random_seed)
    split = ds.train_test_split(test_size=test_size, seed=random_seed)

    # Write outputs
    train_df = pd.DataFrame({"messages": split["train"]["messages"]})
    test_df = pd.DataFrame({"messages": split["test"]["messages"]})
    logger.info(f"Writing train dataset to {train_output_path}...")
    train_df.to_csv(train_output_path, index=False)
    logger.info(f"Writing test dataset to {test_output_path}...")
    test_df.to_csv(test_output_path, index=False)
    logger.info("Data transformation process completed successfully")


