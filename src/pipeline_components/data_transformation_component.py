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
    raw_dataset_uri: str,
    train_test_split_ratio: float,
    train_dataset: OutputPath("Dataset"),  # type: ignore
    test_dataset: OutputPath("Dataset"),  # type: ignore
) -> None:
    """Format and split Yoda Sentences for Phi-3 fine-tuning.

    Structure mirrors Correction's component interface while preserving existing
    transformation logic (column choices and variant handling with defaults).
    """
    import logging
    import json
    import pandas as pd
    from datasets import Dataset

    # Defaults preserved from previous implementation
    user_column = "sentence"
    assistant_column = "translation"
    assistant_fallback_column = "translation_extra"
    prefer_extra = True
    emit_both_variants = False
    random_seed = 42

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting data transformation process...")

    logger.info(f"Reading from {raw_dataset_uri}")
    df = pd.read_csv(raw_dataset_uri)

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
    logger.info("Splitting dataset: test_size=%.2f seed=%d", train_test_split_ratio, random_seed)
    split = ds.train_test_split(test_size=train_test_split_ratio, seed=random_seed)

    logger.info(f"Writing train dataset to {train_dataset}...")
    split["train"].to_csv(train_dataset, index=False)

    logger.info(f"Writing test dataset to {test_dataset}...")
    split["test"].to_csv(test_dataset, index=False)

    logger.info("Data transformation process completed successfully")


