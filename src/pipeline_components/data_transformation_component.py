from typing import List
import json
from kfp import dsl
from kfp.dsl import component, OutputPath


@component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas==2.2.2",
        "datasets==3.0.1",
        "gcsfs==2024.6.1",
        "pyarrow==17.0.0",
    ],
)
def data_transformation_component(
    input_gcs_csv_uri: str,
    user_column: str,
    assistant_column: str = "translation",
    assistant_fallback_column: str = "translation_extra",
    prefer_extra: bool = True,
    emit_both_variants: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
    train_output_path: OutputPath(str) = "",
    test_output_path: OutputPath(str) = "",
) -> None:
    import logging
    import json
    import pandas as pd
    from datasets import Dataset

    def _to_conversation_rows(
        df: "pd.DataFrame",
        user_col: str,
        assistant_col: str,
        assistant_fallback_col: str | None = None,
        prefer_extra: bool = True,
        emit_both_variants: bool = False,
    ) -> List[List[dict]]:
        conversation_rows: List[List[dict]] = []
        for _, row in df.iterrows():
            user_text = str(row[user_col])
            primary_available = assistant_col in df.columns and pd.notna(row[assistant_col])
            extra_available = (
                assistant_fallback_col is not None
                and assistant_fallback_col in df.columns
                and pd.notna(row[assistant_fallback_col])
            )

            if prefer_extra and extra_available:
                assistant_text = str(row[assistant_fallback_col])
            elif primary_available:
                assistant_text = str(row[assistant_col])
            elif extra_available:
                assistant_text = str(row[assistant_fallback_col])
            else:
                assistant_text = ""

            conversation_rows.append([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ])

            if emit_both_variants and primary_available and extra_available:
                other_text = str(row[assistant_col]) if prefer_extra else str(row[assistant_fallback_col])
                conversation_rows.append([
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": other_text},
                ])
        return conversation_rows

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("data_transformation_component")

    logger.info("Reading CSV from %s", input_gcs_csv_uri)
    df = pd.read_csv(input_gcs_csv_uri)
    expected_cols = {user_column}
    if assistant_column:
        expected_cols.add(assistant_column)
    if assistant_fallback_column:
        expected_cols.add(assistant_fallback_column)
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in input: {missing}")

    logger.info("Converting to HF Dataset and applying chat template")
    conversations = _to_conversation_rows(
        df=df,
        user_col=user_column,
        assistant_col=assistant_column,
        assistant_fallback_col=assistant_fallback_column,
        prefer_extra=prefer_extra,
        emit_both_variants=emit_both_variants,
    )
    ds = Dataset.from_dict({"messages": [json.dumps(conv) for conv in conversations]})

    logger.info("Splitting dataset: test_size=%.2f seed=%d", test_size, random_seed)
    split = ds.train_test_split(test_size=test_size, seed=random_seed)
    train_ds = split["train"]
    test_ds = split["test"]

    logger.info("Writing train dataset to %s", train_output_path)
    train_df = pd.DataFrame({"messages": train_ds["messages"]})
    train_df.to_csv(train_output_path, index=False)

    logger.info("Writing test dataset to %s", test_output_path)
    test_df = pd.DataFrame({"messages": test_ds["messages"]})
    test_df.to_csv(test_output_path, index=False)


