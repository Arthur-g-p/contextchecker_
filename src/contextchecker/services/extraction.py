"""
Extraction service — orchestrates the extract pipeline.

Pipeline steps:
1. Validation: Ensures API key is configured and input data has 'response' keys.
2. Filtering: Skips items that already contain a {model}_response_kg key.
3. Payloading: Plucks response strings, builds ExtractionPayload for worker.
4. Execution: Delegates to Extractor worker.
5. Serialization: Mutates dicts in-place with returned triplets.
"""

from contextchecker import settings
from contextchecker.exceptions import InvalidInputError, FilterError
from contextchecker.models import ExtractionPayload

logger = settings.get_logger(__name__)


def run_extract_service(data: list[dict], model: str) -> list[dict]:
    """
    Orchestrates the extraction pipeline.

    Receives raw JSON records from the CLI, validates them,
    filters already-processed items, sends the rest through
    the Extractor worker, and writes results back into the dicts.
    """

    # 1. Validate config
    if not settings.EXTRACTOR_API_KEY:
        raise InvalidInputError(
            "EXTRACTOR_API_KEY is required for extraction. Set it in your .env file."
        )

    # 2. Validate input
    valid = []
    for i, item in enumerate(data):
        if "response" not in item:
            logger.warning("Item %d has no 'response' key — skipping.", i)
            continue
        valid.append(item)

    if not valid:
        raise InvalidInputError("No items contain a 'response' key.")

    # 3. Filter already-processed
    kg_key = f"{model}_response_kg"
    pending = [item for item in valid if kg_key not in item]

    if not pending:
        raise FilterError(
            f"All {len(valid)} items already have '{kg_key}'. Nothing to extract."
        )

    logger.info(
        "Extraction: %d items total, %d valid, %d pending.",
        len(data), len(valid), len(pending),
    )

    # 4. Build payloads and execute
    # TODO: instantiate Extractor worker and call it
    # payloads = [ExtractionPayload(text=item["response"]) for item in pending]
    # results = await extractor.extract_batch(payloads)

    # 5. Serialize results back into dicts
    # for item, triplets in zip(pending, results):
    #     item[kg_key] = triplets

    return data
