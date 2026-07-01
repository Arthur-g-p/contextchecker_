"""
RefChecker - extraction + checking in one run, one output document.

Composes ExtractionService then CheckingService over one dataset. Talks to
services only. Validation drops items lacking 'response' or 'reference' up
front, so the checker never has to re-drop them. The pipeline owns the
output artifact: the merged document, saved in the app's JSON format.
"""

import json
from pathlib import Path

from contextchecker import settings
from contextchecker.pipelines.base import BasePipeline
from contextchecker.services.extraction import ExtractionService
from contextchecker.services.checking import CheckingService

logger = settings.get_logger(__name__)


class RefCheckerPipeline(BasePipeline):
    """Extraction + checking as a single reference-checking run."""

    def __init__(
        self,
        extractor_model: str,
        checker_model: str,
        *,
        extractor_base_url: str | None = None,
        checker_base_url: str | None = None,
        concurrency: int = 10,
        extractor_max_retries: int | None = 2,
        dedup: bool = True,
        joint: bool = True,
        joint_num: int = settings.DEFAULT_JOINT_NUM,
        max_words: int | None = None,
        checker_max_retries: int | None = None,
    ):
        self._extractor_model = extractor_model
        self._checker_model = checker_model

        # Both services fail-fast on their own API key here, before any run.
        self._extraction = ExtractionService(
            model=extractor_model,
            base_url=extractor_base_url,
            concurrency=concurrency,
            max_retries=extractor_max_retries,
            quiet=True,
            dedup=dedup,
        )
        self._checking = CheckingService(
            model=checker_model,
            extractor_model=extractor_model,
            base_url=checker_base_url,
            concurrency=concurrency,
            joint=joint,
            joint_num=joint_num,
            max_words=max_words,
            max_retries=checker_max_retries,
            quiet=True,
        )

    def _required_keys(self) -> tuple[str, ...]:
        # extraction needs 'response'; checking needs 'reference'.
        return ("response", "reference")

    async def run(self, data: list[dict]) -> list[dict]:
        """Run extraction then checking over *data*, in place; return data.

        Raises InvalidInputError if no item carries both 'response' and
        'reference'.
        """
        self._canonicalize_keys(data)      # Step 0: normalize aliases
        valid = self._validate(data)       # Step 1: drop items missing fields

        await self._extraction.run(valid)  # writes {extractor_model}_response_kg
        await self._checking.run(valid)    # writes verdicts onto each triplet

        return data

    # -- Output artifact (pipeline owns it) --

    def process_file(
        self, input_path: str | Path, output_path: str | Path | None = None
    ) -> Path:
        """Load a JSON dataset, run the pipeline, save the merged document.

        Format and default location mirror the other commands: pretty JSON,
        UTF-8, written to results/{input_name} next to the input.
        """
        input_path = Path(input_path)
        data = json.loads(input_path.read_text(encoding="utf-8"))

        result = self.run_sync(data)

        if output_path is None:
            output_path = input_path.parent / "results" / input_path.name
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return output_path
