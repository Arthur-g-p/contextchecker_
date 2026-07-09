"""
RefChecker - reference-checking use case: extraction + checking in one run.

RefChecker is a *pipeline*: a service whose run() composes other services
(ExtractionService then CheckingService) instead of driving a single worker.
The caller-facing interface is identical, so it subclasses BaseService - there
is no separate pipeline base (see the note in services/base.py).

It talks to services only, never a worker. The two services consolidate their
results into the shared data in place, so this class serializes nothing of its
own; the controller (cli.py) persists the returned data.
"""

from contextchecker import settings
from contextchecker.exceptions import InvalidInputError
from contextchecker.services.base import BaseService
from contextchecker.services.extraction import ExtractionService
from contextchecker.services.checking import CheckingService

logger = settings.get_logger(__name__)


class RefCheckerPipeline(BaseService):
    """Extraction + checking composed into a single reference-checking run."""

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
        verbosity: str = "full",
    ):
        self._extractor_model = extractor_model
        self._checker_model = checker_model
        self._init_verbosity(verbosity)

        # Compose the two services. Each fail-fasts on its own API key here.
        self._extraction = ExtractionService(
            model=extractor_model,
            base_url=extractor_base_url,
            concurrency=concurrency,
            max_retries=extractor_max_retries,
            verbosity=verbosity,
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
            verbosity=verbosity,
        )

    # -- Pipeline: the BaseService 7-step run() shape --

    async def run(self, data: list[dict]) -> list[dict]:
        """Run extraction then checking over *data*, in place; return data.

        1. Validate     - drop items missing 'response' or 'reference'
        2. Filter       - none (pass); child services filter their own
        3. Log pre-exec - validation + config
        4. Execute      - delegate to ExtractionService then CheckingService
        5. Serialize    - none; the services consolidated into data already
        6. Log results  - done line
        7. Return mutated data

        Raises InvalidInputError if no item carries both keys.
        """
        self._canonicalize_keys(data)                 # Step 0: normalize aliases
        valid = self._validate(data)                  # 1
        #self._filter(valid)                          # 2 (no-op)
        self._log_validation(len(data), len(valid))   # 3
        self._log_config()

        await self._extraction.run(valid)             # 4: writes {extractor_model}_response_kg
        await self._checking.run(valid)               #    writes verdicts onto each triplet

        self._serialize()                             # 5 (no-op)
        self._log_results(len(data), len(valid))      # 6
        return data                                   # 7

    # -- Validation: drop items dynamically, like every service --

    def _validate(self, data: list[dict]) -> list[dict]:
        """Step 1: Drop items missing 'response' or 'reference'.

        Both stages' inputs are checked up front - extraction needs
        'response', checking needs 'reference' - so the checker never has to
        re-drop. Raises InvalidInputError if nothing is valid.
        """
        valid = []
        for i, item in enumerate(data):
            missing = [k for k in ("response", "reference") if k not in item]
            if missing:
                logger.debug("Item %d missing %s - skipping.", i, ", ".join(missing))
                continue
            valid.append(item)

        if not valid:
            raise InvalidInputError(
                "No items contain both 'response' and 'reference'."
            )
        return valid

    def _filter(self, valid):
        """No pipeline-level filtering: child services filter their own
        already-processed items."""
        pass

    # -- Serialization: none; children consolidated into data in place --

    def _serialize(self, *args, **kwargs) -> None:
        pass

    # -- Logging --

    def _log_validation(self, total: int, valid: int) -> None:
        if self.verbosity != "full":
            return
        dropped = total - valid
        logger.info(" 📂 Validation")
        logger.info("    Total:       %d items", total)
        if dropped:
            logger.info("    ├─ dropped:  %d  (missing response/reference)", dropped)
        logger.info("    └─ valid:    %d items", valid)
        logger.info("")

    def _log_skip(self, *args, **kwargs) -> None:
        pass

    def _log_config(self) -> None:
        if self.verbosity != "full":
            return
        logger.info(" ⚙️  Config")
        logger.info("    Extractor:   %s", self._extractor_model)
        logger.info("    Checker:     %s", self._checker_model)
        logger.info("")

    def _log_results(self, total: int, valid: int) -> None:
        """Step 6: post-execution report. No BL for RefChecker; just the done line."""
        self._log_bl_results()
        self._log_done(total, valid)

    def _log_bl_results(self, *args, **kwargs) -> None:
        pass

    def _log_done(self, total: int, valid: int) -> None:
        if self.verbosity != "full":
            return
        logger.info(" ✅ Done: ref-checked %d/%d items", valid, total)
