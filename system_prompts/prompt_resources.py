import pathlib
import logging
import dataclasses

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_file_handler = logging.FileHandler('resource_log.log')
_stream_handler = logging.StreamHandler()

_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_file_handler.setFormatter(_formatter)
_stream_handler.setFormatter(_formatter)

_logger.addHandler(_file_handler)
_logger.addHandler(_stream_handler)


ORIGINAL_COVERLETTER_PATH: pathlib.Path = (pathlib.Path(__file__).parent.parent
                                           / "data/original_anschreiben.txt").resolve()
DESCRIPTION_PATH: pathlib.Path = (pathlib.Path(__file__).parent.parent
                                  / "data/job_descriptions").resolve()

_logger.info(f"{ORIGINAL_COVERLETTER_PATH=}")
_logger.info(f"{DESCRIPTION_PATH=}")
assert ORIGINAL_COVERLETTER_PATH.exists(), "the path to the original cover letter does not exist"
assert DESCRIPTION_PATH.exists(), "the path to the description does not exist"


@dataclasses.dataclass
class PromptResourceReader:
    file_location: pathlib.Path

    def read(self) -> str:
        cache = ""
        _logger.info(f"Reading from file {self.file_location}")
        with open(self.file_location, mode="r", encoding="utf-8") as raw_file:
            cache += "".join(raw_file.readlines())

        return cache

def resolve_company_description(company_name: str) -> str | None:
    company_description_path: pathlib.Path = DESCRIPTION_PATH / f"{company_name}_description.txt"
    if company_description_path.exists():
        return PromptResourceReader(file_location=company_description_path).read()
    return None




