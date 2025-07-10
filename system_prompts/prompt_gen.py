import pathlib
from langchain import prompts

from typing import TypedDict

SYS_PROMPT_BASE_PATH: pathlib.Path = pathlib.Path(__file__).parent / "prompt_cache/"
_COVER_LETTER_FILE_NAME: str = "cover_letter_refinement_system_prompt.txt"
_LLM_JUDGE_FILE_NAME: str = "llm_judge_system_prompt.txt"

_ORIGINAL_COVER_LETTER: str = "original_anschreiben"
_INTERNSHIP_DOCUMENT: str = "internship_document"
_WORKING_STUDENT_PROOF_1: str = "working_student_proof_1"
_WORKING_STUDENT_PROOF_2: str = "working_student_proof_2"

_DESCRIPTION_PATH: pathlib.Path = (pathlib.Path(__file__).parent.parent / "data/job_descriptions").resolve()
_CANDIDATE_META_DATA_PATH: pathlib.Path = (pathlib.Path(__file__).parent.parent / "data/candidate_meta_data").resolve()

def helper_read_candidate_meta_data(file_name: str) -> str:
    temp: str = ""

    with open(_CANDIDATE_META_DATA_PATH / f"{file_name}.txt", encoding="utf-8", mode="r") as raw_description:
        temp += "".join(raw_description.readlines())

    return temp

class CoverLetterRefinementDict(TypedDict):
    job_description: str
    original_cover_letter: str
    internship_certificate: str
    recommendation_letter_1: str
    recommendation_letter_2: str


def to_path(file_name: str) -> pathlib.Path:
    result: pathlib.Path = (SYS_PROMPT_BASE_PATH / file_name).resolve()
    print(SYS_PROMPT_BASE_PATH)
    print(SYS_PROMPT_BASE_PATH.exists())
    print(result.exists())
    return result.resolve()


def load_sys_prompt_from_disk(file_name: str) -> str:
    cache = ""
    with open(to_path(file_name), encoding="utf-8", mode="r") as raw_file:
        cache += "".join(raw_file.readlines())
    return cache

def create_cover_letter_refinement_inputs(company_name: str) -> CoverLetterRefinementDict:
    job_description: str = ""

    with open(_DESCRIPTION_PATH / f"{company_name}.txt", encoding="utf-8", mode="r") as raw_description:
        job_description += "".join(raw_description.readlines())

    results: CoverLetterRefinementDict = {
        "job_description": job_description,
        "original_cover_letter": helper_read_candidate_meta_data(_ORIGINAL_COVER_LETTER),
        "internship_certificate": helper_read_candidate_meta_data(_INTERNSHIP_DOCUMENT),
        "recommendation_letter_1": helper_read_candidate_meta_data(_WORKING_STUDENT_PROOF_1),
        "recommendation_letter_2": helper_read_candidate_meta_data(_WORKING_STUDENT_PROOF_2)
    }
    return results


def cover_letter_refinement_prompt_template_gen() -> prompts.PromptTemplate:
    return prompts.PromptTemplate(
        input_variables=["job_description", "original_cover_letter", "internship_certificate", "recommendation_letter_1", "recommendation_letter_2"],
        template=load_sys_prompt_from_disk(file_name=_COVER_LETTER_FILE_NAME)
    )


def llm_judge_prompt_template_gen() -> prompts.PromptTemplate:
    return prompts.PromptTemplate(
        input_variables=["", ],
        template=load_sys_prompt_from_disk(file_name=_LLM_JUDGE_FILE_NAME)
    )