import pathlib
from langchain import prompts
from langchain import schema
from langchain_core import output_parsers
from pydantic import Field, BaseModel
from typing import TypedDict, List, NamedTuple

SYS_PROMPT_BASE_PATH: pathlib.Path = pathlib.Path(__file__).parent / "prompt_cache/"
_COVER_LETTER_FILE_NAME: str = "cover_letter_refinement_system_prompt.txt"
_LLM_JUDGE_FILE_NAME: str = "llm_judge_system_prompt.txt"
_COVER_LETTER_GENERATION_FROM_REFINEMENT_PROMT: str = "cover_letter_generation_from_refinement_promt.txt"
_LINGUISTIC_STYLE_CHECKER_PROMPT: str = "linguistic_style_checker_prompt.txt"

_ORIGINAL_COVER_LETTER: str = "original_anschreiben"
_INTERNSHIP_DOCUMENT: str = "internship_document"
_WORKING_STUDENT_PROOF_1: str = "working_student_proof_1"
_WORKING_STUDENT_PROOF_2: str = "working_student_proof_2"

_DESCRIPTION_PATH: pathlib.Path = (pathlib.Path(__file__).parent.parent / "data/job_descriptions").resolve()
_CANDIDATE_META_DATA_PATH: pathlib.Path = (pathlib.Path(__file__).parent.parent / "data/candidate_meta_data").resolve()

class RequiredSkills(BaseModel):
    technical_skills: List[str] = Field(description="A list of technical skills extracted from Recommendation Letter 1 or Recommendation Letter 2")
    soft_skills: List[str] = Field(description="A list of soft skills extracted from Recommendation Letter 1 or Recommendation Letter 2")

class CandidateStrengthsRelevantToRole(BaseModel):
    skill_or_ares: str = Field("Specific Skill/Area (e.g., Data Analysis, Project Management)")
    evidence_or_impact: str = Field("Quantifiable achievement or specific example from internship/recommendations demonstrating this skill. E.g., 'Analyzed X data sets, leading to Y% improvement in Z process.'")
    source: List[str] = Field("source of evidence e.g., 'Internship Certificate', 'Recommendation Letter 1', 'Recommendation Letter 2', 'Original Cover Letter'")

class TargetRoleSummary(BaseModel):
    role_title: str = Field(description="Extracted Role Title")
    company_name: str = Field(description="Extracted Company Name")
    key_responsibilities: List[str] = Field(description="A detailed list of key responsibilities extracted from job_description")
    required_skills: RequiredSkills = Field(description="A list of required skills extracted from job description")
    qualifications: List[str] = Field(description="A list of qualification extracted from the internship document Recommendation Letter 1 and Recommendation Letter 2")
    company_value_or_focus: List[str] = Field(description="Any specific values, mission, or project types mentioned in Job Description")
    candidate_strengths_relevant_to_role: List[CandidateStrengthsRelevantToRole] = Field(description="")
    candidate_selling_points: List[str] = Field(description="A list of unique experiences or qualities that makes the candidate stand out (e.g., 'Cross-functional team leadership in a startup environment').")
    original_letter_highlights: List[str] = Field(description="Any particularly strong phrases or unique insights from the original that should be retained or built upon.")
    areas_for_emphasis_or_improvement_in_new_letter: List[str] = Field(description="Specific points from the job description that need stronger emphasis in the new letter and or Areas where original documents were vague and need more concrete examples/quantification.")
    overall_tone_and_fit_notes: str = Field(description="Suggestions for the tone (e.g., 'enthusiastic and innovative', 'detail-oriented and reliable') and how to emphasize cultural fit.")


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
    return result.resolve()


def load_sys_prompt_from_disk(file_name: str) -> str:
    cache = ""
    with open(to_path(file_name), encoding="utf-8", mode="r") as raw_file:
        cache += "".join(raw_file.readlines())
    return cache

def create_cover_letter_refinement_inputs(company_job_description_name: str) -> CoverLetterRefinementDict:
    """
    Method Creates the inputs to the cover letter refinement.
    The requirements are in disk and are loaded directly to a dictionary
    :param company_job_description_name: name of the company we'd like to create a cover letter for ,
    :return:
    """
    job_description: str = ""

    with open(_DESCRIPTION_PATH / f"{company_job_description_name}.txt", encoding="utf-8", mode="r") as raw_description:
        job_description += "".join(raw_description.readlines())

    results: CoverLetterRefinementDict = {
        "job_description": job_description,
        "original_cover_letter": helper_read_candidate_meta_data(_ORIGINAL_COVER_LETTER),
        "internship_certificate": helper_read_candidate_meta_data(_INTERNSHIP_DOCUMENT),
        "recommendation_letter_1": helper_read_candidate_meta_data(_WORKING_STUDENT_PROOF_1),
        "recommendation_letter_2": helper_read_candidate_meta_data(_WORKING_STUDENT_PROOF_2)
    }
    return results

class CoverLetterResult_T(NamedTuple):
    parser: output_parsers.JsonOutputParser
    prompt_template: prompts.PromptTemplate

def cover_letter_refinement_prompt_template_gen() -> CoverLetterResult_T:
    refinement_output_parser = output_parsers.JsonOutputParser(pydantic_object=TargetRoleSummary)
    prompt_template = prompts.PromptTemplate(
        input_variables=["job_description", "original_cover_letter", "internship_certificate", "recommendation_letter_1", "recommendation_letter_2"],
        template=load_sys_prompt_from_disk(file_name=_COVER_LETTER_FILE_NAME),
        partial_variables={"format_instructions": refinement_output_parser.get_format_instructions()}
    )
    return CoverLetterResult_T(
        parser=refinement_output_parser,
        prompt_template=prompt_template
    )


def cover_letter_generator() -> prompts.PromptTemplate:
    return prompts.PromptTemplate(
        input_variables=["refined_context", "job_description", "original_cover_letter"],
        template=load_sys_prompt_from_disk(file_name=_COVER_LETTER_GENERATION_FROM_REFINEMENT_PROMT)
    )

def authenticity_checker_generator() -> prompts.PromptTemplate:
    return prompts.PromptTemplate(
        input_variables=["refined_context", "job_description", "original_cover_letter", "generated_cover_letter"]
    )

def linguistic_style_checker_generator() -> prompts.PromptTemplate:
    return prompts.PromptTemplate(
        input_variables=["cover_letter_text"],
        template=load_sys_prompt_from_disk(file_name=_LINGUISTIC_STYLE_CHECKER_PROMPT)
    )


def llm_judge_prompt_template_gen() -> prompts.PromptTemplate:
    return prompts.PromptTemplate(
        input_variables=["refined_context", "job_description", "original_cover_letter" ],
        template=load_sys_prompt_from_disk(file_name=_LLM_JUDGE_FILE_NAME)
    )