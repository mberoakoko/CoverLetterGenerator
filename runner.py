import dataclasses
import pathlib

import langchain_core
from langchain import prompts
import langchain_core.runnables
import langchain_core.runnables.base
from langchain_ollama import ChatOllama
from langchain import schema
from langchain.schema import runnable

from system_prompts.prompt_gen import (
    cover_letter_refinement_prompt_template_gen,
    create_cover_letter_refinement_inputs
)

RunnableSequence = langchain_core.runnables.base.RunnableSequence

ORIGINAL_COVERLETTER_PATH: pathlib.Path = (pathlib.Path(__file__).parent / "data/original_anschreiben.txt").resolve()
DESCRIPTION_PATH: pathlib.Path = (pathlib.Path(__file__).parent / "data/job_descriptions").resolve()

SELECTED_MODEL: str = "llama3.2"


@dataclasses.dataclass
class Reader:
    file_location: pathlib.Path

    def read(self) -> str:
        cache = ""
        with open(self.file_location, mode="r", encoding="utf-8") as raw_file:
            cache += "".join(raw_file.readlines())

        return cache


def resolve_company_description(company_name: str) -> str | None:
    company_description_path: pathlib.Path = DESCRIPTION_PATH / f"{company_name}_description.txt"
    if company_description_path.exists():
        return Reader(file_location=company_description_path).read()
    return None


def prompt_factory() -> prompts.PromptTemplate:
    system_promt: str = """
        You are a skilled writer tasked with rewriting a cover letter to perfectly fit a given job description. Your goal is to maintain the original tone and style while incorporating relevant keywords and phrases from the job description.
        \n
        Input:
        - Cover Letter: {cover_letter} \n\n\n
        - Job Description: {job_description} \n\n\n
        \n
        Instructions:
        1. Carefully read the cover letter and job description.
        2. Identify key qualifications, skills, and experiences required in the job description.
        3. Rewrite the cover letter to highlight the candidate's relevant qualifications, skills, and experiences that match the job description.
        4. Use specific keywords and phrases from the job description to demonstrate the candidate's fit for the role.
        5. Maintain the original tone and style of the cover letter.
        6. While maintaining the original and tone , try to use compound sentences.
        7. Try not to begin the sentences with Personal Pronouns, mix it up.
        8. Maintain the original language of the cover letter.
        9. The name of the applicant is Martin Odthiambo Mbero. Add it at the conclusion
        \n\n\

        Output: 
    """
    return prompts.PromptTemplate(
        input_variables=["cover_letter", "job_description"],
        template=system_promt
    )


@dataclasses.dataclass
class CoverLetterChef:
    prompt: prompts.PromptTemplate
    model: ChatOllama
    cover_letter: str
    job_description_context: str

    def create_chain(self) -> str:
        p_1 = self.prompt.invoke({"cover_letter": self.cover_letter, "job_description": self.job_description_context})
        out_1 = self.model.invoke(p_1)
        return schema.StrOutputParser().invoke(out_1)



def test_cover_letter_chef():
    for _ in range(10):
        runner = CoverLetterChef(
            prompt=prompt_factory(),
            model=ChatOllama(model=SELECTED_MODEL),
            cover_letter=Reader(file_location=ORIGINAL_COVERLETTER_PATH).read(),
            job_description_context=resolve_company_description(company_name="mbda")
        ).create_chain()
        print(runner)

def test_cover_letter_refinement_prompt_template():
    prompt_inputs = create_cover_letter_refinement_inputs("mbda_description")
    letter_prompt = cover_letter_refinement_prompt_template_gen()
    model = ChatOllama(model=SELECTED_MODEL)
    output = model.invoke(letter_prompt.invoke(prompt_inputs))
    str_result = schema.StrOutputParser().invoke(output)
    print(str_result)


if __name__ == "__main__":
    # test_cover_letter_chef()
    test_cover_letter_refinement_prompt_template()