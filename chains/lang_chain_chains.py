from langchain import prompts
from langchain.schema import  runnable
from langchain import schema
from langchain_core.runnables import RunnableParallel
from langchain_ollama import ChatOllama
from typing import Literal
import logging
import pathlib
from system_prompts.prompt_gen import (
    cover_letter_refinement_prompt_template_gen,
    create_cover_letter_refinement_inputs,
    cover_letter_generator,
    linguistic_style_checker_generator
)
from system_prompts.prompt_resources import (
    ORIGINAL_COVERLETTER_PATH,
    resolve_company_description, PromptResourceReader
)

_LOG_FILE: pathlib.Path = pathlib.Path(__file__).parent.parent / "prompting.log"

_logger = logging.getLogger(__name__)
logging.basicConfig(filename=str(_LOG_FILE), encoding='utf-8', level=logging.DEBUG)
_file_handler = logging.FileHandler(str(_LOG_FILE))
_stream_handler = logging.StreamHandler()
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_file_handler.setFormatter(_formatter)
_stream_handler.setFormatter(_formatter)

_logger.addHandler(_file_handler)
_logger.addHandler(_stream_handler)
_logger.info("Logger Initialized !")

_DESCRIPTION_PATH: pathlib.Path = (pathlib.Path(__file__).parent.parent / "data/job_descriptions").resolve()
_SELECTED_MDOEL: str = "llama3.2"

class ChatFactory:
    @staticmethod
    def of(provider: str ) -> ChatOllama:
        return ChatOllama(model=str(provider), num_ctx=8192)

def cover_letter_refinement_chain(company_job_description_name: str) -> runnable.RunnableSequence:
    prompt_inputs = create_cover_letter_refinement_inputs(company_job_description_name=company_job_description_name)
    json_parser , letter_prompt_template = cover_letter_refinement_prompt_template_gen()
    model = ChatFactory.of(_SELECTED_MDOEL)
    _logger.info("Running Cover letter refinement chain")
    return (runnable.RunnableLambda(lambda _: prompt_inputs)
                | letter_prompt_template
                | model
                | json_parser
             )


def cover_letter_generation_chain(cover_letter_handle: str, company_description_file_name) -> runnable.RunnableSerializable:
    refined_content_chain = cover_letter_refinement_chain(cover_letter_handle)
    cover_letter_prompt: prompts.PromptTemplate = cover_letter_generator()
    model = ChatFactory.of(_SELECTED_MDOEL)
    return (runnable.RunnableSequence(lambda _: {
                    "refined_context": refined_content_chain.invoke({}),
                    "job_description": resolve_company_description(company_description_file_name),
                    "original_cover_letter": PromptResourceReader(file_location=ORIGINAL_COVERLETTER_PATH).read()
                })
            | cover_letter_prompt
            | model
            | schema.StrOutputParser())



def linguistic_checker_chain() -> runnable.RunnableSequence:
    ...



