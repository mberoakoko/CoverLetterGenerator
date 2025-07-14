from langchain import prompts
from langchain.schema import  runnable
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

_LOG_FILE: pathlib.Path = pathlib.Path(__file__).parent.parent / "prompting.log"

logger = logging.getLogger(__name__)
logging.basicConfig(filename=str(_LOG_FILE), encoding='utf-8', level=logging.INFO)
logger.info('Started')


_SELECTED_MDOEL: str = "llama3.2"

class ChatFactory:
    @staticmethod
    def of(provider: str ) -> ChatOllama:
        return ChatOllama(model=str(provider), num_ctx=8192)

def cover_letter_refinement_chain(cover_letter_handle: str) -> runnable.RunnableSequence:
    prompt_inputs = create_cover_letter_refinement_inputs(company_name=cover_letter_handle)
    json_parser , letter_prompt_template = cover_letter_refinement_prompt_template_gen()
    model = ChatFactory.of(_SELECTED_MDOEL)
    return (runnable.RunnableLambda(lambda _: prompt_inputs)
                | letter_prompt_template
                | model
                | json_parser
             )


def cover_letter_generation_chain() -> runnable.RunnableSequence:
    cover_letter_prompt: prompts.PromptTemplate = cover_letter_generator()




def linguistic_checker_chain() -> runnable.RunnableSequence:
    ...



