from chains.lang_chain_chains import *
import unittest

_CHAT_COMPLETIONS: ChatOllama = ChatFactory.of("llama3.5")
_TEST_COVER_LETTER_HANDLE: str = "mbda_description"
class ChainTests(unittest.TestCase):
    def test_cover_letter_refinement_chain(self) -> None:
        refinement_chain = cover_letter_refinement_chain(_TEST_COVER_LETTER_HANDLE)
        self.assertIsNotNone(refinement_chain.invoke({}), "Something went wrong with refining cover letter")

    def test_letter_generation_chain(self) -> None:
        cover_letter_chain = cover_letter_generation_chain(_TEST_COVER_LETTER_HANDLE, )
        self.assertIsNotNone(cover_letter_chain.invoke({}), "Something went wrong with cover letter generation")

    # def test_linguistic_checher_chain(self) -> None:
        # instance_linguistic_checker = linguistic_checker_chain()
        # self.assertIsNotNone(instance_linguistic_checker.invoke({}), "Something is wrong with the linguistic checker")


if __name__ == '__main__':
    unittest.main()
