from random import randint
from typing import List, Dict

class WordleGame:
    # Class Variables
    word_list: List[str]
    attempts: List[str]
    target_word: str
    
    def __init__(self, word_list: List[str]):
        self.word_list = word_list
        self.target_word = self._select_random_word()
        self
    
    def _select_random_word(self) -> str:
        return self.word_list[randint(0, len(self.word_list))]
