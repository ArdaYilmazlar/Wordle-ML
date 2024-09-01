from random import randint
from typing import List, Dict
from enums import GuessStatus

class WordleGame:
    # Class Variables
    word_list: List[str]
    attempts: List[str]
    target_word: str

    def __init__(self, word_list: List[str]):
        self.word_list = word_list
        self.target_word = self._select_random_word()
        self.target_word = list()
    
    def _select_random_word(self) -> str:
        return self.word_list[randint(0, len(self.word_list))]
    
    def make_attempt(self, attempted_word: str) -> Dict[str, int]:
        if attempted_word not in self.word_list:
            raise ValueError("'{attempted_word}' is not a valid word.")
        elif attempted_word in self.attempts:
            raise ValueError("'{attempted_word}' have been guessed before.")
        
        result: Dict[str, int] = dict()
        
        for i, char in enumerate(attempted_word):
            if char == self.target_word[i]:
                result[char] = GuessStatus.CORRECT.value
            elif char in self.target_word:
                result[char] = GuessStatus.PRESENT.value
            else:
                result[char] = GuessStatus.ABSENT.value
            
        self.attempts.append(attempted_word)
        return result

