from random import randint
from typing import List, Tuple
from enums import GuessStatus

class WordleGame:
    # Class Variables
    word_list: List[str]
    attempts: List[str]
    target_word: str
    max_attempts: int

    def __init__(self, word_list: List[str], max_attempts: int = 6):
        self.word_list = word_list
        self.target_word = self._select_random_word()
        self.attempts = list()
        self.max_attempts = max_attempts
    
    def _select_random_word(self) -> str:
        return self.word_list[randint(0, len(self.word_list))]
    
    def make_attempt(self, attempted_word: str) -> List[Tuple[str, int]]:
        if attempted_word not in self.word_list:
            raise ValueError(f"'{attempted_word}' is not a valid word.")
        elif attempted_word in self.attempts:
            raise ValueError(f"'{attempted_word}' have been guessed before.")
        
        result: List[Tuple[str, int]] = list()
        
        for i, char in enumerate(attempted_word):
            if char == self.target_word[i]:
                result.append((char, GuessStatus.CORRECT.value))
            elif char in self.target_word:
                result.append((char, GuessStatus.PRESENT.value))
            else:
                result.append((char, GuessStatus.ABSENT.value))
            
        self.attempts.append(attempted_word)
        return result
    
    def is_game_over(self) -> bool:
        return len(self.attempts) >= self.max_attempts or self.attempts[-1] == self.target_word

