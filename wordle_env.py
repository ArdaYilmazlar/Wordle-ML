import numpy as np
import gymnasium as gym
from gymnasium import spaces
from wordle_game import WordleGame

class WordleEnv(gym.Env):
    def __init__(self, word_list: list[str], max_attempts: int = 6):
        super(WordleEnv, self).__init__()
        
        self.word_list = word_list
        self.game = WordleGame(word_list, max_attempts)
        self.max_attempts = max_attempts
        
        # Action space: the index of a word in the word_list
        self.action_space = spaces.Discrete(len(word_list))
        
        # Observation space: each character can be CORRECT, PRESENT, or ABSENT
        self.observation_space = spaces.MultiDiscrete([3] * len(self.game.target_word))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game = WordleGame(self.word_list)
        return self._get_observation(), {}

    def step(self, action):
        guess_word = self.word_list[action]
        result = self.game.make_attempt(guess_word)
        
        # Convert result to observation (0 for absent, 1 for present, 2 for correct)
        observation = self._get_observation(result)
        
        # Define reward
        if self.game.is_game_over():
            if guess_word == self.game.target_word:
                reward = 1.0  # Reward for correctly guessing the word
            else:
                reward = -1.0  # Penalty for failing to guess within the allowed attempts
        else:
            reward = -0.1  # Small penalty for each guess to encourage quicker guesses
        
        done = self.game.is_game_over()
        return observation, reward, done, False, {}

    def _get_observation(self, result=None):
        if result is None:
            return np.zeros(len(self.game.target_word))
        
        observation = np.zeros(len(self.game.target_word))
        for i, (char, status) in enumerate(result):
            observation[i] = status  # 0: ABSENT, 1: PRESENT, 2: CORRECT
        return observation

    def render(self, mode='human'):
        for attempt in self.game.attempts:
            print(attempt)

# Example usage
if __name__ == "__main__":
    word_list = ['apple', 'grape', 'berry', 'melon', 'lemon']
    env = WordleEnv(word_list)
    env.reset()
    env.render()
