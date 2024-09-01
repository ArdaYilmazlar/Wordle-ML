import numpy as np
import gymnasium as gym
from gymnasium import spaces
from wordle_game import WordleGame, GuessStatus

class WordleEnv(gym.Env):
    def __init__(self, word_list, max_history=6):
        super(WordleEnv, self).__init__()
        
        self.word_list = word_list
        self.game = WordleGame(word_list)
        self.max_attempts = self.game.max_attempts
        self.max_history = max_history  # Maximum number of guesses to consider
        
        # Store guess-status pairs
        self.guess_status_history = []

        # Action space: the index of a word in the word_list
        self.action_space = spaces.Discrete(len(word_list))
        
        # Observation space: we'll track up to max_history guesses, each with len(target_word) statuses
        self.observation_space = spaces.MultiDiscrete([3] * len(self.game.target_word) * self.max_history)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game = WordleGame(self.word_list)
        self.guess_status_history = []  # Reset the history
        return self._get_observation(), {}

    def step(self, action):
        guess_word = self.word_list[action]
        result = self.game.make_attempt(guess_word)
        
        # Save the guess-status pair in the history
        self.guess_status_history.append(result)
        
        # Calculate rewards based on the new strategy
        reward = self.calculate_reward(result)
        
        # Convert result to observation
        observation = self._get_observation()
        
        done = self.game.is_game_over()
        return observation, reward, done, False, {}

    def calculate_reward(self, result):
        reward = 0.0
        
        # Reward based on the status of each character
        for char, status in result:
            if status == GuessStatus.CORRECT.value:
                reward += 1.0  # High reward for correct letters
            elif status == GuessStatus.PRESENT.value:
                reward += 0.5  # Medium reward for present letters
            else:
                reward -= 0.1  # Smaller penalty for absent letters to not be too discouraging
        
        # Penalty for the number of guesses made so far (encourages faster guessing)
        reward -= len(self.guess_status_history) * 0.1
        
        # Large reward for guessing the entire word correctly, scaled by how quickly it was found
        if all(status == GuessStatus.CORRECT.value for _, status in result):
            remaining_attempts = 7 - len(self.guess_status_history)  # Assuming max 6 attempts
            reward += 5.0 * (1 + remaining_attempts * 0.5)  # Scaled reward to emphasize early success
        
        return reward


    def _get_observation(self):
        # Create an observation with the history of guesses and their statuses
        observation = np.zeros(len(self.game.target_word) * self.max_history)
        
        # Fill the observation with previous guesses and their statuses
        for i, attempt_result in enumerate(self.guess_status_history[-self.max_history:]):
            for j, (char, status) in enumerate(attempt_result):
                observation[i * len(self.game.target_word) + j] = status
        
        return observation

    def render(self, mode='human'):
        for attempt in self.game.attempts:
            print(attempt)