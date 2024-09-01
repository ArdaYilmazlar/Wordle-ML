from flask import Flask, render_template, request, jsonify
from wordle_game import WordleGame, GuessStatus
from wordle_env import WordleEnv
from wordle_agent import WordleAgent
import torch
import config
import os
import time

class WordleController:
    def __init__(self, wordle_game: WordleGame):
        self.app = Flask(__name__)
        self.wordle_game = wordle_game
        self.setup_routes()

        # Load words from file
        self.word_list = self.load_words(config.VALID_WORDS_FILE_NAME)
        
        # Initialize the Wordle environment
        self.env = WordleEnv(self.word_list)
        
        # Define the size of the state and action spaces
        state_size = len(self.word_list[0]) * self.env.max_history
        action_size = len(self.word_list)
        
        # Initialize the RL agent
        self.agent = WordleAgent(state_size=state_size, action_size=action_size, seed=0)
        
        # Load pre-trained model weights if available
        self.load_model_weights(config.MODEL_PATH)
        
        # Initialize state
        self.state = self.env.reset()[0]  # Get the initial state from the environment

    def load_words(self, file_name):
        with open(file_name, 'r') as file:
            words = [line.strip() for line in file]
        return words

    def load_model_weights(self, model_path):
        if os.path.exists(model_path):
            # Load the saved model weights into qnetwork_local
            self.agent.qnetwork_local.load_state_dict(torch.load(os.path.join(model_path, "qnetwork_local.pth")))
            print(f"Loaded model weights from {model_path}")
        else:
            print(f"No pre-trained model found at {model_path}. Starting with a new model.")

    def setup_routes(self):
        # Set up the routes
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/attempt', 'attempt', self.attempt, methods=['POST'])
        self.app.add_url_rule('/new_game', 'new_game', self.new_game, methods=['GET'])
        self.app.add_url_rule('/make_guess', 'make_guess', self.make_guess, methods=['POST'])
        self.app.add_url_rule('/auto_play', 'auto_play', self.auto_play, methods=['POST'])

    def index(self):
        # Serve the main page (View)
        return render_template('index.html')

    def attempt(self):
        # Handle a word guess and return the result to the View
        word = request.json.get('word', '').lower()
        try:
            result = self.wordle_game.make_attempt(word)
            return jsonify(result=result, game_over=self.wordle_game.is_game_over())
        except ValueError as e:
            return jsonify(error=str(e)), 400

    def new_game(self):
        # Reset the environment and start a new game
        self.state = self.env.reset()[0]
        return jsonify({"message": "New game started!"})

    def make_guess(self):
        # Model makes a guess based on the current state
        action = self.agent.act(self.state)  # Model selects an action
        next_state, reward, done, _, _ = self.env.step(action)  # Apply the guess
        
        # Update the environment state
        self.state = next_state
        
        # Get the guessed word from the word list
        guess_word = self.word_list[action]
        
        # Check if the guess is correct
        success = (guess_word == self.wordle_game.target_word)
        
        # Retrieve status for each letter
        result = self.wordle_game.make_attempt(guess_word)
        status = [{'char': char, 'status': status} for char, status in result]
        
        response = {
            "guess": guess_word,
            "status": status,
            "reward": reward,
            "done": done or success,  # End the game if the guess is correct
            "success": success,
            "target_word": self.wordle_game.target_word
        }

        return jsonify(response)

    def auto_play(self):
        # Auto-play the game until it's done
        game_log = []
        done = False
        
        while not done:
            action = self.agent.act(self.state)
            next_state, reward, done, _, _ = self.env.step(action)
            self.state = next_state
            
            guess_word = self.word_list[action]
            
            # Check if the guess is correct
            success = (guess_word == self.wordle_game.target_word)
            
            result = self.wordle_game.make_attempt(guess_word)
            status = [{'char': char, 'status': status} for char, status in result]
            
            game_log.append({
                "guess": guess_word,
                "status": status,
                "reward": reward,
                "done": done or success,  # End the game if the guess is correct
                "success": success,
                "target_word": self.wordle_game.target_word
            })
            
            if success:  # If the correct word is guessed, end the loop
                done = True
            
            time.sleep(0.5)  # Add delay for visualization
        
        return jsonify(game_log)


    def run(self, host="0.0.0.0", port=3333, debug=True):
        # Start the Flask server
        self.app.run(host=host, port=port, debug=debug)
