from flask import Flask, render_template, request, jsonify
from wordle_game import WordleGame

class WordleController:
    def __init__(self, wordle_game: WordleGame):
        self.app = Flask(__name__)
        self.wordle_game = wordle_game
        self.setup_routes()

    def setup_routes(self):
        # Set up the routes
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/attempt', 'attempt', self.attempt, methods=['POST'])

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

    def run(self, host="0.0.0.0", port=3333, debug=True):
        # Start the Flask server
        self.app.run(host=host, port=port, debug=debug)