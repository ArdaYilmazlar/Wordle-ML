from wordle_game import WordleGame
from wordle_controller import WordleController
import config

def load_words(file_name):
    with open(file_name, 'r') as file:
        words = [line.strip() for line in file]
        return words

if __name__ == "__main__":
    valid_word_list = load_words(config.VALID_WORDS_FILE_NAME)
    wordle_game = WordleGame(valid_word_list)
    controller = WordleController(wordle_game)
    # Run the web server
    controller.run(config.APP_IP, config.APP_PORT, False)
