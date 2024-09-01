from wordle_game import WordleGame
import config

def load_words(file_name):
    with open(file_name, 'r') as file:
        words = [line.strip() for line in file]
        return words

if __name__ == "__main__":
    word_list = load_words(config.VALID_WORDS_FILE_NAME)
    wordle_game = WordleGame(word_list)
    result = wordle_game.make_attempt("genre")
    for char, status in result.items():
        print(f"Character: {char}, Status: {status}")
    print(wordle_game.target_word)
