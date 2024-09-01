from wordle_env import WordleEnv
from wordle_agent import WordleAgent
import os
from datetime import datetime
import torch
import config
import matplotlib.pyplot as plt

def load_words(file_name):
    with open(file_name, 'r') as file:
        words = [line.strip() for line in file]
        return words

def main(verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("saved_models", start_time)
    os.makedirs(save_dir, exist_ok=True)
    if verbose:
        print(f"Saving models to directory: {save_dir}")

    # Define the word list for the Wordle game
    word_list = load_words(config.VALID_WORDS_FILE_NAME)
    
    # Initialize the Wordle environment
    env = WordleEnv(word_list)
    
    # Define the size of the state and action spaces
    state_size = len(word_list[0]) * env.max_history
    action_size = len(word_list)   # Number of possible words in the word list
    
    # Initialize the RL agent
    agent = WordleAgent(state_size=state_size, action_size=action_size, seed=0)
    
    # Define the number of episodes for training
    num_episodes = 10000
    
    # Lists to track rewards, success counts, and guess counts
    rewards = []
    successes = []
    guess_counts = []

    # Training loop
    for i_episode in range(1, num_episodes + 1):
        # Reset the environment to start a new episode
        state, _ = env.reset()  # Gymnasium returns (observation, info), so we take the observation
        
        total_reward = 0
        success = False
        guess_count = 0
        
        for t in range(env.max_attempts):
            # Agent selects an action based on the current state
            action = agent.act(state)
            
            # Environment returns the next state, reward, and done flag
            next_state, reward, done, _, _ = env.step(action)
            
            # Agent steps through the environment and learns from the experience
            agent.step(state, action, reward, next_state, done)
            
            # Move to the next state
            state = next_state
            total_reward += reward
            guess_count += 1
            
            # Check if the agent successfully guessed the word
            if done:
                # Success if the guessed word matches the target word
                if reward > 0 and next_state.tolist().count(2) == len(env.word_list[0]):
                    success = True
                break
        
        # Record the total reward, success, and number of guesses for this episode
        rewards.append(total_reward)
        successes.append(success)
        guess_counts.append(guess_count if success else None)

        # Decay the exploration rate (epsilon) after each episode
        agent.decay_epsilon()

        if i_episode % 5000 == 0:
            snapshot_dir = os.path.join(save_dir, f"step_{i_episode}")
            os.makedirs(snapshot_dir, exist_ok=True)
            torch.save(agent.qnetwork_local.state_dict(), os.path.join(snapshot_dir, "qnetwork_local.pth"))
            torch.save(agent.qnetwork_target.state_dict(), os.path.join(snapshot_dir, "qnetwork_target.pth"))
            if verbose:
                print(f"Saved model snapshot at episode {i_episode} in {snapshot_dir}")
        
        # Output training progress
        if verbose:
            print(f"Episode {i_episode}, Total Reward: {total_reward}")
        
        if verbose and i_episode % 100 == 0:
            print(f"Episode {i_episode}: Epsilon = {agent.epsilon}")

    print("Training complete!")

    # Plot the results
    plot_results(successes, guess_counts)

def plot_results(successes, guess_counts):
    episodes = range(1, len(successes) + 1)
    
    # Calculate success rate in percentages for every 100 episodes
    success_rate_percentages = []
    for i in range(0, len(successes), 100):
        success_rate = sum(successes[i:i + 100]) / 100 * 100
        success_rate_percentages.append(success_rate)
    
    # Filter out the guess counts for successful episodes only
    successful_guess_counts = [g for g in guess_counts if g is not None]
    
    plt.figure(figsize=(12, 5))

    # Plot success rate over episodes (as percentage)
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(success_rate_percentages) + 1), success_rate_percentages, label="Success Rate (%)", color="green")
    plt.xlabel("Episode Batches (100 per batch)")
    plt.ylabel("Success Rate (%)")
    plt.title("Success Rate (%) per 100 Episodes")
    plt.legend()

    # Plot distribution of guesses required in successful episodes
    plt.subplot(1, 2, 2)
    plt.hist(successful_guess_counts, bins=range(1, max(successful_guess_counts)+2), alpha=0.75, color="blue", edgecolor="black")
    plt.xlabel("Number of Guesses")
    plt.ylabel("Frequency")
    plt.title("Distribution of Guesses per Successful Episode")
    plt.xticks(range(1, max(successful_guess_counts)+1))
    plt.legend(["Guesses Distribution"])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(verbose=True)  # Set verbose=False to disable print statements
