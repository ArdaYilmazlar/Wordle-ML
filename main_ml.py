from wordle_env import WordleEnv
from wordle_agent import WordleAgent
import os
from datetime import datetime
import torch
import config

def load_words(file_name):
    with open(file_name, 'r') as file:
        words = [line.strip() for line in file]
        return words


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("saved_models", start_time)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving models to directory: {save_dir}")

    # Define the word list for the Wordle game
    word_list = load_words(config.VALID_WORDS_FILE_NAME)
    
    # Initialize the Wordle environment
    env = WordleEnv(word_list)
    
    # Define the size of the state and action spaces
    state_size = len(word_list[0])  # Length of a word (e.g., 5 for "apple")
    action_size = len(word_list)   # Number of possible words in the word list
    
    # Initialize the RL agent
    agent = WordleAgent(state_size=state_size, action_size=action_size, seed=0)
    
    # Define the number of episodes for training
    num_episodes = 1000
    
    # Training loop
    for i_episode in range(1, num_episodes + 1):
        # Reset the environment to start a new episode
        state, _ = env.reset()  # Gymnasium returns (observation, info), so we take the observation
        
        total_reward = 0
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
            
            # Break the loop if the episode is done
            if done:
                break
        
        # Decay the exploration rate (epsilon) after each episode
        agent.decay_epsilon()
        if i_episode % 500 == 0:
            snapshot_dir = os.path.join(save_dir, f"step_{i_episode}")
            os.makedirs(snapshot_dir, exist_ok=True)
            torch.save(agent.qnetwork_local.state_dict(), os.path.join(snapshot_dir, "qnetwork_local.pth"))
            torch.save(agent.qnetwork_target.state_dict(), os.path.join(snapshot_dir, "qnetwork_target.pth"))
            print(f"Saved model snapshot at episode {i_episode} in {snapshot_dir}")
        # Output training progress
        print(f"Episode {i_episode}, Total Reward: {total_reward}")
        
        # Optional: Save the model or log additional metrics
        
        if i_episode % 100 == 0:
            print(f"Episode {i_episode}: Epsilon = {agent.epsilon}")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
