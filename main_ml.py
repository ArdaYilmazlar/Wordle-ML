from wordle_env import WordleEnv
from wordle_agent import WordleAgent

def main():
    # Define the word list for the Wordle game
    word_list = ['apple', 'grape', 'berry', 'melon', 'lemon']
    
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
        
        # Output training progress
        print(f"Episode {i_episode}, Total Reward: {total_reward}")
        
        # Optional: Save the model or log additional metrics
        
        if i_episode % 100 == 0:
            print(f"Episode {i_episode}: Epsilon = {agent.epsilon}")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
