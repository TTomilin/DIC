import gymnasium as gym


def main():
    # Create the MountainCar environment
    env = gym.make('MountainCar-v0', render_mode="human")

    truncated = terminated = False

    # Reset the environment to start a new episode
    env.reset()
    while not (truncated or terminated):
        # Take a random action
        action = env.action_space.sample()

        # Execute the action and get the new state, reward, done, and info
        observation, reward, truncated, terminated, info = env.step(action)
    print(f"Episode finished")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
