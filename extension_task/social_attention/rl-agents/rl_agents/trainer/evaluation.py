import datetime
import json
import logging
import os
import time
from multiprocessing.pool import Pool
from pathlib import Path
import numpy as np
from tensorboardX import SummaryWriter
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

import rl_agents.trainer.logger
from rl_agents.agents.common.factory import load_environment, load_agent
from rl_agents.agents.common.graphics import AgentGraphics
from rl_agents.agents.common.memory import Transition
from rl_agents.utils import near_split, zip_with_singletons
from rl_agents.configuration import serialize
from rl_agents.trainer.graphics import RewardViewer

logger = logging.getLogger(__name__)

def capped_cubic_video_schedule(episode_id):
    return episode_id < 1000 and int(round(episode_id ** (1/3))) ** 3 == episode_id
    # return episode_id % 10 == 0

def safe_capture_frame(self):
    if not self.recording:
        return
    frame = self.env.render()
    if isinstance(frame, list):
        if len(frame) == 0:
            return
        self.render_history += frame
        frame = frame[-1]
    if isinstance(frame, np.ndarray):
        self.recorded_frames.append(frame)

RecordVideo._capture_frame = safe_capture_frame


class Evaluation(object):
    """
        The evaluation of an agent interacting with an environment to maximize its expected reward.
    """

    OUTPUT_FOLDER = 'out'
    SAVED_MODELS_FOLDER = 'saved_models'
    RUN_FOLDER = 'run_{}_{}'
    METADATA_FILE = 'metadata.{}.json'
    LOGGING_FILE = 'logging.{}.log'

    def __init__(self,
                 env,
                 agent,
                 directory=None,
                 run_directory=None,
                 num_episodes=1000,
                 training=True,
                 sim_seed=None,
                 recover=None,
                 display_env=True,
                 display_agent=True,
                 display_rewards=True,
                 close_env=True,
                 step_callback_fn=None):
        """

        :param env: The environment to be solved, possibly wrapping an AbstractEnv environment
        :param AbstractAgent agent: The agent solving the environment
        :param Path directory: Workspace directory path
        :param Path run_directory: Run directory path
        :param int num_episodes: Number of episodes run
        !param training: Whether the agent is being trained or tested
        :param sim_seed: The seed used for the environment/agent randomness source
        :param recover: Recover the agent parameters from a file.
                        - If True, it the default latest save will be used.
                        - If a string, it will be used as a path.
        :param display_env: Render the environment, and have a monitor recording its videos
        :param display_agent: Add the agent graphics to the environment viewer, if supported
        :param display_rewards: Display the performances of the agent through the episodes
        :param close_env: Should the environment be closed when the evaluation is closed
        :param step_callback_fn: A callback function called after every environment step. It takes the following
               arguments: (episode, env, agent, transition, writer).

        """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.training = training
        self.sim_seed = sim_seed if sim_seed is not None else np.random.randint(0, 1e6)
        self.close_env = close_env
        self.display_env = display_env
        self.step_callback_fn = step_callback_fn

        self.directory = Path(directory or self.default_directory)
        self.run_directory = self.directory / (run_directory or self.default_run_directory)
        self.wrapped_env = RecordVideo(env,
                                       self.run_directory,
                                       episode_trigger=(None if self.display_env else lambda e: False))
        try:
            self.wrapped_env.unwrapped.set_record_video_wrapper(self.wrapped_env)
        except AttributeError:
            pass
        self.wrapped_env = RecordEpisodeStatistics(self.wrapped_env)
        self.episode = 0
        self.writer = SummaryWriter(str(self.run_directory))
        self.agent.set_writer(self.writer)
        self.agent.evaluation = self
        self.write_logging()
        self.write_metadata()
        self.filtered_agent_stats = 0
        self.best_agent_stats = -np.inf, 0

        self.recover = recover
        if self.recover:
            self.load_agent_model(self.recover)

        if display_agent:
            try:
                # Render the agent within the environment viewer, if supported
                self.env.render()
                self.env.unwrapped.viewer.directory = self.run_directory
                self.env.unwrapped.viewer.set_agent_display(
                    lambda agent_surface, sim_surface: AgentGraphics.display(self.agent, agent_surface, sim_surface))
                self.env.unwrapped.viewer.directory = self.run_directory
            except AttributeError:
                logger.info("The environment viewer doesn't support agent rendering.")
        self.reward_viewer = None
        if display_rewards:
            self.reward_viewer = RewardViewer()
        self.observation = None

    def train(self):
        print("in training")
        self.training = True
        results = self.run_episodes_train()
        self.close()
        return results

    def test(self):
        """
        Test the agent.

        If applicable, the agent model should be loaded before using the recover option.
        """
        self.training = False
        if self.display_env:
            self.wrapped_env.episode_trigger = lambda e: True
        try:
            self.agent.eval()
        except AttributeError:
            pass
        print(self.agent.exploration_policy)
        self.run_episodes_test()
        self.close()

    def run_episodes_test(self):

        for self.episode in range(self.num_episodes):
            # Run episode
            terminal = False
            self.reset(seed=self.episode)
            rewards = []
            start_time = time.time()
            while not terminal:
                # Step until a terminal step is reached
                reward, terminal = self.step()
                rewards.append(reward)

                # Catch interruptions
                try:
                    if self.env.unwrapped.done:
                        break
                except AttributeError:
                    pass

            # End of episode
            duration = time.time() - start_time
            self.after_all_episodes(self.episode, rewards, duration)
            self.after_some_episodes(self.episode, rewards)

    def run_episodes_train(self, eval_interval=50, eval_episodes=10):
        episode_rewards, episode_lengths = [], []
        all_losses, losses_steps = [], []
        epsilons, returns = [], []
        
        # Initialisation des listes pour les résultats d'évaluation
        eval_steps, eval_reward_means, eval_reward_stds = [], [], []
        eval_length_means, eval_length_std = [], []

        for self.episode in range(self.num_episodes):
            # --- PHASE D'ENTRAÎNEMENT ---
            terminal = False
            self.reset(seed=self.episode)
            rewards = []
            start_time = time.time()
            
            while not terminal:
                # Step classique avec enregistrement dans la mémoire de l'agent
                reward, terminal = self.step()
                rewards.append(reward)
                
                # Récupération de la perte si l'agent est en train de s'optimiser
                if hasattr(self.agent, 'loss'):
                    all_losses.append(self.agent.loss)
                    losses_steps.append(self.episode)

            # Enregistrement de l'epsilon actuel avant la fin de l'épisode (pour le log)
            if hasattr(self.agent, 'exploration_policy'):
                epsilons.append(getattr(self.agent.exploration_policy, 'epsilon', 0))

            # Stats de fin d'épisode d'entraînement
            duration = time.time() - start_time
            episode_rewards.append(sum(rewards))
            episode_lengths.append(len(rewards))
            
            gamma = self.agent.config.get("gamma", 1)
            returns.append(sum(r * (gamma ** t) for t, r in enumerate(rewards)))
            
            self.after_all_episodes(self.episode, rewards, duration)
            self.after_some_episodes(self.episode, rewards)

            # --- PHASE D'ÉVALUATION PÉRIODIQUE ---
            if self.episode > 0 and self.episode % eval_interval == 0:
                logger.info("Starting evaluation")

                # Passage en Greedy via votre fonction eval()
                self.agent.eval()
                
                # 2. Désactivation vidéo (recherche récursive du wrapper RecordVideo)
                video_wrapper = None
                curr_env = self.wrapped_env
                while hasattr(curr_env, "env"):
                    if isinstance(curr_env, RecordVideo):
                        video_wrapper = curr_env
                        break
                    curr_env = curr_env.env
                
                was_recording = False
                if video_wrapper:
                    was_recording = video_wrapper.recording
                    video_wrapper.recording = False

                eval_rewards, eval_lengths = [], []
                for i in range(eval_episodes):
                    obs, info = self.wrapped_env.reset(seed=self.sim_seed + self.episode + i + 1000)
                    self.observation = obs
                    done = False
                    ep_rew, ep_len = 0, 0
                    while not done:
                        action = self.agent.act(self.observation, step_exploration_time=False) # act() ne stocke rien en mémoire
                        obs, reward, terminated, truncated, info = self.wrapped_env.step(action)
                        self.observation = obs
                        done = terminated or truncated
                        ep_rew += reward
                        ep_len += 1
                    eval_rewards.append(ep_rew)
                    eval_lengths.append(ep_len)

                # 3. RESTAURATION de l'état d'entraînement
                self.agent.train(training=True) # Réinstalle la factory d'exploration
                
                if video_wrapper:
                    video_wrapper.recording = was_recording
                
                # Logs des résultats de validation
                eval_steps.append(self.episode)
                eval_reward_means.append(np.mean(eval_rewards))
                eval_reward_stds.append(np.std(eval_rewards))
                eval_length_means.append(np.mean(eval_lengths))
                eval_length_std.append(np.std(eval_lengths))
                
                logger.info(f"Evaluation @ Episode {self.episode}: Mean Reward: {eval_reward_means[-1]:.2f}")

        return (
            episode_rewards, all_losses, losses_steps,
            eval_steps, eval_reward_means, eval_reward_stds, 
            episode_lengths, eval_length_means, eval_length_std, 
            epsilons, returns
        )

    def step(self):
        """
            Plan a sequence of actions according to the agent policy, and step the environment accordingly.
        """
        # Query agent for actions sequence
        actions = self.agent.plan(self.observation)
        if not actions:
            raise Exception("The agent did not plan any action")

        # Forward the actions to the environment viewer
        try:
            self.env.unwrapped.viewer.set_agent_action_sequence(actions)
        except AttributeError:
            pass

        # Step the environment
        previous_observation, action = self.observation, actions[0]
        transition = self.wrapped_env.step(action)
        self.observation, reward, done, truncated, info = transition
        terminal = done or truncated

        # Call callback
        if self.step_callback_fn is not None:
            self.step_callback_fn(self.episode, self.wrapped_env, self.agent, transition, self.writer)

        # Record the experience.
        try:
            self.agent.record(previous_observation, action, reward, self.observation, done, info)
        except NotImplementedError:
            pass

        return reward, terminal

    def save_agent_model(self, identifier, do_save=True):
        # Create the folder if it doesn't exist
        permanent_folder = self.directory / self.SAVED_MODELS_FOLDER
        os.makedirs(permanent_folder, exist_ok=True)

        episode_path = None
        if do_save:
            episode_path = Path(self.run_directory) / "checkpoint-{}.tar".format(identifier)
            try:
                self.agent.save(filename=permanent_folder / "latest.tar")
                episode_path = self.agent.save(filename=episode_path)
                if episode_path:
                    logger.info("Saved {} model to {}".format(self.agent.__class__.__name__, episode_path))
            except NotImplementedError:
                pass
        return episode_path

    def load_agent_model(self, model_path):
        if model_path is True:
            model_path = self.directory / self.SAVED_MODELS_FOLDER / "latest.tar"
            # model_path = self.directory / "run_20260404-171122_28896/checkpoint-729.tar"
            # model_path = "C:/Users/louis/Documents/3A/RL Apprentissage par Renforcement/projet_rl/Reinforcement_learning_highway/extension_task/social_attention/out_true/run_20260403-095139_39712/checkpoint-best.tar"
        if isinstance(model_path, str):
            model_path = Path(model_path)
            if not model_path.exists():
                model_path = self.directory / self.SAVED_MODELS_FOLDER / model_path
        try:
            model_path = self.agent.load(filename=model_path)
            print(f"loading {model_path}")
            if model_path:
                logger.info("Loaded {} model from {}".format(self.agent.__class__.__name__, model_path))
        except FileNotFoundError:
            logger.warning("No pre-trained model found at the desired location.")
        except NotImplementedError:
            pass

    def after_all_episodes(self, episode, rewards, duration):
        rewards = np.array(rewards)
        gamma = self.agent.config.get("gamma", 1)
        self.writer.add_scalar('episode/length', len(rewards), episode)
        self.writer.add_scalar('episode/total_reward', sum(rewards), episode)
        self.writer.add_scalar('episode/return', sum(r*gamma**t for t, r in enumerate(rewards)), episode)
        self.writer.add_scalar('episode/fps', len(rewards) / max(duration, 1e-6), episode)
        self.writer.add_histogram('episode/rewards', rewards, episode)
        logger.info("Episode {} score: {:.1f}".format(episode, sum(rewards)))

    def after_some_episodes(self, episode, rewards,
                            best_increase=1.1,
                            episodes_window=50):
        if capped_cubic_video_schedule(episode):
            # Save the model
            if self.training:
                self.save_agent_model(episode)

        if self.training:
            # Save best model so far, averaged on a window
            best_reward, best_episode = self.best_agent_stats
            self.filtered_agent_stats += 1 / episodes_window * (np.sum(rewards) - self.filtered_agent_stats)
            if self.filtered_agent_stats > best_increase * best_reward \
                    and episode >= best_episode + episodes_window:
                self.best_agent_stats = (self.filtered_agent_stats, episode)
                self.save_agent_model("best")

    @property
    def default_directory(self):
        return Path(self.OUTPUT_FOLDER) / self.env.unwrapped.__class__.__name__ / self.agent.__class__.__name__

    @property
    def default_run_directory(self):
        return self.RUN_FOLDER.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), os.getpid())

    def write_metadata(self):
        metadata = dict(env=serialize(self.env), agent=serialize(self.agent))
        file_infix = '{}.{}'.format(id(self.wrapped_env), os.getpid())
        file = self.run_directory / self.METADATA_FILE.format(file_infix)
        with file.open('w') as f:
            json.dump(metadata, f, sort_keys=True, indent=4)

    def write_logging(self):
        file_infix = '{}.{}'.format(id(self.wrapped_env), os.getpid())
        rl_agents.trainer.logger.configure()
        rl_agents.trainer.logger.add_file_handler(self.run_directory / self.LOGGING_FILE.format(file_infix))

    def reset(self, seed=0):
        seed = self.sim_seed + seed if self.sim_seed is not None else None
        self.observation, info = self.wrapped_env.reset()
        self.agent.seed(seed)  # Seed the agent with the main environment seed
        self.agent.reset()

    def close(self):
        """
            Close the evaluation.
        """
        if self.training:
            self.save_agent_model("final")
        self.wrapped_env.close()
        self.writer.close()
        if self.close_env:
            self.env.close()
