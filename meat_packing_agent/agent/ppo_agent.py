"""
PPO Agent for Meat Packing Optimization

This module implements a PPO (Proximal Policy Optimization) agent that learns
to optimally place meat slices in cubes. The agent uses the stable-baselines3
library for the core RL algorithm.
"""

from typing import Optional, Dict, Any, Callable, List, Tuple
import numpy as np
from pathlib import Path
import json

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from meat_packing_agent.env.cube_environment import MeatPackingEnv


class TrainingMetricsCallback(BaseCallback):
    """Callback for logging training metrics."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.fill_percentages: List[float] = []
        self.flatness_scores: List[float] = []
    
    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None:
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][idx]
                    if "episode" in info:
                        self.episode_rewards.append(info["episode"]["r"])
                        self.episode_lengths.append(info["episode"]["l"])
                    if "fill_percentage" in info:
                        self.fill_percentages.append(info["fill_percentage"])
                    if "flatness" in info:
                        self.flatness_scores.append(info["flatness"])
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        return {
            "mean_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            "mean_fill_percentage": np.mean(self.fill_percentages[-100:]) if self.fill_percentages else 0,
            "mean_flatness": np.mean(self.flatness_scores[-100:]) if self.flatness_scores else 0,
            "total_episodes": len(self.episode_rewards)
        }


class MeatPackingAgent:
    """
    RL Agent for meat packing optimization.
    
    This agent uses PPO to learn optimal placement strategies for meat slices
    in a cube container. It can be trained, saved, loaded, and used for inference.
    """
    
    def __init__(
        self,
        env: Optional[MeatPackingEnv] = None,
        model_path: Optional[str] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        verbose: int = 1
    ):
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.verbose = verbose
        
        self.env = env
        self.vec_env: Optional[VecNormalize] = None
        self.model: Optional[PPO] = None
        self.metrics_callback: Optional[TrainingMetricsCallback] = None
        
        if model_path:
            self.load(model_path)
        elif env:
            self._setup_model()
    
    def _setup_model(self):
        """Initialize the PPO model with the environment."""
        if self.env is None:
            raise ValueError("Environment must be provided to setup model")
        
        def make_env():
            env = MeatPackingEnv()
            return Monitor(env)
        
        self.vec_env = DummyVecEnv([make_env])
        self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True)
        
        self.model = PPO(
            "MultiInputPolicy",
            self.vec_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            verbose=self.verbose,
            tensorboard_log="./logs/tensorboard/"
        )
    
    def train(
        self,
        total_timesteps: int = 100000,
        eval_freq: int = 10000,
        save_path: Optional[str] = None,
        callback: Optional[BaseCallback] = None
    ) -> Dict[str, Any]:
        """
        Train the agent.
        
        Args:
            total_timesteps: Total number of environment steps to train for
            eval_freq: Frequency of evaluation during training
            save_path: Path to save the best model
            callback: Additional callback for training
            
        Returns:
            Dictionary containing training metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized. Provide an environment or load a model.")
        
        self.metrics_callback = TrainingMetricsCallback(verbose=self.verbose)
        
        callbacks = [self.metrics_callback]
        
        if save_path:
            eval_env = DummyVecEnv([lambda: Monitor(MeatPackingEnv())])
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        if callback:
            callbacks.append(callback)
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        return self.metrics_callback.get_metrics()
    
    def predict(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = True
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        Predict the best action for a given observation.
        
        Args:
            observation: Current environment observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, action_probabilities)
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return int(action), None
    
    def get_placement_decision(
        self,
        height_map: np.ndarray,
        slice_info: Dict[str, float],
        fill_percentage: float
    ) -> Dict[str, Any]:
        """
        Get a placement decision for a meat slice.
        
        Args:
            height_map: Current height map of the cube (normalized)
            slice_info: Dictionary with width, length, thickness of current slice
            fill_percentage: Current fill percentage of the cube
            
        Returns:
            Dictionary containing placement coordinates and gripper pattern
        """
        observation = {
            "height_map": height_map.flatten().astype(np.float32),
            "current_slice": np.array([
                slice_info["width"] / 200.0,
                slice_info["length"] / 200.0,
                slice_info["thickness"] / 40.0
            ], dtype=np.float32),
            "fill_percentage": np.array([fill_percentage / 100.0], dtype=np.float32)
        }
        
        action, _ = self.predict(observation)
        
        w_voxels = int(np.sqrt(len(height_map)))
        l_voxels = w_voxels
        resolution = 5.0
        
        x_pos = action // (l_voxels * 4)
        remainder = action % (l_voxels * 4)
        y_pos = remainder // 4
        rotation = remainder % 4
        
        x_mm = x_pos * resolution
        y_mm = y_pos * resolution
        
        masked_heights = height_map.reshape(w_voxels, l_voxels)
        base_height = float(np.max(masked_heights[
            max(0, x_pos-2):min(w_voxels, x_pos+2),
            max(0, y_pos-2):min(l_voxels, y_pos+2)
        ])) * 250.0
        
        gripper_pattern = self._calculate_gripper_pattern(
            slice_info, rotation * 90
        )
        
        return {
            "x": x_mm,
            "y": y_mm,
            "z": base_height,
            "rotation": rotation * 90,
            "gripper_pattern": gripper_pattern,
            "confidence": 0.85
        }
    
    def _calculate_gripper_pattern(
        self,
        slice_info: Dict[str, float],
        rotation: int
    ) -> List[int]:
        """
        Calculate which gripper fingers to activate based on slice size.
        
        The 5-finger vacuum gripper pattern depends on the slice dimensions
        and orientation.
        """
        width = slice_info["width"]
        length = slice_info["length"]
        
        if rotation in [90, 270]:
            width, length = length, width
        
        pattern = [1, 1, 1, 1, 1]
        
        if width < 100:
            pattern[0] = 0
            pattern[4] = 0
        
        if length < 100:
            pattern[1] = 0
            pattern[3] = 0
        
        if width < 70 and length < 70:
            pattern = [0, 0, 1, 0, 0]
        
        return pattern
    
    def save(self, path: str):
        """Save the model and normalization statistics."""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save(save_path / "ppo_model")
        
        if self.vec_env is not None:
            self.vec_env.save(save_path / "vec_normalize.pkl")
        
        config = {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef
        }
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    def load(self, path: str):
        """Load a saved model."""
        load_path = Path(path)
        
        if (load_path / "config.json").exists():
            with open(load_path / "config.json", "r") as f:
                config = json.load(f)
                self.learning_rate = config.get("learning_rate", self.learning_rate)
                self.n_steps = config.get("n_steps", self.n_steps)
                self.batch_size = config.get("batch_size", self.batch_size)
        
        def make_env():
            return Monitor(MeatPackingEnv())
        
        self.vec_env = DummyVecEnv([make_env])
        
        if (load_path / "vec_normalize.pkl").exists():
            self.vec_env = VecNormalize.load(
                load_path / "vec_normalize.pkl",
                self.vec_env
            )
            self.vec_env.training = False
            self.vec_env.norm_reward = False
        
        self.model = PPO.load(
            load_path / "ppo_model",
            env=self.vec_env
        )
    
    def evaluate(
        self,
        n_episodes: int = 10,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate the agent's performance.
        
        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render the environment
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        env = MeatPackingEnv(render_mode="rgb_array" if render else None)
        
        rewards = []
        fill_percentages = []
        flatness_scores = []
        slices_placed = []
        
        for _ in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            rewards.append(episode_reward)
            fill_percentages.append(info.get("fill_percentage", 0))
            flatness_scores.append(info.get("flatness", 0))
            slices_placed.append(info.get("slices_placed", 0))
        
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_fill_percentage": float(np.mean(fill_percentages)),
            "mean_flatness": float(np.mean(flatness_scores)),
            "mean_slices_placed": float(np.mean(slices_placed))
        }


def create_agent(
    model_path: Optional[str] = None,
    **kwargs
) -> MeatPackingAgent:
    """
    Factory function to create a MeatPackingAgent.
    
    Args:
        model_path: Path to load a pre-trained model
        **kwargs: Additional arguments for the agent
        
    Returns:
        Configured MeatPackingAgent instance
    """
    if model_path:
        return MeatPackingAgent(model_path=model_path)
    
    env = MeatPackingEnv()
    return MeatPackingAgent(env=env, **kwargs)
