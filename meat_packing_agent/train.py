#!/usr/bin/env python3
"""
Training script for the Meat Packing RL Agent.

This script trains a PPO agent to optimally place meat slices in cubes.
"""

import argparse
import os
from pathlib import Path

from meat_packing_agent.env.cube_environment import MeatPackingEnv
from meat_packing_agent.agent.ppo_agent import MeatPackingAgent, create_agent


def main():
    parser = argparse.ArgumentParser(description="Train the Meat Packing RL Agent")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps (default: 100000)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/meat_packing_agent",
        help="Path to save the trained model (default: models/meat_packing_agent)"
    )
    parser.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="Path to load a pre-trained model for continued training"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation (default: 10)"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate, don't train"
    )
    
    args = parser.parse_args()
    
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Meat Packing RL Agent Training")
    print("=" * 60)
    print(f"Timesteps: {args.timesteps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save path: {args.save_path}")
    print("=" * 60)
    
    if args.load_path:
        print(f"Loading pre-trained model from {args.load_path}")
        agent = MeatPackingAgent(model_path=args.load_path)
    else:
        print("Creating new agent...")
        agent = create_agent(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size
        )
    
    if args.eval_only:
        print("\nRunning evaluation only...")
        metrics = agent.evaluate(n_episodes=args.eval_episodes)
        print("\nEvaluation Results:")
        print(f"  Mean Reward: {metrics['mean_reward']:.2f} (+/- {metrics['std_reward']:.2f})")
        print(f"  Mean Fill %: {metrics['mean_fill_percentage']:.1f}%")
        print(f"  Mean Flatness: {metrics['mean_flatness']:.2f}")
        print(f"  Mean Slices Placed: {metrics['mean_slices_placed']:.1f}")
        return
    
    print("\nStarting training...")
    training_metrics = agent.train(
        total_timesteps=args.timesteps,
        save_path=args.save_path
    )
    
    print("\nTraining Complete!")
    print(f"  Total Episodes: {training_metrics['total_episodes']}")
    print(f"  Mean Reward (last 100): {training_metrics['mean_reward']:.2f}")
    print(f"  Mean Fill % (last 100): {training_metrics['mean_fill_percentage']:.1f}%")
    print(f"  Mean Flatness (last 100): {training_metrics['mean_flatness']:.2f}")
    
    print(f"\nSaving model to {args.save_path}...")
    agent.save(args.save_path)
    
    print("\nRunning final evaluation...")
    eval_metrics = agent.evaluate(n_episodes=args.eval_episodes)
    print("\nFinal Evaluation Results:")
    print(f"  Mean Reward: {eval_metrics['mean_reward']:.2f} (+/- {eval_metrics['std_reward']:.2f})")
    print(f"  Mean Fill %: {eval_metrics['mean_fill_percentage']:.1f}%")
    print(f"  Mean Flatness: {eval_metrics['mean_flatness']:.2f}")
    print(f"  Mean Slices Placed: {eval_metrics['mean_slices_placed']:.1f}")
    
    print("\n" + "=" * 60)
    print("Training complete! Model saved to:", args.save_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
