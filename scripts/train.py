#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from src.training.trainer import PegasusTrainer
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig

def main():
    parser = argparse.ArgumentParser(description="Train Pegasus model")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--data-size", type=int, default=2000, help="Dataset size")
    parser.add_argument("--output-dir", type=str, default="./results")
    
    args = parser.parse_args()
    
    # Initialize configs
    model_config = ModelConfig()
    training_config = TrainingConfig()
    training_config.output_dir = args.output_dir
    
    # Initialize trainer
    trainer = PegasusTrainer(model_config, training_config)
    
    # Train model
    trainer.train(data_size=args.data_size)
    
    # Save model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))

if __name__ == "__main__":
    main()
