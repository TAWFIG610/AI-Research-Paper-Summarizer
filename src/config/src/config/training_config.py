from dataclasses import dataclass

@dataclass
class TrainingConfig:
    output_dir: str = "./results"
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    num_train_epochs: int = 1
    weight_decay: float = 0.01
    warmup_steps: int = 100
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 10
    fp16: bool = False
    dataloader_num_workers: int = 0
    report_to: str = "none"
