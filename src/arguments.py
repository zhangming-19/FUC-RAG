from email.policy import default
import os
from dataclasses import dataclass, field
from typing import Optional
import typing
from transformers import TrainingArguments



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    
@dataclass
class DataArguments:
    dataset_type: str = field(default="supervised_finetune")
    train_file: str = field(default=None, metadata={"help": "Path to the training data."})
    max_len: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

@dataclass
class EasyTrainArguments(TrainingArguments):
    alpha: float = field(default=0.5, metadata={"help": "The alpha parameter for the input contrastive loss."})
    beta: float = field(default=0.5, metadata={"help": "The beta parameter for the input contrastive loss."})

    model_type: str = field(default='llama3_input_input_contrastive', metadata={"help": "The model type."})
    use_lora: bool = field(default=False)
    train_mode: str = field(default='sft', metadata={"help": "The training mode."}) # sft input_contrastive dpo
    initial_margin: float = field(default=1.0, metadata={"help": "initial margin"})
    final_margin: float = field(default=1.0, metadata={"help": "Final margin"})
    cd_margin: float = field(
        default=1.0,
        metadata={"help": "Initial D-KPO conflict-discrimination margin."},
    )
    cd_final_margin: typing.Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Final D-KPO conflict-discrimination margin. "
                "None keeps the legacy fixed cd_margin."
            )
        },
    )
    dkpo_eta: float = field(
        default=0.0,
        metadata={"help": "Weight of the D-KPO CD loss."},
    )
    dkpo_detach_cc: bool = field(
        default=False,
        metadata={"help": "Stop the CD-loss gradient through CE_cc."},
    )
    dkpo_pr_eta: float = field(
        default=0.0,
        metadata={"help": "Weight of the closed-book parametric-retention loss."},
    )
    dkpo_pr_margin: float = field(
        default=1.0,
        metadata={"help": "Margin of the closed-book parametric-retention loss."},
    )
    dkpo_rgdu_eta: float = field(
        default=0.0,
        metadata={"help": "Weight of risk-gated divergent-token unlikelihood."},
    )
    dkpo_rgdu_gate_margin: float = field(
        default=1.0,
        metadata={"help": "Activate RGDU when CE_pc - CE_cc is below this margin."},
    )
    lora_r: int = field(default=64, metadata={"help": "lora r"})
    lora_alpha: int = field(default=64, metadata={"help": "lora alpha"})

    mask_path: str = field(default='', metadata={"help": "Parameter level mask path"})
    wandb_project: str = field(default="default_project", metadata={"help": "W&B project name"})
    # wandb_run_name: str = field(default="default_run", metadata={"help": "W&B run name"})
    debug_mode: bool = field(default=False)
    inhibit_strength: float = field(default=1.0, metadata={"help": "Inhibit strength for the model."})
    inhibit_layer_list: typing.List[int] = field(
        default_factory=lambda: [],
        metadata={"help": "List of layers to apply inhibition."}
    )