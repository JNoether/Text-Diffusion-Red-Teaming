from dataclasses import dataclass

@dataclass
class RTD_Config:
    
    name: str = "unnamed_experiment"
    seed: int = 0
    log_wandb: bool = False

    reward : str = "toxicity"

    eps:float = 0.1
    regularization: str = "threshold_mean"
    reg_coef: float = 1
    normalize_noise : bool = False

    lr: float = 1e-5
    gamma: float = 1.0
    _lambda: float = 0.95
    clipping_parameter : float = 0.1
    batch_size: int = 256
    minibatch_size: int = 64
    ppo_epochs: int = 4
    vf_coef: float = 0.5
    gradient_clipping : float = 1.0
    std_anneal : float = 0.05
    target_kl : float = 0.01
    normalize_advantage : bool = True
    lr_schedule : bool = False
    clip_vfloss : bool = True
    vf_clipping_parameter : float = 0.2
