## Text-Diffusion Red Teaming

This repository contains the official implementation of the paper: *"Text-Diffusion Red-Teaming of Large Language Models: Unveiling Harmful Behaviors with Proximity Constraints"* ([arXiv:2501.08246](https://arxiv.org/pdf/2501.08246)).

The project introduces a novel red-teaming framework that employs diffusion models to identify and analyze potentially harmful behaviors in large language models, guided by semantic proximity constraints.

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. (Optional but recommended) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

This project includes several core scripts:

* `Diffusion_Model.py` – Defines the architecture and functionality of the Text Diffusion model.
* `red_team_diffusion.py` – Contains the trainer logic for red-teaming using the diffusion model.
* `rtdConfig.py` – Configuration settings for red-teaming experiments.
* `train_diffusion_model.py` – Training script for the diffusion model, with built-in logging.
* `safety_audit.py` – Uses the trained diffusion model to detect and categorize areas of vulnerability in target models.

---