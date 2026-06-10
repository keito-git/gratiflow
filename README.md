# GratiFlow: A Scaffolding-Fading Multi-Agent LLM Framework\\for Positive Reframing Skill Internalization


## Overview

GratiFlow coordinates four specialized agents to adaptively control scaffolding intensity from high (modeling) to low (independent production) based on the Spontaneous Reframing Rate (SRR). The framework is designed to promote users' autonomous cognitive reframing skills rather than providing AI-generated reframes.

### Four-Agent Architecture

1. **Savoring Agent** - Generates deepening questions to elaborate on positive aspects of journal entries
2. **Reframing-Coach Agent** - Prompts reframing at adaptive scaffolding levels (High/Mid/Low)
3. **Affect-Analysis Agent** - Judges whether user-produced reframing qualifies as spontaneous (SRR rubric R1-R3/F1-F5)
4. **Curriculum-Progress Agent** - Tracks skill estimates and manages scaffolding transitions

## Repository Structure

```
gratiflow-release/
├── python/                 # Experiment and analysis scripts
│   ├── generate_and_run_ext_study1_fixed.py   # Main experiment script
│   ├── ext_sensitivity_analysis.py            # Sensitivity analysis (4 regimes)
│   ├── dry_run_ext_study1_fixed.py            # Dry-run screening
│   ├── generate_figures_ext_study1_fixed.py   # Figure generation
│   ├── srr_human_validation_*.py              # SRR instrument validation
│   ├── srr_realdata_*.py                      # Real-data validation (Ziems et al.)
│   ├── analyze_*.py                           # Analysis scripts
│   ├── latent_skill_model.py                  # Latent skill model definition
│   ├── personas_ext_study1.json               # Persona parameter definitions
│   └── figures/                               # Figure generation scripts
├── data/
│   └── raw/                # Raw LLM responses and public datasets
│       ├── synthetic_users_*_raw_responses.json  # Synthetic simulation outputs
│       └── srr_realdata_ziems2022/               # Ziems et al. (2022) sample
├── requirements.txt
├── LICENSE                 # MIT License
└── .gitignore
```

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/keito-inosita/gratiflow.git
cd gratiflow

# Create .env file with your API key
echo "OPENAI_API_KEY=your_api_key_here" > .env

# Install Python dependencies
pip install -r requirements.txt
```

## Reproducing the Experiments

### Synthetic Simulation (Main Experiment)

```bash
# Run the main experiment (10 personas x 2 conditions x 14 sessions)
python python/generate_and_run_ext_study1_fixed.py

# Generate figures
python python/generate_figures_ext_study1_fixed.py

# Run sensitivity analysis (4 multiplier regimes, 20 seeds)
python python/ext_sensitivity_analysis.py
```

### SRR Instrument Validation

```bash
# Run LLM judge on public dataset (N=70)
python python/srr_human_validation_llm_judge_r2_clarified.py

# Compute agreement metrics
python python/srr_human_validation_compare_r2_clarified.py
```

## Key Results

- **Paired RNG Analysis (20 seeds):** Mean A-B = +0.081 (SD = 0.102), A > B in 80% of seeds
- **Flat Regime:** A-B = 0.000 (mathematical necessity confirming assumption dependence)
- **Reversed Regime:** A-B = -0.092 (sign inversion)
- **SRR Instrument Validation:** Cohen's kappa = 0.430 (Moderate), N = 70
- **Scaffold Transitions:** Confirmed in 9 of 10 personas

## External Data

This project uses the [Positive Psychology Frames](https://github.com/SALT-NLP/positive_reframing) dataset by Ziems et al. (2022), available under CC BY-SA 4.0.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code or data, please cite:

```bibtex
@article{inoshita2026gratiflow,
  title={GratiFlow: A Scaffolding-Fading Multi-Agent LLM Framework for Positive Reframing Skill Internalization},
  author={Inoshita, Keito},
  journal={arXiv},
  year={2026}
}
```
