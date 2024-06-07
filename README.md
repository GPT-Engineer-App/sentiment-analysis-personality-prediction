# Big Five Personality Prediction

This project fine-tunes a sentiment analysis model to predict Big Five personality traits using parameter-efficient methods like LoRA and PEFT.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/big-five-personality-prediction.git
   cd big-five-personality-prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The dataset is located in the `dataset/` directory. It contains a CSV file `big_five_dataset.csv` with the following columns:

- `text`: The input text.
- `openness`: Openness score.
- `conscientiousness`: Conscientiousness score.
- `extraversion`: Extraversion score.
- `agreeableness`: Agreeableness score.
- `neuroticism`: Neuroticism score.

## Training

To train the model, run:
