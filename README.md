# Luddite-Decisions-GPT2-XL

A System 1 decision model built on **[openai-community/gpt2-xl](https://huggingface.co/openai-community/gpt2-xl)** by
**OpenAI**. OpenAI published openai-community/gpt2-xl under the MIT License, and
this model exists because of that release. It is released under the MIT License too.

It reads a state and a question with 2 to 10 lettered options, and returns a probability for every
option from one forward pass. It does not generate text.

- **Base model:** [openai-community/gpt2-xl](https://huggingface.co/openai-community/gpt2-xl), 1.5B parameters, MIT License (OpenAI). Its license file ships here as `BASE_LICENSE` (OpenAI's license text from github.com/openai/gpt-2).
- **Readout:** a small trained head that scores each option from the hidden state at the end of its line and at `Answer:` (the base model does not bind answers to letters), softmaxed over the options, temperature 1.75.
- **Training:** fine-tuning of the top 24 blocks (8-bit AdamW) on 49,241 MIT-licensed examples, 2 epochs, 96.3 minutes on one RTX 3090.
- **License:** MIT. The base model's MIT license is kept alongside ours.
- **Part of:** Jeff, an MIT Godot plug-in in which a small local decision model plays a game character.

## Use

```python
from luddite_decisions import DecisionModel

model = DecisionModel(".")          # quant="nf4" loads it in 4-bit
model.decide(
    {"health": 35, "ammo": 0, "enemy_in_sight": True},
    "Rules for the guard: if ammo is 0, reload. Otherwise, if enemy_in_sight is true, fire. Otherwise, hold position. What should happen now?",
    ["reload", "fire", "hold position"],
)
# -> {"pick": "hold position", "probs": {"reload": 0.3768, "fire": 0.2077, "hold position": 0.4155}}  (a wrong answer: see Limits)
```

The weights are in the [v0.1 release](https://github.com/TechLuddite/Luddite-Decisions-GPT2-XL/releases/tag/v0.1)
(3.1 GB in files under 2 GB). Get the code and the weights:

```
git clone https://github.com/TechLuddite/Luddite-Decisions-GPT2-XL
cd Luddite-Decisions-GPT2-XL
pip install -r requirements.txt
python download_weights.py
```

`download_weights.py` uses only the Python standard library and checks every file against
`weights.sha256`. Requirements: Python 3.12, PyTorch, transformers, safetensors, and bitsandbytes
for 4-bit loading.

## Prompt format

```
State: <text, or JSON>
Question: <the question, including any rules it needs>
A. <option>
B. <option>
Answer:
```

## Results

Accuracy on questions the model never saw, next to the base model given the same prompt. The game
decisions, the held-out synthetic families and MMLU are also question types it never trained on.
The in-training synthetic families and CommonsenseQA are types it trained on, with new questions.

| Set | Chance | Base model | This model (bf16) | This model (4-bit) |
|---|---|---|---|---|
| Game decisions (bench v1, 8 families never trained on) | 24.4% | 23.0% | 50.5% | 50.2% |
| Held-out synthetic families (2 families never trained on) | 25.0% | 26.2% | 29.5% | 27.8% |
| Synthetic families seen in training (new questions) | 25.1% | 22.8% | 71.2% | 69.5% |
| CommonsenseQA validation | 20.0% | 20.6% | 60.7% | 59.1% |
| MMLU test, 25 per subject (never trained on) | 25.0% | 23.6% | 30.4% | 30.9% |

Calibration error (ECE, 0 is perfect) and letter consistency (the share of questions whose pick
stays the same option when the options are listed in reverse order):

| Set | Base: ECE / letter consistency | This model: ECE / letter consistency |
|---|---|---|
| Game decisions (bench v1, 8 families never trained on) | 0.326 / 8.8% | 0.035 / 72.0% |
| Held-out synthetic families (2 families never trained on) | 0.470 / 0.3% | 0.171 / 54.2% |
| Synthetic families seen in training (new questions) | 0.485 / 2.5% | 0.061 / 81.4% |
| CommonsenseQA validation | 0.596 / 0.7% | 0.042 / 78.9% |

Per family (this model, bf16):

- Game decisions: fits 100.0%, value per cell 81.3%, exit for quest 69.3%, nearest threat 48.0%, heal 33.3%, weapon 28.0%, cover 24.0%, count seen 20.0%.
- Held-out synthetic: except category 33.0%, multi hop 26.0%.
- Synthetic families seen in training: compare two 100.0%, direction 100.0%, fit capacity 100.0%, intent 100.0%, list difference 100.0%, lookup 90.0%, ratio best 90.0%, yes no 90.0%, category rule 80.0%, times 80.0%, rule policy 45.0%, arithmetic 42.5%, rank position 35.0%, count condition 32.5%, filtered extreme 30.0%, extreme 25.0%.

GPU memory for the weights: 3.13 GB in bf16, 1.01 GB in 4-bit.

## Training data

All MIT:

- **Synthetic decision questions** written for this project, with answers computed by code: 16
  families (picking a highest or lowest value, with and without a filter, rule policies,
  counting, table lookups, ratios, capacity fits, times, rankings, yes or no checks, two-way
  comparisons, directions, classification rules, short-text intent, list differences,
  arithmetic). States come as JSON, as key-value lines or as tables.
- **[CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa)** train split (MIT, per its authors).

Options are reshuffled every time an example is seen, so no letter carries information.

## Limits

- The synthetic questions state the rules they need. The model applies stated rules and reads
  states. It has no game doctrine of its own.
- English only. Up to 10 options.
- MMLU measures general knowledge, which this training does not add.
- Much weaker than larger models. It handles direct questions (intent, yes or no, two-way comparisons, directions, capacity fits) but not picking the highest or lowest from a list, counting, ranking or rule chains: the example in Use above is one it gets wrong.
- GPT-2 does not bind answers to letters. Training it to answer with a letter stayed at chance, so it uses the option head. The top 24 of its 48 blocks were fine-tuned for 2 epochs. The lower blocks and the embeddings are unchanged.
- Its raw scores are overconfident: the fitted temperature is 1.75, applied automatically from `decisions_config.json`.
- Early release, not yet tested in play. Feedback is welcome.
