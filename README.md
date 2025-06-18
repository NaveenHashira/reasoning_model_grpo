# 🧠 GRPO Reasoning Model for GSM8K 📚

This project implements a Group Relative Preference Optimization (GRPO) approach to fine-tune the Qwen2.5-3B-Instruct language model for mathematical reasoning tasks using the GSM8K dataset.

## 🌟 Overview

The model is trained to solve grade school math problems by:
1. 🔍 Generating step-by-step reasoning in a structured XML format
2. 🎯 Providing a final numerical answer
3. ⚙️ Optimizing for both correctness and output format compliance using GRPO

Key features:
- 🚀 Uses Unsloth for efficient 4-bit quantization and LoRA adaptation
- 🏆 Implements custom reward functions for answer correctness and XML formatting
- ⚡ Leverages vLLM for high-speed text generation
- 📊 Trained on the GSM8K dataset of 8.5K math word problems

## ⚙️ Installation

```bash
pip install -U unsloth transformers trl vllm
pip install --upgrade unsloth unsloth_zoo
```

##   Model Architecture

- **Base Model**: `Qwen/Qwen2.5-3B-Instruct` 🤖
- **LoRA Configuration**:
  - Rank (r): 16
  - Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Sequence Length**: 2048 tokens

## 🚂 Training Configuration

```python
GRPOConfig(
    use_vllm = True,          # ⚡ Turbo-charged generation
    learning_rate = 5e-6,      # 📉 Optimized learning rate
    per_device_train_batch_size = 1,
    num_generations = 3,       # 🔄 Multiple generations per prompt
    max_prompt_length = 500,   # 📏 Input constraints
    max_completion_length = 1500,
    max_steps = 150            # ⏱️ Training duration
)
```

## 🏆 Reward Functions

Two custom reward functions guide the training:

1. **✅ Correctness Reward**:
   - Returns 1.0 if extracted answer matches gold answer
   - Returns 0.0 otherwise

2. **📝 Format Reward**:
   - Returns 0.5 if output strictly follows XML format
   - Returns 0.0 otherwise

```python
def extract_xml_answer(text: str) -> str:
    # ✂️ Extracts content between <answer> tags

def correctness_reward_func(prompts, completions, answer, **kwargs):
    # 🆚 Compares model output with gold answers

def strict_format_reward_func(prompts, completions, **kwargs):
    # 🔍 Validates XML structure using regex
```

## 📂 Data Preparation

The GSM8K dataset is processed with:
- System prompt requiring XML-style output
- Question formatting as user input
- Answer extraction from original dataset

```python
SYSTEM_PROMPT = """
Respond in the format:
<reasoning>...</reasoning>
<answer>...</answer>
"""

def get_gsm8k_questions():
    # 📥 Loads and preprocesses dataset
```

## 🚀 Usage

### Training
```python
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[correctness_reward_func, strict_format_reward_func],
    train_dataset=dataset
)
trainer.train()  # 🚂 Start the training process!
```

### Inference
```python
# Generate response
text = tokenizer.apply_chat_template([...], tokenize=False)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
output = model.fast_generate([text], sampling_params)[0].text

# Example output:
"""
<reasoning>
The name "Naveen" has 5 letters. 
Looking for 'n's: 
- First letter 'N' (capital) 
- Fifth letter 'n' (lowercase)
Total n's: 2
</reasoning>
<answer>
2
</answer>
"""
```

### 💾 Saving/Loading LoRA Weights
```python
# Save trained adapter
model.save_lora("grpo_saved_lora")  # 💾

# Load for inference
model.load_lora("grpo_saved_lora")  # 🔄
```

## 📂 File Structure
- `reasoning_model_grpo.py`: Main training/inference script 🐍
- `outputs/`: Directory for training outputs 📁
- `grpo_saved_lora/`: Saved LoRA adapter weights 💽

## 📦 Dependencies
- Python 3.8+ 🐍
- PyTorch 🔥
- transformers 🤗
- trl 🧪
- vllm ⚡
- unsloth 🦥
- datasets 📊

## 🎯 Key Benefits
- ✨ 4-bit quantization reduces memory requirements
- 🧠 Structured reasoning improves answer accuracy
- ⚡ vLLM enables rapid generation during training
- 🏆 Dual-reward system ensures both correctness and formatting

