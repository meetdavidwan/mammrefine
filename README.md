# MAMM-Refine: Multi-Agent Multi-Modal Refinement

A framework for improving factual consistency in text generation through multi-agent collaboration and debate.

**Authors:** David Wan, Justin Chen, Elias Stengel-Eskin, Mohit Bansal

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/meetdavidwan/mammrefine.git
cd mammrefine
pip install -r requirements.txt
```

### Setup API Keys
```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Run Complete Pipeline
```bash
bash run_script.sh
```

### Try a Simple Example
```bash
python example.py
```

## 📖 Overview

MAMM-Refine improves text generation faithfulness through a three-stage pipeline:

1. **Detection**: Identifies factually inconsistent sentences
2. **Critique**: Generates detailed feedback explaining inconsistencies
3. **Refinement**: Produces improved summaries based on critiques

Each stage supports both single-agent and multi-agent debate modes.

## 🔧 Usage

### Single Agent Mode
```bash
# Detection
python src/detect.py gpt-4o data/mediasum.json output/detection.json

# Critique
python src/critique.py gpt-4o detection.json output/critiques.json

# Refinement
python src/refine.py claude-3-sonnet critiques.json output/refined.json
```

### Multi-Agent Debate Mode
```bash
# Run the complete debate pipeline
bash run_script.sh
```

## 📊 Input Format

Your input JSON should contain documents with this structure:

```json
[
  {
    "document": "Source document text...",
    "summary": "Summary to be refined...",
    "summary_sentences": ["Sentence 1", "Sentence 2"],
    "topic": "Document topic"
  }
]
```

## 🤖 Supported Models

- **OpenAI**: `gpt-4o`
- **Anthropic**: `claude-3-sonnet`

## 📁 Project Structure

```
mammrefine/
├── src/                    # Core implementation
│   ├── detect.py          # Factual consistency detection
│   ├── critique.py        # Critique generation
│   ├── refine.py          # Summary refinement
│   ├── model.py           # Model interface
│   └── prompts.py         # System prompts
├── data/                  # Input datasets
├── output/                # Generated results
├── run_script.sh          # Complete pipeline script
└── requirements.txt       # Dependencies
```

## 🔬 Multi-Agent Debate Process

The debate system uses iterative rounds:

1. **Initial**: Multiple agents process input independently
2. **Selection**: Agents choose between different outputs
3. **Debate**: Agents review reasoning and make informed choices
4. **Final**: Consensus or best-selected output

## 📝 Citation

```bibtex
@inproceedings{wan-etal-2025-mamm,
    title = "{MAMM}-Refine: A Recipe for Improving Faithfulness in Generation with Multi-Agent Collaboration",
    author = "Wan, David and Chen, Justin and Stengel-Eskin, Elias and Bansal, Mohit",
    booktitle = "Proceedings of NAACL 2025",
    year = "2025"
}
```