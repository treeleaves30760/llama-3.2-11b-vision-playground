# LLAMA 3.2 vision playground

This is a project for llama 3.2 vision playground

## Install

Copy the .env.example file, an replace the HUGGINGFACE_TOKEN to yours

```
HUGGINGFACE_TOKEN=your token
```

Create the conda environment

```bash
conda create -n llama3.2-vision python==3.11.9
conda activate llama3.2-vision
```

Then run the install script

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txt
```

## Usage

```bash
python main.py
```