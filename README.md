# Multimodal Video Analysis Application

A real-time video analysis application combining object detection, visual language models, and speech processing using Gradio interface.

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd ProjectCode
```

2. Install Miniconda (if not already installed):
- Download from: https://docs.conda.io/en/latest/miniconda.html
- Follow the installation instructions for your operating system

3. Create and activate the conda environment:
```bash
conda env create -f envi.yml
conda activate projectcode
```

4. Create a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

5. Start the application:
```
python app.py
```

The application will be available at:
- Local URL: http://localhost:7860
- Public URL will be provided in the console (expires after 72 hours)
## Todo
- Test Qwen2.5VL, InternVL, NVIDIA EAGLE
- Test other object  detection models (DETR, Detectron, Faster RCNN, etc)