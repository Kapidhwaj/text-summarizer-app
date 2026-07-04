# Text Summarizer Web Application

A web application that utilizes a fine-tuned Hugging Face T5 Transformer model to generate summaries of conversational dialogue. The project features a high-performance FastAPI backend paired with a custom-designed, clean user interface.

## Live Deployment
The application is hosted online and can be accessed here:
https://huggingface.co/spaces/kapidhwaj/text-summarizer-app

## Project Structure
* text_summarizer.ipynb: Jupyter Notebook containing the full training pipeline, data preprocessing, and model fine-tuning process.
* app.py: FastAPI application server and model inference pipeline.
* index.html: Stylized user interface template.
* requirements.txt: Python dependency definitions.
* Dockerfile: Deployment configuration container script.
* .gitignore: Specifies files to be ignored by version control.
* saved_summary_model/: Directory containing fine-tuned T5 model weights.

## Local Installation

1. Clone the repository:
   git clone https://github.com/kapidhwaj/YOUR_REPOSITORY_NAME.git
   cd text-summarizer-app

2. Install dependencies:
   pip install -r requirements.txt

3. Launch the local server:
   uvicorn app:app --reload

The application will be accessible locally at http://127.0.0.1:8000.

## Docker Deployment
To build and run the application inside a container locally:
docker build -t text-summarizer .
docker run -p 7860:7860 text-summarizer
