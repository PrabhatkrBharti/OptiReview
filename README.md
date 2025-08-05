# LengthPR: Peer Review Length Analysis

This repository contains the research paper titled "Dynamic Optimization of Peer Review Length Using Information Density Analysis". The paper explores methods to optimize the length of peer reviews by analyzing the density of information provided.

## Abstract

The effectiveness of peer reviews in academic publishing is often influenced by the balance between brevity and comprehensiveness. Reviews that are too verbose or too concise may fail to convey critical insights, reducing their utility. This paper presents a novel heuristic system for dynamically optimizing peer review lengths, leveraging information density and argumentation metrics. Using a curated dataset that contains quantitative and qualitative metrics such as content relevance, argument strength, readability index, and unique insights per word, we develop a composite score to assess review quality. Our system employs thresholds for normalized length, information density, and adjusted argument strength to classify reviews as poor, moderate, or excellent. Through empirical refinement and analysis, the heuristic framework demonstrates its ability to enhance review quality by balancing word count with clarity and argument consistency. Scatter plots and histograms reveal critical relationships between composite scores and key metrics, offering actionable insights into optimal review lengths. The results highlight that optimizing peer reviews can significantly improve their quality and relevance. This study provides a foundation for integrating heuristic systems into academic review platforms, ensuring that reviews achieve the desired balance of brevity, depth, and clarity.

## Project Structure

### Core Directories

- **`Code/`** - Main codebase containing analysis tools
  - `HuggingFaceAPI/` - Integration with Hugging Face models for text analysis
  - `OptimalLengthCalculator/` - Tools for calculating optimal peer review lengths
  - `PrepareDataset/` - Data preprocessing and preparation utilities
  - `SavedPlots/` - Generated visualizations and analysis plots

- **`Datasets/`** - Processed data files
  - `abstracts_list.csv` - Collection of paper abstracts
  - `analysis_output.csv` - Results from length analysis
  - `cleaned_reviews.csv` - Preprocessed peer review data
  - `final_dataset.csv` - Final processed dataset for analysis
  - `length_optimization_output.csv` - Results from optimal length calculations
  - `processed_reviews.csv` - Intermediate processed review data
  - `segmented_reviews.csv` - Reviews divided into segments

- **`annotation/`** - Manual annotation data and expert reviews
  - `annotation_review_comments.txt` - General annotation comments
  - `expert_1/`, `expert_2/`, `expert_3/` - Individual expert annotations

## Features

- **Length Analysis**: Calculate and analyze optimal peer review lengths
- **Dataset Processing**: Clean and prepare peer review datasets
- **Expert Annotation**: Multi-expert annotation for quality assessment
- **Visualization**: Generate plots and charts for analysis results
- **API Integration**: Leverage Hugging Face models for text analysis

## Getting Started

### Prerequisites

- Python 3.x
- Required dependencies (check individual module requirements)

### Installation

1. Set up virtual environments for different modules:
   ```bash
   # For dataset preparation
   cd Code/PrepareDataset
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # For Hugging Face API
   cd ../HuggingFaceAPI
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies for each module as needed.

## Usage

### Data Processing
Navigate to `Code/PrepareDataset/` to access data preprocessing tools.

### Length Optimization
Use tools in `Code/OptimalLengthCalculator/` to calculate optimal review lengths.

### API Integration
Access Hugging Face model integration through `Code/HuggingFaceAPI/`.

### API Usage

Once you run the Gradio app locally (see **Running Locally** below), an HTTP REST API is exposed:

- **Endpoint**:  

#### POST /api/predict

- **Payload** (JSON):  
```json
{
  "data": [
    ["<review_comments_text>", "<paper_abstract_text>"]
  ]
}
```

- **Response** (JSON):

  ```json
  {
    "data": [
      ["<composite_score_number>", "<optimization_suggestions_string>"]
    ],
  }
  ```

- **Example with `curl`**:

  ```bash
  curl -X POST http://localhost:7860/api/predict \
    -H "Content-Type: application/json" \
    -d '{"data":[["The paper is solid but could improve on clarity.","This paper explores..."]]}'
  ```

#### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the app
python app.py
```

By default, the interface and API will be available at [http://0.0.0.0:7860](http://0.0.0.0:7860).

### Hugging Face Space

This Gradio demo is also deployed as a Hugging Face Space for easy web access:

[![Hugging Face Space](https://img.shields.io/badge/HuggingFace-Space-blue?logo=huggingface)](https://huggingface.co/spaces/Legal-NLP-404/Length_Optimization_Peer_Review)

You can interact with the web UI directly or call the same `/api/predict` endpoint over HTTPS:

```bash
curl -X POST https://huggingface.co/spaces/Legal-NLP-404/Length_Optimization_Peer_Review \
  -H "Content-Type: application/json" \
  -d '{"data":[["Great motivation and clear methodology.","This paper explores..."]]}'
```

## License

See [LICENSE](LICENSE) file for details.
