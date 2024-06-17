# NLP Job Description Keyword Predictor

This project analyzes job descriptions to predict the most important keywords. By leveraging Natural Language Processing (NLP) techniques, it aims to help recruiters and job seekers identify key skills and qualifications in job postings.

## Project Overview

The NLP Job Description Keyword Predictor project involves the following key components:

- **Data Collection**: Gathering a dataset of job descriptions.
- **Data Preprocessing**: Cleaning and preparing the text data for analysis.
- **Model Training**: Using NLP models to analyze job descriptions and predict keywords.
- **Evaluation**: Assessing the model's performance and accuracy.
- **Deployment**: Providing an interface for users to input job descriptions and receive keyword predictions.

## Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- Required Python packages (listed in `requirements.txt`)

## Usage

1. Clone the repository
2. Extract the models.zip and data.zip files. This includes the pre-processed dataset and pre-trained model versions. 
3. Navigate to the `ovr_model.ipynb` and the `knn_model.ipynb` demo notebooks to run the OVR Model and KNN Model respectively. This can be performed by running all the cells. This will load the pre-processed data and pre-trained models, and evaluate the model on the testing set and sample data points. Each file should take roughly three minutes to run, with most of the time being attributed to the large amount of testing points. 
4. Configure the `force_reload` variables to force the pre-processed data or pre-trained model to be reproduced. Be mindful that this will overwrite the previously loaded files and you may need to re-download.  