# SMS Spam Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://g114064015lab-aiot-da-hw3-app-qw62t5.streamlit.app)

A simple but effective spam classifier using scikit-learn. Includes preprocessing, model training, and inference capabilities.

🔗 **Try it now:** [SMS Spam Classifier Web App](https://g114064015lab-aiot-da-hw3-app-qw62t5.streamlit.app)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train baseline model (saves to models/final_model.pkl)
python main.py

# Start the web UI
streamlit run app.py

# Or make predictions from command line
python scripts/predict.py --message "WINNER!! You have won £1000, call now to claim!"
```
## Dataset Reference

The dataset used in this project is sourced from the book **"Hands-On Artificial Intelligence for Cybersecurity"** by Packt Publishing.

**File:** [`sms_spam_no_header.csv`](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/blob/master/chapter3/dataset/sms_spam_no_header.csv)  
**Repository:** [PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity)  
**Chapter:** Chapter 3 — SMS Spam Detection  

## Web Interface

The project includes a user-friendly web interface built with Streamlit. To use it:

1. Make sure you've trained the model first using `python main.py`
2. Run `streamlit run app.py`
3. Open your browser to the URL shown (usually http://localhost:8501)
4. Enter any SMS message to test the classifier!

## Project Structure

- `data/` - Dataset storage
  - `sms_spam_no_header.csv` - Raw dataset (downloaded on first run)
- `src/` - Source code
  - `preprocess.py` - Text cleaning and feature extraction
  - `train.py` - Model training utilities
- `scripts/` - Command-line tools
  - `predict.py` - Make predictions with trained model
- `models/` - Saved model artifacts
  - `final_model.pkl` - Trained pipeline (created after training)
- `main.py` - End-to-end training script
- `requirements.txt` - Python dependencies

## Training

The default training script (`main.py`) uses:
- TF-IDF vectorization with word bigrams
- Multinomial Naive Bayes classifier
- 80/20 train/test split

For more options:
```bash
python main.py --help
```

## Making Predictions

Use `scripts/predict.py` to classify new messages:

```bash
# Single message
python scripts/predict.py --message "Your message here"

# Multiple messages from file
python scripts/predict.py --input-file messages.txt --output-file predictions.csv
```

## Development

To run with debug logging:
```bash
python run_main_debug.py  # Saves logs to run.log
```

## Testing

Run the test suite with:
```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Run tests with coverage report
pytest tests/ --cov=src/
```

Tests include:
- Text preprocessing and cleaning
- Model training and evaluation
- End-to-end pipeline validation

## Live Demo

The app is deployed on Streamlit Cloud and can be accessed here:
https://g114064015lab-aiot-da-hw3-app-qw62t5.streamlit.app

Features of the web interface:
- 🚀 Instant spam detection
- 📊 Confidence scores for predictions
- 🔍 Text preprocessing visualization
- 💡 Model information and details
- 📱 Mobile-friendly design

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies
- For development: See `tests/requirements-test.txt`
- For Streamlit deployment: See `requirements-streamlit.txt`

## Deployment

This project is deployed on Streamlit Cloud. To deploy your own instance:

1. Fork this repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Create a new app using your forked repository
4. Set the following:
   - Main file path: `app.py`
   - Requirements file: `requirements-streamlit.txt`
   - Python version: 3.9
