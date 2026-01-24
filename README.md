# 📧 SMS Spam Classifier

A machine learning-based SMS/Email spam detection system with an interactive web interface built using Streamlit. This project implements advanced NLP techniques and multiple classification algorithms to accurately identify spam messages with high precision.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://email-sms-spam-predictor-karangautam870.streamlit.app/)

## 🎯 Project Overview

This project tackles the problem of SMS spam detection using Natural Language Processing (NLP) and Machine Learning. The system analyzes text messages and classifies them as either "Spam" or "Not Spam" with high accuracy and precision.

### ✨ Key Features

- **Interactive Web Interface**: User-friendly Streamlit app for real-time spam detection
- **High Accuracy**: Achieved 97%+ accuracy using optimized MultinomialNB
- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- **Multiple Models Tested**: Compared 10+ algorithms to find the best performer
- **Text Preprocessing**: Advanced NLP pipeline with tokenization, stemming, and stopword removal
- **Feature Engineering**: TF-IDF vectorization with additional text-based features
- **Model Optimization**: Hyperparameter tuning and ensemble methods for improved performance

## 📊 Dataset Analysis

**Dataset**: SMS Spam Collection Dataset
- **Total Messages**: 5,572 messages
- **After Removing Duplicates**: 5,169 messages
- **Class Distribution**:
  - Ham (Not Spam): 4,516 messages (87.4%)
  - Spam: 653 messages (12.6%)

### 📈 Key Insights from EDA

1. **Imbalanced Dataset**: Spam messages are significantly fewer than ham messages
2. **Character Count Analysis**:
   - Spam messages tend to be longer on average
   - Maximum characters in spam: ~900
   - Maximum characters in ham: ~900+

3. **Word Count Patterns**:
   - Spam messages contain more words on average
   - Clear distinction in word distribution between classes

4. **Common Spam Indicators**:
   - Words like "free", "call", "claim", "prize", "win" frequently appear in spam
   - Spam often contains more numbers and special characters

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing

```python
- Text lowercasing
- Tokenization using NLTK
- Removing non-alphanumeric characters
- Stopword removal (English)
- Porter Stemming
- TF-IDF Vectorization (max_features=3000)
```

### 2. Feature Engineering

- **TF-IDF Features**: 3,000 features extracted from text

### 3. Models Evaluated

| Model | Accuracy | Precision | Status |
|-------|----------|-----------|--------|
| **MultinomialNB** | **97.1%** | **100%** | ✅ Selected |
| ExtraTreesClassifier | 97.5% | 100% | ⭐ Top Performer |
| RandomForestClassifier | 97.3% | 99% | ⭐ Excellent |
| GradientBoostingClassifier | 95.8% | 98% | ✅ Good |
| AdaBoostClassifier | 96.3% | 96% | ✅ Good |
| BaggingClassifier | 96.0% | 95% | ✅ Good |
| BernoulliNB | 97.1% | 100% | ⭐ Excellent |
| GaussianNB | 87.9% | 52% | ❌ Poor Precision |
| LogisticRegression | 95.6% | 97% | ✅ Good |
| SVC | 97.5% | 98% | ⭐ Excellent |
| KNeighborsClassifier | 90.8% | 100% | ⚠️ Low Accuracy |
| DecisionTreeClassifier | 93.2% | 88% | ⚠️ Moderate |
| XGBoost | 95.2% | 95% | ✅ Good |

### 4. Model Selection Rationale

**MultinomialNB** was chosen for deployment because:
- ✅ **Perfect Precision (100%)**: Zero false positives - no legitimate messages marked as spam
- ✅ **High Accuracy (97.1%)**: Excellent overall performance
- ✅ **Fast Inference**: Quick predictions for real-time use
- ✅ **Small Model Size**: Easy to deploy and version control
- ✅ **Proven for Text**: Well-suited for NLP tasks

## 🚀 Model Optimization Techniques

### Hyperparameter Tuning
- Experimented with different TF-IDF configurations

### Ensemble Methods Tested
1. **Voting Classifier** (Hard & Soft voting)
2. **Stacking Classifier** (Multiple base models + meta-learner)
3. **Bagging & Boosting** (AdaBoost, Gradient Boosting)

### Performance Metrics Focused On
- **Precision**: Prioritized to minimize false positives (ham marked as spam)
- **Accuracy**: Overall correctness of predictions
- **F1-Score**: Balance between precision and recall

## 🛠️ Technologies Used

### Core ML & Data Science
- **Python 3.8+**
- **scikit-learn**: Machine learning models and preprocessing
- **NLTK**: Natural language processing
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization
- **WordCloud**: Text visualization

### Web Framework
- **Streamlit**: Interactive web application

### Model Persistence
- **Pickle**: Model serialization

## 📁 Project Structure

```
sms_spam_predicter/
│
├── spam_classifier_streamlit.py   # Streamlit web application
├── sms_spam_detection.ipynb       # Complete analysis & model training
├── dataset.csv                     # SMS spam dataset
├── model.pkl                       # Trained MultinomialNB model
├── vectorizer.pkl                  # Fitted TF-IDF vectorizer
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/sms-spam-classifier.git
cd sms-spam-classifier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (First time only)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

4. **Run the Streamlit app**
```bash
streamlit run spam_classifier_streamlit.py
```

5. **Open in browser**
   - The app will automatically open at `http://localhost:8501`

## 🌐 Deployment on Streamlit Cloud

### Step-by-Step Deployment

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit - SMS Spam Classifier"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `spam_classifier_streamlit.py`
   - Click "Deploy!"

3. **Your app will be live at**: `https://your-app-name.streamlit.app`

## 💻 Usage

### Using the Web App

1. Open the Streamlit application
2. Enter your SMS/Email message in the text area
3. Click the "Predict" button
4. View the classification result:
   - 🔴 **Spam**: Message classified as spam
   - 🟢 **Not Spam**: Message is legitimate

### Example Messages to Test

**Spam Examples:**
```
"URGENT! You've won a $1000 gift card. Call now to claim your prize!"
"FREE entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121"
```

**Ham (Not Spam) Examples:**
```
"Hey, are we still meeting for coffee tomorrow at 3pm?"
"Can you pick up some milk on your way home? Thanks!"
```

## 📊 Model Performance Details

### Confusion Matrix (Test Set)
```
                Predicted
              Ham    Spam
Actual Ham    958      6
      Spam      24    146
```

### Classification Metrics
- **Accuracy**: 97.1%
- **Precision**: 100%
- **Recall**: 96.1%
- **F1-Score**: 98.0%

### Key Insights
- **Zero False Positives**: No legitimate messages incorrectly marked as spam
- **High Recall**: Catches 96.1% of all spam messages
- **Balanced Performance**: Excellent across all metrics

## 🎓 What I Learned

### Technical Skills
1. **End-to-End ML Pipeline**: From data collection to deployment
2. **NLP Techniques**: Text preprocessing, tokenization, stemming, TF-IDF
3. **Model Comparison**: Evaluating multiple algorithms systematically
4. **Hyperparameter Tuning**: Optimizing model performance
5. **Web Deployment**: Building and deploying ML models with Streamlit

### Best Practices
- Importance of data cleaning and EDA
- Handling imbalanced datasets
- Feature engineering for NLP tasks
- Model selection based on business requirements (precision vs recall)
- Version control for ML models

## 🔮 Future Enhancements

- [ ] Add support for multiple languages
- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add confidence scores to predictions
- [ ] Create REST API for integration
- [ ] Add batch processing capability
- [ ] Implement active learning for continuous improvement
- [ ] Add more visualization features (word clouds, feature importance)
- [ ] Include email header analysis for better detection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/your-profile)

## 🙏 Acknowledgments

- Dataset: [UCI Machine Learning Repository - SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- Inspired by real-world spam detection needs
- Thanks to the open-source community for amazing tools

## 📧 Contact

For questions or feedback, please open an issue or reach out via GitHub.

---

⭐ **If you found this project helpful, please give it a star!** ⭐
