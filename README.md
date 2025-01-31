**Interactive Chatbot for Personalized Movie Recommendations**


**Project Overview**
This project, Interactive Chatbot for Personalized Movie Recommendations, is an AI-driven system that leverages Large Language Models (LLMs) and hybrid recommendation techniques to provide users with personalized movie suggestions. The chatbot dynamically processes natural language inputs and integrates collaborative filtering, content-based filtering, and neural networks to enhance recommendation accuracy and user interaction.

**Features**


**Hybrid Recommendation Model:** Combines collaborative filtering (SVD), content-based filtering, and multi-tower neural networks for improved accuracy.

**Natural Language Processing (NLP):** Uses Hugging Face Transformers to understand user queries such as "I want to watch a thriller with a detective lead."

**Chatbot Interface:** A user-friendly interactive chatbot that allows dynamic query handling based on user preferences.

**Movie Metadata Integration:** Incorporates movies_metadata.csv and ratings.csv datasets for recommendations, with plans to extend to keywords.csv and credits.csv for more detailed suggestions.

**Real-Time Personalization:** The chatbot adapts recommendations based on user feedback and refined inputs.

**System Architecture**

**Dataset Architecture:** Movie metadata and user ratings are processed for training the recommendation models.

**Model Integration:**

Collaborative Filtering with SVD (Singular Value Decomposition) for user-based recommendations.

Content-Based Filtering using metadata attributes such as genres, popularity, and keywords.

Multi-Tower Neural Networks to enhance feature extraction and representation.

Hugging Face Integration: Enhances NLP capabilities for processing user queries dynamically.

Chatbot Interface: Provides real-time interaction and response to user queries.

**Installation & Setup**

**To set up the project on your local machine:**

Prerequisites

Python 3.8+

**Required Libraries:**

pip install numpy pandas scikit-learn surprise torch transformers streamlit fastapi uvicorn
**Datasets:** Ensure that movies_metadata.csv and ratings.csv are available in the dataset directory.

**Train the Recommendation Model:**

python train_model.py

**Run the Chatbot Interface:**

streamlit run webapp.py

**Start Backend API (FastAPI):**

uvicorn api:app --host 0.0.0.0 --port 8000

**Evaluation Metrics**

**The system's performance is evaluated using:**

**Mean Squared Error (MSE):** Measures prediction accuracy.

**Mean Absolute Error (MAE):** Evaluates deviation from actual ratings.

**Precision & Accuracy:** Determines the relevance of recommendations.

**Future Improvements**

Fine-Tuning LLMs: Enhancing chatbot responses through domain-specific training.

Scalability for Streaming Platforms: Deployment on cloud-based architectures for real-world applications.

**Contributors**

Soham Chakraborty
Vaishnavi Rajasekaran

Advisor
Prof. Alaa Bhakthi

License
This project is licensed under the MIT License.

This README provides all necessary details for setting up and using the chatbot effectively. Let me know if you'd like any modifications!

