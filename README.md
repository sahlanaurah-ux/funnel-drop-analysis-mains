Funnel Drop-Off Analysis & E-Commerce Dashboard
A data-driven machine learning and interactive visualization platform designed to track end-to-end customer conversion journeys, analyze stage-by-stage drop-off rates, and predict user purchase behavior on e-commerce platforms.
📌 Features
 * 5-Stage Conversion Funnel Tracking: Maps user navigation across Home Page \rightarrow Product View \rightarrow Add to Cart \rightarrow Checkout \rightarrow Purchase / Confirmation.
 * Drop-Off & Cart Abandonment Analysis: Pinpoints critical conversion bottlenecks, session durations, and bounce rates.
 * Machine Learning Purchase Prediction: Utilizes Random Forest and Gradient Boosting Regression models to evaluate drop-off risks based on features like TimeOnPage, ItemsInCart, SessionDuration_seconds, and HourOfDay.
 * Multi-Segment Analytics: Filters and analyzes metrics across device types (Desktop, Mobile, Tablet) and top geographic regions.
 * Interactive Dashboards: Built using Dash and Streamlit with responsive Plotly Express visualizations.
🛠️ Tech Stack
 * Programming Language: Python
 * Data Processing: Pandas, NumPy
 * Machine Learning: Scikit-Learn
 * Visualization: Plotly Express
 * Web Frameworks: Dash, Streamlit
 * Database Systems: MySQL / PostgreSQL
📁 Repository Structure
├── data/
│   ├── raw_sessions.csv
│   └── processed_funnel_metrics.csv
├── models/
│   ├── random_forest_model.pkl
│   └── gradient_boosting_model.pkl
├── pages/
│   ├── 1_Funnel_Analysis.py
│   ├── 2_ML_Predictions.py
│   └── 3_Customer_Segmentation.py
├── app.py
├── requirements.txt
└── README.md

🚀 Getting Started
1. Prerequisites
Ensure Python 3.8+ is installed on your system.
2. Clone the Repository
git clone https://github.com/your-username/funnel-drop-analysis.git
cd funnel-drop-analysis

3. Create a Virtual Environment & Install Dependencies
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

4. Database Setup
Configure your database connection parameters in config.py or .env to connect with your MySQL/PostgreSQL instance.
5. Run the Application
Launching Streamlit Dashboard:
streamlit run app.py

Launching Dash Dashboard:
python app.py

📊 Model Training & Machine Learning
To re-train the predictive models on new session data:
python train_model.py

The pipeline evaluates session features (such as TimeOnPage, SessionDuration_seconds, Pageviews, and HourOfDay) to classify high-risk drop-offs and predict purchase conversion probabilities.
👩‍💻 Author
 * Sahla Nawra VP (Reg No: CUAYMCS020)
 * Master of Science in Computer Science
 * Centre for Computer Science and Information Technology (CCSIT), University of Calicut