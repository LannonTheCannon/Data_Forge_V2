Fraud Detection Dashboard

Overview

This project is a Fraud Detection Dashboard designed to analyze and visualize credit card fraud transactions. The application is built with Streamlit, PandasAI, and GPT-4 Vision, offering users the ability to explore fraud datasets, generate visualizations, and interact with an AI assistant for insights.

Features

🔹 Dataset Overview

Preview the dataset (first 5 rows).

Display quick statistics (df.describe()).

Show dataset structure (df.info()).

🧠 Assistant Chat (RAG Chatbot)

AI-powered chatbot for credit card fraud insights.

Context-aware responses with OpenAI’s Assistant API.

Persistent chat history using Streamlit session state.

📊 PandasAI Insights

Query-based data analysis using PandasAI.

AI-generated Python code for visualizations.

Sample questions to guide analysis.

GPT-4 Vision interpretation of generated charts.

✍️ Code Editor & Execution

Modify AI-generated Python code using Streamlit ACE Editor.

Execute the modified code within Streamlit.

Display custom plots and insights.

📜 Documentation

Project overview, technical details, and usage instructions.

🛠️ Technologies Used

Python (Pandas, Matplotlib, Seaborn, OpenAI API, SQLite)

Streamlit (UI framework for interactive applications)

PandasAI (AI-powered dataframe analysis)

GPT-4 Vision API (Image interpretation for visual insights)

🚀 Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/your-repo/fraud-detection-dashboard.git
cd fraud-detection-dashboard

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Set Up OpenAI API Key

Create a .streamlit/secrets.toml file and add:

[secrets]
OPENAI_API_KEY = "your-openai-api-key"

4️⃣ Run the Application

streamlit run app.py

📌 Sample Visualization Questions

These are some powerful queries you can ask:

Show a violin plot of transaction amounts for fraud vs. non-fraud.

Plot a scatter chart of transaction amount vs. account age, colored by fraud label.

Show a boxplot of transaction amounts grouped by merchant type, highlighting fraud.

Create a bar chart of fraudulent transactions by hour of the day.

Use a heatmap to explore correlations between transaction risk factors.

🎯 Roadmap

✅ Implement PandasAI for data-driven insights

✅ Add GPT-4 Vision for chart analysis

⏳ Expand fraud detection algorithms (Machine Learning models)

⏳ Enhance UI with improved interactivity

🤝 Contributing

Want to improve this project? Feel free to fork and submit a PR!

📜 License

This project is open-source under the MIT License.

📞 Contact

For inquiries, reach out at: your.email@example.com
