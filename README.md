## Medical Assistance System

A Flask-based web application that predicts diseases based on symptoms and provides personalized health recommendations, including medications, diets, and workouts.

Features

Symptom-based disease prediction using a trained SVC model

Personalized medication, diet, and workout suggestions

User-friendly web interface

Integration of multiple health-related datasets

Technologies Used

Python

Flask

Machine Learning (Support Vector Classification - SVC)

Pandas, NumPy

Pickle (for model storage)

HTML/CSS (for templates)

Installation

Clone the repository:

git clone https://github.com/yourusername/medical-assistance.git
cd medical-assistance

Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

Run the application:

python app.py

Open your browser and go to http://127.0.0.1:5000/

Dataset

The project utilizes multiple datasets for symptoms, medications, diets, and precautions. Ensure that all necessary CSV files are placed in the dataset/ directory.

Model Training

If you want to retrain the model, use the medicine recommendation.ipynb notebook to preprocess the data and train the SVC model. Save the trained model as svc.pkl inside the models/ directory.

Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss your ideas.
