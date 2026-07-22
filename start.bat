@echo off
echo Starting ResearchPilot Agent...

:: Activate virtual environment
call .\venv\Scripts\activate

:: Install dependencies (skips if already installed)
pip install -r requirements.txt --quiet

:: Run the Streamlit app
streamlit run app.py
