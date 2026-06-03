@echo off
echo Installing required packages...
pip install -r requirements.txt
echo.
echo Starting LoanGuard Streamlit Application...
echo.
python -m streamlit run app.py
pause
