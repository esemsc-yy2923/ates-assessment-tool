
import streamlit.web.cli as stcli
from pathlib import Path

def main():
    
    app_path = Path(__file__).parent / "main_app.py"
    
    args = ["run", str(app_path)] 
    
    stcli.main(args)

if __name__ == "__main__":
    main()