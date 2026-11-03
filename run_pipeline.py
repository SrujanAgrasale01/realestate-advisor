"""Run the whole project end to end:  python run_pipeline.py"""
from src import eda, preprocess, train

if __name__ == "__main__":
    print("== 1/3 Preprocessing ==")
    preprocess.run()
    print("== 2/3 EDA ==")
    eda.run()
    print("== 3/3 Training + MLflow ==")
    train.main()
    print("\nDone. Launch the app with:  streamlit run app/streamlit_app.py")
