from experiments.results_merge import merge_summary_results
from experiments.runner import run_all
from src.datasets_cleaner import clean_iot_intrusion_dataset

# folder_path = r"C:/Users/Dinero/Desktop/Final_Year_Project/cloud-ids-ml/results/summary"
if __name__ == "__main__":
    run_all()
    
    # merge_summary_results(folder_path)
