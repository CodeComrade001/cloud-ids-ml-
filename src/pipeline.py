from src.data_loader import clean_dataset_1, clean_dataset_2, clean_dataset_3
from src.preprocessing import preprocess, split_data
from src.feature_engineering import apply_feature_engineering
from src.dimensionality import apply_pca
from src.models import get_model
from src.trainer import train
from src.evaluator import evaluate


def run_pipeline(config, model_name):

    # Datasets Cleaning
    cleaned_1 = clean_dataset_1("data/raw/cybersecurity_intrusion_data.csv", "data/cleaned/cybersecurity_intrusion_data_cleaned.csv") 
    cleaned_2 = clean_dataset_2("data/raw/Network_logs.csv", "data/cleaned/Network_logs_cleaned.csv") 
    cleaned_3 = clean_dataset_3("data/raw/IoT_Intrusion.csv", "data/cleaned/IoT_Intrusion_cleaned.csv") 

    # if cleaned_1["status"] != "success":
    #     print(f"Dataset 1 cleaning failed: {cleaned_1}")
    if cleaned_2["status"] != "success":
        print(f"Dataset 2 cleaning failed: {cleaned_2}")
    # if cleaned_3["status"] != "success":
    #     print(f"Dataset 3 cleaning failed: {cleaned_3}")
        
        

    # df = load_data("data/raw/dataset.csv")
    # df = preprocess(df)

    # X_train, X_test, y_train, y_test = split_data(df)

    # if config["features"]:
    #     X_train = apply_feature_engineering(X_train)
    #     X_test = apply_feature_engineering(X_test)

    # if config["pca"]:
    #     X_train, X_test = apply_pca(X_train, X_test)

    # model = get_model(model_name)

    # params = {}  # add grid params later

    # trained_model = train(model, params, X_train, y_train)

    # results = evaluate(trained_model, X_test, y_test)

    return { "status": "success" , "data_cleaned": [cleaned_1, cleaned_2, cleaned_3] }
