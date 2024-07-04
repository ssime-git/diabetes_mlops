import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

def load_data(reference_path: str, new_data_path: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Load reference and new data from specified paths.
    
    :param reference_path: Path to the reference dataset.
    :param new_data_path: Path to the new dataset.
    :return: A tuple of DataFrames (reference_data, new_data).
    """
    reference_data = pd.read_csv(reference_path)
    new_data = pd.read_csv(new_data_path)
    return reference_data, new_data

def detect_data_drift(reference_data: pd.DataFrame, new_data: pd.DataFrame) -> bool:
    """
    Detect data drift between reference and new datasets.
    
    :param reference_data: DataFrame containing the reference dataset.
    :param new_data: DataFrame containing the new dataset.
    :return: Boolean indicating if drift was detected.
    """
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=reference_data.drop('Outcome', axis=1), 
                          current_data=new_data.drop('Outcome', axis=1), column_mapping=None)
    report_json = data_drift_report.as_dict()
    drift_detected = report_json['metrics'][0]['result']['dataset_drift']
    return drift_detected

def write_drift_status_to_file(status: str, file_path: str) -> None:
    """
    Write the drift detection status to a file.
    
    :param status: The drift status ('drift_detected' or 'no_drift').
    :param file_path: The path to the file where to write the status.
    """
    with open(file_path, 'w') as f:
        f.write(status)

def main():
    reference_path = 'data/processed/reference_from_train_data.csv'
    new_data_path = 'data/processed/new_data.csv'
    status_file_path = 'drift_detected.txt'
    
    reference_data, new_data = load_data(reference_path, new_data_path)
    drift_detected = detect_data_drift(reference_data, new_data)
    
    if drift_detected:
        print("Data drift detected. Retraining the model.")
        write_drift_status_to_file('drift_detected', status_file_path)
    else:
        print("No data drift detected.")
        write_drift_status_to_file('no_drift', status_file_path)

if __name__ == "__main__":
    main()