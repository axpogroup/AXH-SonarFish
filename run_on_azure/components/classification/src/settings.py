import argparse
from copy import deepcopy
from pathlib import Path
from typing import List, Type, Union

import yaml
from pydantic import BaseModel, field_validator

from analysis.classification_utils.classifier_evaluation import (
    ProbaLogisticRegression,
    ProbaXGBClassifier,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Classify the Tracked Objects")
    parser.add_argument("--classification_settings_file", required=True)
    parser.add_argument("--train_val_gt_data_dir", required=True)
    parser.add_argument("--files_to_classify_dir", required=True)
    parser.add_argument("--classified_detections_dir", required=True)
    parser.add_argument("--train_val_data_yaml", default="fish_labels.yaml")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    return args


def load_yaml_config(file_path: Union[str, Path]) -> dict:
    with open(Path(file_path), "r") as file:
        return yaml.safe_load(file)


class ClassificationSettings(BaseModel):
    classifier_name: str
    min_track_length: int
    proba_threshold: float
    random_state: int
    features_to_use: List[str]

    @field_validator("classifier_name")
    def validate_classifier_name(cls, value: str) -> str:
        valid_classifiers = ["ProbaLogisticRegression", "ProbaXGBClassifier"]
        if value not in valid_classifiers:
            raise ValueError(f"Invalid classifier specified. Choose from {valid_classifiers}")
        return value

    @property
    def classifier(self) -> Type[Union[ProbaLogisticRegression, ProbaXGBClassifier]]:
        if self.classifier_name == "ProbaLogisticRegression":
            return ProbaLogisticRegression(
                proba_threshold=self.proba_threshold,
                class_weight="balanced",
            )
        elif self.classifier_name == "ProbaXGBClassifier":
            return ProbaXGBClassifier(
                proba_threshold=self.proba_threshold,
                random_state=self.random_state,
                verbosity=0,
            )
        else:
            raise ValueError("Invalid classifier specified.")


args = parse_arguments()
yaml_config = load_yaml_config(Path(__file__).parent / args.classification_settings_file)
classification_settings = ClassificationSettings.model_validate(yaml_config)

# Example usage
if __name__ == "__main__":
    classifier_instance = deepcopy(classification_settings.classifier)

    print(f"Classifier Type: {classifier_instance.__class__.__name__}")
    print("Settings:")
    print(f"Probability Threshold: {classification_settings.proba_threshold}")
    print(f"Random State: {classification_settings.random_state}")
    print(f"Features to Use: {classification_settings.features_to_use}")
