import json
import torch
from torch import nn
from torchvision import transforms

INPUT_JSON = "./sample_input.json"
OUTPUT_JSON = "./sample_output.json"


def load_img(path: str) -> torch.Tensor:
    pass

class OODDetector:
    def __init__(self) -> None:
        super().__init__()
        self.model: nn.Module
        self.transform: transforms.Compose

    def fit(self, input_json: str) -> None:
        with open(input_json, "r") as fp:
            train_images = json.load(fp)["train_images"]
        pass

    def predict(self, input_json: str) -> dict:
        with open(input_json, "r") as fp:
            test_images = json.load(fp)["test_images"]

        output = {}
        for x in test_images:
            input = self.transform(load_img(x))
            score = self.model(input).item()
            output.update({x: score})
        return output


if __name__ == "__main__":

    ood_detector = OODDetector()
    output = ood_detector.predict(INPUT_JSON)

    with open(OUTPUT_JSON, "w") as fp:
        json.dump(output, fp, indent=4)