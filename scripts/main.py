
from fire import Fire


detectors_map: Dict[str, Any] = {"yolov4": YoloV4}


def main(base_config_path: str, model_config_path):
    """Entrypoint for the project

    Args:
        base_config_path: path to the desired configuration file
        model_config_path: path to the detection model configuration file

    """
    print("done")

if __name__ == "__main__":
    Fire(main)
