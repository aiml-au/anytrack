# python visualize.py --data /home/aaron-peter/data/mbw/synced_jpeg/synced_jpeg


import argparse
import json
from pathlib import Path
import fiftyone as fo


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images", type=str, default="", help="path to views", required=True
    )
    parser.add_argument(
        "--annot", type=str, default="", help="path to views", required=True
    )
    opt = parser.parse_args()
    return opt


def viz(images, annot):
    images_path = Path(images)
    json_file_path = Path(annot)

    dataset = fo.Dataset.from_images_dir(images_path)

    with open(json_file_path) as file:
        json_data = json.load(file)

    fiftyone_dict = {}
    for item in json_data:
        if item.get("img"):
            img_name = item["img"].split("-")[-1]
        else:
            raise ValueError(f"Missing key img {json_file_path}")
        fiftyone_dict[img_name] = []

        kps = item.get("kp-1", [])
        for kp in kps:
            pixel_x = kp.get("x") * kp.get("original_width") / 100.0
            pixel_y = kp.get("y") * kp.get("original_height") / 100.0
            fiftyone_dict[img_name].append(
                (
                    pixel_x / kp.get("original_width"),
                    pixel_y / kp.get("original_height"),
                )
            )

    for sample in dataset:
        filename = sample.filepath.split("/")[-1]
        sample["keypoints"] = fo.Keypoint(points=fiftyone_dict[filename])
        sample.save()

    session = fo.launch_app(dataset)  # (optional) port=XXXX
    # Blocks execution until the App is closed
    session.wait()


if __name__ == "__main__":
    opt = parse_opt()
    viz(**vars(opt))
