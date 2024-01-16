import argparse
import json
from pathlib import Path
import _pickle as cPickle
import numpy as np


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="", help="path to data folder", required=True
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="name of dataset",
        required=True,
    )
    parser.add_argument(
        "--joints", type=str, default="", help="path to joints", required=True
    )
    opt = parser.parse_args()
    return opt


def create_pickle(data, name, joints):
    data_path = Path(data)
    annotated_imgs = data_path/name/'ann_images'
    views = [x.name for x in annotated_imgs.iterdir() if x.is_dir()]
    annot = annotated_imgs.parent / "annot"
    annot.mkdir(exist_ok=True, parents=True)
    with open(joints) as file:
        joints_name = json.load(file)["joints_name"]

    for view in views:
        # json-min keypoint dataset from labelstudio
        json_file_path = annot / f"{view}.json"
        images_path = annotated_imgs / view
        pckl_data = []
        imgs_data = {}

        with open(json_file_path) as file:
            annotated_data = json.load(file)

        # creat a dict of sll the inf required
        for item in annotated_data:
            if item.get("img"):
                img_name = item["img"].split("-")[-1]
            else:
                raise ValueError(f"Missing key img {json_file_path}")
            imgs_data[img_name] = {}
            joints = {}
            kps = item.get("kp-1", [])
            for kp in kps:
                pixel_x = kp.get("x") * kp.get("original_width") / 100.0
                pixel_y = kp.get("y") * kp.get("original_height") / 100.0
                imgs_data[img_name][kp.get("keypointlabels")[0]] = [
                    int(pixel_x),
                    int(pixel_y),
                ]

        # get annotated images
        ann_img_names = []
        for path in images_path.rglob("*.jpg"):
            ann_img_names.append(Path(path).name)

        all_images = data_path/ name/ 'images'/ view
        all_images_names = [x.name for x in all_images.iterdir()]
        all_images_names = [
            p for p in sorted([str(p) for p in all_images_names])
        ]
        for img_name in all_images_names:
            # if annotated img
            if img_name in ann_img_names:
                img_info = imgs_data[img_name]
                keypoint_data = {"W_GT": [], "confidence": True}
                for joint_name in joints_name:
                    result = img_info.get(joint_name)
                    if result is None:
                        result = [np.NAN, np.NAN]
                    keypoint_data["W_GT"].append(result)
                keypoint_data["W_GT"] = np.array(keypoint_data["W_GT"])
                pckl_data.append(keypoint_data)
            else:
                keypoint_data = {"W_GT": [], "confidence": False}
                for joint_name in joints_name:
                    keypoint_data["W_GT"].append([np.nan, np.nan])
                keypoint_data["W_GT"] = np.array(keypoint_data["W_GT"])
                pckl_data.append(keypoint_data)

        pickle_file = annot / f"{view}.pkl"
        pickle_file.unlink(missing_ok=True)
        with open(pickle_file.as_posix(), "wb") as fid:
            cPickle.dump(pckl_data, fid)
    print(f'Pickle files created at {annot}')

if __name__ == "__main__":
    opt = parse_opt()
    create_pickle(**vars(opt))
