import argparse
from pathlib import Path
import numpy as np
import shutil
from tqdm import tqdm


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="", help="path to views", required=True
    )
    parser.add_argument(
        "--name", type=str, default="", help="name of dataset", required=True
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=10.0,
        help="percentage of images to annotate",
        required=True,
    )

    opt = parser.parse_args()
    return opt


def make_dataset(data, name, percent):
    root_path = Path.cwd()
    data_dir = Path(data)
    if not data_dir.exists():
        raise FileNotFoundError
    # get all the views
    views = [x.name for x in data_dir.iterdir() if x.is_dir()]

    dataset_folder = root_path/'data'/ name
    if dataset_folder.exists():
        print('Warning: folder already exists, rewriting..')

    # create a folder to put images ofr annotation
    ann_images = dataset_folder / "ann_images"
    ann_images.mkdir(parents=True, exist_ok=True)

    full_images = dataset_folder / "images"
    full_images.mkdir(parents=True, exist_ok=True)

    annot = dataset_folder / "annot"
    annot.mkdir(parents=True, exist_ok=True)

    for i, view in enumerate(views):
        view_folder = ann_images / f"CAM_{i+1}"
        view_folder.mkdir(parents=True, exist_ok=True)

        full_images_view_folder = full_images / f"CAM_{i+1}"
        full_images_view_folder.mkdir(parents=True, exist_ok=True)

        images_path = data_dir / view
        # get all images
        img_names = []
        for path in images_path.rglob("*.jpg"):
            img_names.append(path)
        img_names = [Path(p) for p in sorted([str(p) for p in img_names])]
        print("Total frames in the video are: ", len(img_names))
        # choose images
        annotation_indxs = np.arange(
            0,
            int(len(img_names)),
            int(
                int(len(img_names))
                / ((int(len(img_names)) * int(percent)) / 100)
            ),
        )
        # this will give you index of images to annotate
        annotation_samples = [img_names[indx] for indx in annotation_indxs]
        print(f"Need to annotate {len(annotation_samples)}")
        print(f"Copying images for CAM_{i+1}")
        for sample in tqdm(annotation_samples):
            shutil.copy(
                sample.as_posix(), (view_folder / sample.name).as_posix()
            )
        for img_name in tqdm(img_names):
            shutil.copy(
                img_name.as_posix(),
                (full_images_view_folder / img_name.name).as_posix(),
            )

        with open("rename.txt", "a+") as file:
            file.write(f"{images_path.name} : CAM_{i+1} \n")

    print(f'Dataset created at {dataset_folder}')
if __name__ == "__main__":
    opt = parse_opt()
    make_dataset(**vars(opt))
