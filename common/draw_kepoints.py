# usage: python draw_kepoints.py --dataset RobotDog
import argparse
from tqdm import tqdm
import cv2
import matplotlib.image as img
from util_common import load_datasets


""" Loading the dataset """


class ARGS:
    def __init__(self, dataset):
        self.dataset = dataset
        self.field_of_validation = "Predictions"
        self.MBW_Iteration = 6
        self.validate_manual_labels = False
        self.img_type = ".jpg"


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="Corgi",
        help="name of dataset",
        required=True,
    )
    parser.add_argument(
        "--vid-name",
        type=str,
        default="output.mp4",
        help="name of video",
        
    )
    parser.add_argument(
        "--fps", type=float, default=23.0, help="fps", 
    )
    opt = parser.parse_args()
    return opt


def create_video_from_images(image_list, output_file, frame_rate=60.0):
    """
    Creates a video from a list of images.

    Parameters:
        image_list (list): List of images (numpy.ndarray).
        output_file (str): Path to the output video file.
        frame_rate (float, optional):
                                Frame rate of the output video. Default is 30.0

    Returns:
        None
    """

    # Check if the image_list is not empty
    if not image_list:
        raise ValueError("The image_list is empty.")

    # Get the dimensions of the first image in the list
    height, width, channels = image_list[0].shape

    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )  # You can also use other codecs like 'XVID'
    video_writer = cv2.VideoWriter(
        output_file, fourcc, frame_rate, (width, height)
    )

    # Write each frame to the video
    for image in tqdm(image_list):
        video_writer.write(image)

    # Release the VideoWriter
    video_writer.release()


def concat_images_side_by_side(image_list):
    """
    Concatenates a list of images side by side.

    Parameters:
        image_list (list): List of images (numpy.ndarray).

    Returns:
        numpy.ndarray: The concatenated image.
    """

    # Check if the image_list is not empty
    if not image_list:
        raise ValueError("The image_list is empty.")

    # Get the height of the tallest image in the list
    max_height = max(image.shape[0] for image in image_list)

    # Resize all images to have the same height (tallest height)
    resized_images = [
        cv2.resize(
            image,
            (int(image.shape[1] * max_height / image.shape[0]), max_height),
        )
        for image in image_list
    ]

    # Concatenate images side by side
    concatenated_image = cv2.hconcat(resized_images)

    return concatenated_image


def draw_joints_on_image(image, joints, connections):
    """
    Draws joints on an image using OpenCV.

    Parameters:
        image (numpy.ndarray): The image on which to draw the joints.
        joints (list): List of x, y joints or keypoints.
        connections (list): List of connections between joints.

    Returns:
        numpy.ndarray: The image with the joints drawn on it.
    """

    # Convert the image to RGB format if it is grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Draw each joint on the image
    for joint in joints:
        x, y = joint
        cv2.circle(
            image, (int(x), int(y)), 5, (0, 255, 0), -1
        )  # Green circle with radius 5

    # Draw the connections between joints
    for connection in connections:
        start_point = tuple(map(int, joints[connection[0]]))
        end_point = tuple(map(int, joints[connection[1]]))
        cv2.line(
            image, start_point, end_point, (0, 0, 255), 2
        )  # Red line with thickness 2

    return image


def draw(images, bboxes, joints):
    annotated_images = []
    for i, image in enumerate(images):
        output_image = draw_joints_on_image(image, bboxes[i], joints)
        annotated_images.append(output_image)
    conacted_im = concat_images_side_by_side(annotated_images)
    return conacted_im


def run(dataset, vid_name, fps):
    args = ARGS(dataset)
    loaded_data, misc_data = load_datasets(args)

    print("--- DATASET STATISTICS --- ")
    print("Dataset loaded: {}".format(args.dataset))
    print("Total views: {}".format(misc_data["total_views"]))
    print("Total frames: {}".format(misc_data["total_frames"]))
    print("# of joints: {}".format(misc_data["num_joints"]))
    print("Groundtruth available: ", misc_data["GT_Flag"])

    multi_views_visualize = True
    view_to_visualize = 0
    concated_ims = []
    print("Creating frames")
    for i in tqdm(range(misc_data["total_frames"])):
        """2D Joints"""
        images = []
        if multi_views_visualize:
            for cam_idx in range(misc_data["total_views"]):
                images.append(
                    cv2.imread(str(misc_data["image_paths"][cam_idx][i]))
                )
        else:
            images.append(
                img.imread(misc_data["image_paths"][view_to_visualize][i])
            )

        concated_im = draw(
            images,
            loaded_data["W_Pred"][:, i, :, :],
            misc_data["joint_connections"],
        )
        concated_ims.append(concated_im)
    print("creating video")
    # Create the video from the concatenated image
    create_video_from_images(concated_ims, vid_name, fps)
    print(f'Video saved at {vid_name}')


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
