from PIL import Image
import cv2
import os


def init_dataset(args):
    video_path = args.video_path
    k = args.skip_every
    seq_name = video_path.split("/")[2]

    # Path to the directory where frames will be saved
    output_dir = f"./data/{seq_name}/processed/raw_images/"

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        # Check if the frame is retrieved successfully
        if not ret:
            break
        # Save the frame
        frame_filename = os.path.join(output_dir, f"{frame_number:04}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_number += 1

    cap.release()
    print(f"Extracted {frame_number} frames.")

    # Directory containing the extracted frames
    source_dir = f"./data/{seq_name}/processed/raw_images/"
    # Directory where every kth frame will be saved and resized
    target_dir = f"./data/{seq_name}/images/"
    # Interval for frames to be copied

    # remove target_dir if exists
    if os.path.exists(target_dir):
        import shutil

        shutil.rmtree(target_dir)

    # Maximum dimension
    max_dim = 2000

    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # List all frame files in the source directory, sorted to maintain order
    frame_files = sorted(os.listdir(source_dir))

    # Copy every kth frame to the target directory and resize it
    num_selected = 0
    for i, frame_file in enumerate(frame_files):
        if i % k == 0:  # Check if the frame is the kth frame
            # Construct full file paths
            source_path = os.path.join(source_dir, frame_file)
            target_path = os.path.join(target_dir, frame_file)

            # Open the image
            with Image.open(source_path) as img:
                if img.size[0] > max_dim or img.size[1] > max_dim:
                    # Calculate the scaling factor
                    scale_factor = min(max_dim / img.size[0], max_dim / img.size[1])
                    # Calculate new size, maintaining aspect ratio
                    new_size = (
                        int(img.size[0] * scale_factor),
                        int(img.size[1] * scale_factor),
                    )
                    # Resize the image
                    # resized_img = img.resize(new_size, Image.ANTIALIAS)
                    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                else:
                    resized_img = img

                # Save the resized image
                resized_img.save(target_path)
                num_selected += 1

    print(
        f"Copied and resized every {k}th frame to {target_dir}, leading to {num_selected} frames."
    )

    directories_to_create = [
        f"./data/{seq_name}/processed/sam/right",
        f"./data/{seq_name}/processed/sam/object",
        f"./data/{seq_name}/processed/sam/left",
    ]

    # Create each directory if it doesn't already exist
    for directory in directories_to_create:
        if not os.path.exists(directory):
            os.makedirs(directory)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--skip_every", type=int, default=2)
    args = parser.parse_args()
    init_dataset(args)
