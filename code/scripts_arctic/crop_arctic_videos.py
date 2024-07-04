import cv2
import os
import glob
import random
import argparse
from tqdm import tqdm


def random_color():
    return (
        random.randint(100, 255),
        random.randint(100, 255),
        random.randint(100, 255),
    )


def click_and_crop(event, x, y, flags, param):
    global refPt, cropping, image
    global refPt_crop

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False

        # Ensure the rectangle is drawn from top-left to bottom-right
        refPt = [
            (min(refPt[0][0], refPt[1][0]), min(refPt[0][1], refPt[1][1])),
            (max(refPt[0][0], refPt[1][0]), max(refPt[0][1], refPt[1][1])),
        ]

        # scale the coordinates back to the original image size
        refPt_crop = [
            (int(refPt[0][0] / scale), int(refPt[0][1] / scale)),
            (int(refPt[1][0] / scale), int(refPt[1][1] / scale)),
        ]

        # Draw a rectangle around the region of interest
        rect_color = random_color()
        cv2.rectangle(image, refPt[0], refPt[1], rect_color, 2)
        cv2.imshow("image", image)


def main(input_dir, output_dir):
    global image, refPt, cropping

    # Check if output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all image files in the directory, sorted by name
    global image_files
    image_files = sorted(
        glob.glob(os.path.join(input_dir, "*.jpg"))
    )  # Adjust the pattern if necessary

    # Select a random image to display
    random_image_file = random.choice(image_files)
    image = cv2.imread(random_image_file)
    clone = image.copy()
    # resize image to 1000pix max dim, preserving aspect ratio
    global scale
    scale = 1000 / max(image.shape)
    image = cv2.resize(
        image, (int(image.shape[1] * scale), int(image.shape[0] * scale))
    )

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # Press 'c' to crop and process all images
        if key == ord("c"):
            if len(refPt_crop) == 2:
                # Crop each image in the folder using the refined coordinates
                for img_file in tqdm(image_files):
                    img = cv2.imread(img_file)
                    roi = img[
                        refPt_crop[0][1] : refPt_crop[1][1],
                        refPt_crop[0][0] : refPt_crop[1][0],
                    ]
                    cv2.imwrite(
                        os.path.join(output_dir, os.path.basename(img_file)), roi
                    )
                print(f"All images have been cropped and saved to {output_dir}")

        # Press 'r' to reset the selection
        elif key == ord("r"):
            image = clone.copy()
            refPt = []

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop all images in a directory.")
    parser.add_argument(
        "--input_dir", required=True, help="Directory containing images to crop."
    )
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(args.input_dir), "images")
    main(args.input_dir, output_dir)
