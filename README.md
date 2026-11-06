# Iris Segmentation

A Google Colab notebook that isolates the iris from a face photo. It detects the face, crops both eyes, then segments the iris with classical circle detection. A second path uses Meta's Segment Anything model (SAM) on the cropped right eye.

Notebook: `Iris_segmentation_using_Dlib_hough_and_McctN.ipynb`

## What it does

The notebook runs in this order:

1. **Face crop (MTCNN).** Loads `/content/face_image.jpeg`, detects the face, draws a box, and saves the crop as `detected_face.jpg`.
2. **Eye crop (dlib).** Loads dlib's 68-point face landmark model, finds the left eye (points 36–41) and right eye (points 42–47), adds a 10-pixel pad, and saves `left_eye.jpg` and `right_eye.jpg`.
3. **Iris and pupil (Hough circles).** On the left eye:
   - Converts the crop to grayscale, blurs it, and boosts contrast with CLAHE.
   - Tries several Hough circle settings until it finds an iris, then a smaller pupil inside it.
   - If the pupil is too far from the iris center, or larger than the iris, it estimates the pupil as 40% of the iris radius and shares the iris center.
   - Builds a donut mask (iris disk minus pupil disk) and keeps only those pixels.
   - Unwraps the ring into a rectangle (360 columns, one row per pixel of iris radius) and enhances color in LAB space.
   - Saves the unwrapped iris as `iris_only.jpg`.
4. **Optional SAM segmentation.** Clones [segment-anything](https://github.com/facebookresearch/segment-anything), downloads the ViT-H checkpoint (`sam_vit_h_4b8939.pth`, about 2.4 GB), prompts the center of `right_eye.jpg`, and saves the masked region as `iris_extracted1.png`.

A recorded run on the left-eye crop (90×45 px) reported an iris at center `(34, 41)` with radius `34`.

## Requirements

Designed for Google Colab (Python 3). The notebook installs:

- `opencv-python`
- `mtcnn`
- `dlib`
- `numpy`
- `scikit-image`
- `torch` (used by SAM; present in Colab)
- Segment Anything, installed from the cloned repo

It also downloads:

- [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

## How to run

1. Open the notebook in Google Colab.
2. Upload a front-facing face photo and name it `face_image.jpeg` in `/content`, or change the path in the first cell.
3. Run the cells from top to bottom. The face cell must run before the eye cell, and the eye cell must run before the iris cell.
4. Skip the last four cells if you only want the Hough-circle result. Those cells clone SAM and download the 2.4 GB checkpoint.

## Inputs and outputs

| File | Role |
| --- | --- |
| `/content/face_image.jpeg` | Input face photo |
| `detected_face.jpg` | Cropped face |
| `left_eye.jpg`, `right_eye.jpg` | Eye crops |
| `iris_only.jpg` | Unwrapped, color-enhanced left iris |
| `/content/iris_extracted1.png` | SAM mask applied to the right eye |

## Notes

- Paths are hardcoded for Colab (`/content/...`). On a local machine, point them at your own files and place `shape_predictor_68_face_landmarks.dat` where the dlib cell expects it.
- Hough parameters are tuned for a small eye crop. A different image size or lighting often needs a new `minRadius` / `maxRadius` range.
- The SAM cell takes the first of three masks from a single center point. That mask is the region around the click, which may include sclera or eyelids as well as the iris.
- The landmark file is about 61 MB compressed. The SAM ViT-H weights are about 2.4 GB.
