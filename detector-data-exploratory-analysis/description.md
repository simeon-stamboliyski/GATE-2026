# Detector Analysis Program

## Purpose

This C++ program analyses a sequence of MATLAB `.mat` files containing detector frames. It calculates visual movement between neighbouring frames, compares both that movement and the detector-provided `Res.E` value with the threshold `Ini.VThr`, and creates reports, plots, and an annotated video.

## Expected MATLAB data

Each `.mat` file is expected to contain:

- `Res.S`: a three-channel RGB image stored as a MATLAB `double` array.
- `Res.E`: one detector measurement, read as a `double`.
- `Ini.VThr`: the threshold value used for comparisons.
- `Ini.ff`: an optional source-video filename, printed for reference.

## Processing workflow

1. The program finds all `.mat` files in `data_path` and sorts them by filename.
2. It reads `Res.S` from every file and converts it from MATLAB's RGB array layout into an OpenCV image.
3. For every frame after the first, it converts the current and preceding images to grayscale, calculates their absolute pixel difference, and uses the mean difference divided by `255` as the `movement_score`.
4. It reads the detector's `Res.E` value and the file's `Ini.VThr` threshold.
5. It performs two independent comparisons:

   - `movement_score > VThr`: whether the calculated frame-to-frame movement exceeds the threshold.
   - `Res.E > VThr`: whether the detector's own value exceeds the threshold.
6. Movement-threshold exceedances are stored with their filename, frame number, threshold, and amount above the threshold.

## Generated outputs

The program creates the following files in its working directory:

| File | Contents |
| --- | --- |
| `exceedances.csv` | All movement-score exceedances and their values. |
| `movement_analysis.png` | Calculated movement scores (blue), `VThr` (red dashed), and movement exceedances (red markers). |
| `res_e_analysis.png` | Detector `Res.E` values (green), `VThr` (red dashed), and `Res.E` exceedances (red markers). |
| `detector_video_with_analysis.mp4` | A video assembled from `Res.S` frames with frame number, movement score, threshold, and movement-exceeded status overlaid. |

## Important distinction

`movement_score` is calculated by this program from the average brightness change between consecutive images. `Res.E` is read directly from the detector data. They may have different scales and do not necessarily cross `VThr` on the same frames.

The current CSV report, console exceedance report, and video overlay describe **movement-score** exceedances. The `Res.E` comparison is currently visualized in `res_e_analysis.png`.

## Libraries used

- **matio** reads MATLAB `.mat` files.
- **OpenCV** converts images, calculates frame differences, adds overlays, and writes the MP4 video.
- **Matplot++** creates and saves the analysis plots.