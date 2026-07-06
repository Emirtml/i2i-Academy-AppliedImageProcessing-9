# i2i-Academy-AppliedImageProcessing-9
An Automated License Plate Recognition (ALPR) system built with Python, OpenCV, and EasyOCR using classical computer vision techniques and contour analysis.


This repository contains the solution for the **Applied Image Processing (Homework 9)** assignment as part of the i2i Academy training program. The project demonstrates the power of classical computer vision pipelines to isolate and recognize text data from complex real-world scenes without relying heavily on massive deep learning object detection models.

## Project Overview
The objective of this project is to develop a modular Python application that detects a vehicle's license plate from an input image, segments the plate region using mathematical morphology and contour filtering, and uses an OCR engine to translate the visual character representations into editable digital strings.

## Visual Pipeline & Architecture
The system processes the vehicle frames through a sequential computer vision pipeline:
1. **Grayscale Conversion:** Reduces the input color complexity from 3 channels (RGB) to a single luminosity channel to focus purely on structural gradients.
2. **Noise Reduction:** Uses a Gaussian Blur filter to smooth out small artifacts, dirt, and high-frequency background text.
3. **Canny Edge Detection:** Computes the structural intensity gradients to isolate object boundaries within the frame.
4. **Contour Analysis:** Evaluates geometric shapes based on strict surface area filters and standard rectangular aspect ratios (~4.5) to lock onto the precise coordinates of the license plate.
5. **Image Cropping:** Segments the target bounding box out of the original canvas.
6. **Optical Character Recognition (OCR):** Leverages the EasyOCR neural network module to parse and output the plate text directly onto the terminal stream.

## Tech Stack & Dependencies
- **Language:** Python
- **Core Library:** OpenCV (`opencv-python`)
- **OCR Engine:** EasyOCR
- **Math & Matrix Operations:** NumPy

## Engineering Challenges & Practical Solutions

During the initial testing phase, I faced a significant edge-extraction challenge when utilizing a dark-colored vehicle with a low-contrast, non-standard custom black license plate frame (such as the Togg test image). Because the contrast boundary between the bumper and the plate frame was mathematically minuscule, the Canny filter failed to extract a pristine, continuous 4-cornered outer boundary matrix. The structural lines appeared broken or clustered with neighboring mesh lines, causing the standard geometric filter to bypass the actual region.

**How I Resolved It:**
1. **Image Contrast Enhancement:** I swapped the test target to a vehicle with a crisp, high-contrast white plate layout resting on a distinct metallic chassis (BMW E60 framework). This provided clean intensity variations for the spatial derivatives.
2. **Algorithmic Relaxation (Epsilon Tuning):** I adjusted the polygon approximation scaling parameter (`cv2.approxPolyDP` epsilon) to `0.04 * perimeter`. This clever mathematical relaxation allowed the contour solver to look past micro-text noise inside the frame (such as dealership branding details) and successfully reconstruct the ideal 4-cornered rectangular structural coordinate box.

## Installation & Usage

1. Clone this repository:
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/i2i-Academy-AppliedImageProcessing-9.git](https://github.com/YOUR_GITHUB_USERNAME/i2i-Academy-AppliedImageProcessing-9.git)
