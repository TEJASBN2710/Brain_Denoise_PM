# 🧠 Brain MRI Denoising using Perona – Malik Anisotropic Diffusion

A Python-based image processing project for denoising brain MRI scans using **Perona–Malik Anisotropic Diffusion**. The algorithm reduces unwanted noise while preserving important anatomical structures and edges.

## 📌 Project Overview

Medical images such as MRI scans can contain noise that affects image quality and subsequent analysis.

This project implements **Perona–Malik Anisotropic Diffusion**, an edge-preserving smoothing technique that selectively reduces noise while maintaining significant image boundaries.

Unlike conventional smoothing methods that blur the entire image, anisotropic diffusion smooths relatively homogeneous regions while reducing diffusion near important edges.

## 🎯 Objectives

- Reduce noise in brain MRI images.
- Preserve important anatomical edges and structures.
- Improve MRI image quality.
- Implement Perona–Malik anisotropic diffusion using Python.
- Visualize the original and denoised images.
- Provide a preprocessing method for further medical image analysis.

## ⚙️ Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib

## 🔬 Processing Pipeline

```text
Input Brain MRI
       │
       ▼
Image Preprocessing
       │
       ▼
Perona–Malik Anisotropic Diffusion
       │
       ▼
Noise Reduction
       │
       ▼
Edge Preservation
       │
       ▼
Denoised Brain MRI
       │
       ▼
Display / Save Result
```

## 🧮 Perona–Malik Anisotropic Diffusion

The diffusion process controls smoothing according to local image gradients.

The general diffusion equation is:

```text
∂I/∂t = div(c(|∇I|) ∇I)
```

Where:

- `I` = image intensity
- `t` = diffusion time
- `∇I` = image gradient
- `c()` = diffusion coefficient
- `div` = divergence operator

The diffusion coefficient controls how strongly pixels are smoothed depending on local gradients. Diffusion is reduced around strong gradients, helping preserve important image edges.

## 📂 Project Structure

```text
Brain-MRI-Denoising/
│
├── main.py
├── README.md
├── requirements.txt
│
├── input/
│   └── brain_mri.jpg
│
├── output/
│   └── denoised_brain_mri.jpg
│
└── results/
    └── comparison.png
```

> The folder structure can be modified according to your implementation.

```
📦 Requirements -
opencv-python
numpy
matplotlib
```

## ⚙️ Algorithm Parameters

The diffusion process can be controlled using parameters such as:

| Parameter | Description |
|---|---|
| `iterations` | Number of diffusion iterations |
| `delta` | Time step controlling diffusion |
| `kappa` | Controls sensitivity to image gradients |

Example:

```python
anisotropic_diffusion(
    image,
    iterations=25,
    delta=0.15,
    kappa=30
)
```

These parameters can be adjusted depending on the MRI image and desired denoising level.

## 🔍 Key Features

- ✅ Brain MRI image processing
- ✅ Perona–Malik anisotropic diffusion
- ✅ Noise reduction
- ✅ Edge preservation
- ✅ Adjustable diffusion parameters
- ✅ Python and OpenCV implementation
- ✅ Matplotlib visualization

## 📊 Results

The processed MRI should show reduced noise while retaining important structural boundaries.

A typical comparison is:

```text
Original MRI  →  Perona–Malik Diffusion  →  Denoised MRI
```

## 📈 Advantages

- Preserves important image edges.
- Reduces unwanted noise.
- Simple to implement and experiment with.
- Does not require a trained deep-learning model.
- Can be used as a preprocessing step for further image analysis.

## ⚠️ Limitations

- Results depend on the selected diffusion parameters.
- Excessive iterations may remove useful image details.
- Performance varies with image characteristics and noise levels.
- This project is intended for educational and research purposes and is **not a clinical diagnostic system**.

## 🔮 Future Enhancements

- Compare with Gaussian filtering.
- Compare with Median filtering.
- Compare with Bilateral filtering.
- Compare with Non-Local Means denoising.
- Evaluate using **PSNR** and **SSIM**.
- Automatic parameter optimization.
- MRI segmentation after denoising.
- Integration with machine-learning or deep-learning models.
- Development of a web-based MRI denoising interface.

## 📚 Applications

- Brain MRI preprocessing
- Medical image enhancement
- Image segmentation preprocessing
- Anatomical structure preservation
- Computer vision research
- Medical image analysis

## 👨‍💻 Author

**Tejas B N**

Computer Science / Engineering Student

## ⭐ Project Purpose

This project demonstrates the application of **Perona – Malik Anisotropic Diffusion for brain MRI denoising**, focusing on reducing image noise while maintaining important structural information.

If you find this project useful, consider giving the repository a ⭐.

## 📄 License

This project is intended for educational and research purposes.
