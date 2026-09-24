import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# =====================================================
# PERONA-MALIK ANISOTROPIC DIFFUSION
# =====================================================

def anisotropic_diffusion(img, iterations=25, delta=0.15, kappa=30):

    img = img.astype(np.float32)

    for _ in range(iterations):

        north = np.zeros_like(img)
        south = np.zeros_like(img)
        east = np.zeros_like(img)
        west = np.zeros_like(img)

        north[:-1, :] = img[1:, :] - img[:-1, :]
        south[1:, :] = img[:-1, :] - img[1:, :]

        east[:, :-1] = img[:, 1:] - img[:, :-1]
        west[:, 1:] = img[:, :-1] - img[:, 1:]

        cN = np.exp(-(north / kappa) ** 2)
        cS = np.exp(-(south / kappa) ** 2)
        cE = np.exp(-(east / kappa) ** 2)
        cW = np.exp(-(west / kappa) ** 2)

        img += delta * (
            cN * north +
            cS * south +
            cE * east +
            cW * west
        )

    return np.clip(img, 0, 255).astype(np.uint8)

# =====================================================
# LOAD MRI IMAGE
# =====================================================

image_path = "scan.jpg"

original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if original is None:
    print("Error: MRI image not found!")
    exit()

# =====================================================
# APPLY ANISOTROPIC DIFFUSION DIRECTLY
# =====================================================

denoised = anisotropic_diffusion(
    original,
    iterations=25,
    delta=0.15,
    kappa=30
)

# =====================================================
# EVALUATION
# =====================================================

psnr_value = peak_signal_noise_ratio(original, denoised)
ssim_value = structural_similarity(original, denoised)

print("\n========== RESULTS ==========")
print(f"PSNR (Denoised Image): {psnr_value:.2f} dB")
print(f"SSIM (Denoised Image): {ssim_value:.4f}")
print("=============================\n")

# =====================================================
# SAVE OUTPUT
# =====================================================

cv2.imwrite("denoised_mri.jpg", denoised)
print("Denoised image saved as denoised_mri.jpg")

# =====================================================
# DISPLAY RESULTS
# =====================================================

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(original, cmap='gray')
plt.title("Original MRI")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(denoised, cmap='gray')
plt.title("Denoised MRI")
plt.axis('off')

plt.tight_layout()
plt.show()