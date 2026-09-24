import cv2
import numpy as np
import matplotlib.pyplot as plt

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
# OPEN CAMERA
# =====================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera could not be opened!")
    exit()

print("==============================================")
print("      MRI IMAGE DENOISING SYSTEM")
print("==============================================")
print("Camera opened.")
print("Press SPACE to capture ONE image.")
print("Press Q to quit.")
print("==============================================")


# =====================================================
# WAIT FOR MANUAL CAPTURE
# =====================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read camera.")
        break

    # Show live camera only
    cv2.imshow("Camera - Press SPACE to Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    # SPACE = capture
    if key == 32:

        print("\nImage captured!")

        # Convert captured image to grayscale
        original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        break

    # Q = quit
    elif key == ord('q'):

        print("Program cancelled.")
        cap.release()
        cv2.destroyAllWindows()
        exit()


# =====================================================
# CLOSE CAMERA
# =====================================================

cap.release()
cv2.destroyAllWindows()


# =====================================================
# APPLY ANISOTROPIC DIFFUSION
# =====================================================

print("Applying Perona-Malik Anisotropic Diffusion...")

denoised = anisotropic_diffusion(
    original,
    iterations=25,
    delta=0.15,
    kappa=30
)


# =====================================================
# SAVE IMAGES
# =====================================================

cv2.imwrite("captured_image.jpg", original)
cv2.imwrite("denoised_mri.jpg", denoised)

print("Original image saved as: captured_image.jpg")
print("Denoised image saved as: denoised_mri.jpg")


# =====================================================
# DISPLAY FINAL OUTPUT
# =====================================================

plt.figure(figsize=(10, 5))

# Original
plt.subplot(1, 2, 1)
plt.imshow(original, cmap="gray")
plt.title("Captured Image")
plt.axis("off")

# Denoised
plt.subplot(1, 2, 2)
plt.imshow(denoised, cmap="gray")
plt.title("Anisotropic Diffusion")
plt.axis("off")

plt.tight_layout()
plt.show()


print("\n==============================================")
print("             PROCESS COMPLETED")
print("==============================================")