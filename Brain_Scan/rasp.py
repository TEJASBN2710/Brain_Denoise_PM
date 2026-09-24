import cv2
import numpy as np

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

        # Neighbor differences
        north[:-1, :] = img[1:, :] - img[:-1, :]
        south[1:, :] = img[:-1, :] - img[1:, :]

        east[:, :-1] = img[:, 1:] - img[:, :-1]
        west[:, 1:] = img[:, :-1] - img[:, 1:]

        # Diffusion coefficients
        cN = np.exp(-(north / kappa) ** 2)
        cS = np.exp(-(south / kappa) ** 2)
        cE = np.exp(-(east / kappa) ** 2)
        cW = np.exp(-(west / kappa) ** 2)

        # Update image
        img += delta * (
            cN * north +
            cS * south +
            cE * east +
            cW * west
        )

    return np.clip(img, 0, 255).astype(np.uint8)


# =====================================================
# RASPBERRY PI USB WEBCAM
# =====================================================

print("==============================================")
print("   RASPBERRY PI USB WEBCAM - IMAGE DENOISING")
print("==============================================")

# Open USB webcam
cap = cv2.VideoCapture(0)

# Set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check camera
if not cap.isOpened():

    print("ERROR: USB Webcam could not be opened!")
    print("Check:")
    print("1. Webcam is connected")
    print("2. Camera index is correct")
    print("3. OpenCV is installed")
    exit()


print("USB Webcam opened successfully.")
print()
print("Controls:")
print("SPACE -> Capture image")
print("Q     -> Quit")
print("==============================================")


# =====================================================
# WAIT FOR MANUAL CAPTURE
# =====================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read frame from webcam.")
        break

    # Display live camera
    cv2.imshow(
        "Raspberry Pi Webcam - Press SPACE",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # -------------------------------------------------
    # SPACE = CAPTURE
    # -------------------------------------------------

    if key == 32:

        print()
        print("Image captured!")

        # Convert BGR to grayscale
        original = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        break

    # -------------------------------------------------
    # Q = QUIT
    # -------------------------------------------------

    elif key == ord('q'):

        print("Program cancelled.")

        cap.release()
        cv2.destroyAllWindows()

        exit()


# =====================================================
# CLOSE WEBCAM
# =====================================================

cap.release()
cv2.destroyAllWindows()


# =====================================================
# SAVE ORIGINAL IMAGE
# =====================================================

cv2.imwrite(
    "captured_image.jpg",
    original
)

print("Original image saved:")
print("captured_image.jpg")


# =====================================================
# APPLY ANISOTROPIC DIFFUSION
# =====================================================

print()
print("Applying Perona-Malik Anisotropic Diffusion...")

denoised = anisotropic_diffusion(
    original,
    iterations=25,
    delta=0.15,
    kappa=30
)


# =====================================================
# SAVE DENOISED IMAGE
# =====================================================

cv2.imwrite(
    "denoised_mri.jpg",
    denoised
)

print("Denoised image saved:")
print("denoised_mri.jpg")


# =====================================================
# DISPLAY RESULTS
# =====================================================

# Add labels
original_display = cv2.cvtColor(
    original,
    cv2.COLOR_GRAY2BGR
)

denoised_display = cv2.cvtColor(
    denoised,
    cv2.COLOR_GRAY2BGR
)

cv2.putText(
    original_display,
    "Captured Image",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.putText(
    denoised_display,
    "Anisotropic Diffusion",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)


# =====================================================
# RESIZE FOR DISPLAY
# =====================================================

display_original = cv2.resize(
    original_display,
    (640, 480)
)

display_denoised = cv2.resize(
    denoised_display,
    (640, 480)
)


# =====================================================
# SHOW RESULTS
# =====================================================

cv2.imshow(
    "Captured Image",
    display_original
)

cv2.imshow(
    "Denoised Image",
    display_denoised
)

print()
print("==============================================")
print("             PROCESS COMPLETED")
print("==============================================")
print("Press any key to close the images.")

cv2.waitKey(0)
cv2.destroyAllWindows()