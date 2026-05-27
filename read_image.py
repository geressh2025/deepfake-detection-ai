import cv2

image = cv2.imread("images/00005.jpg")

if image is None:
    print("Error: Image not found")
    exit()

# Resize image
image = cv2.resize(image, (600, 600))

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Print number of faces found
print(f"Faces detected: {len(faces)}")

# Loop through detected faces
for i, (x, y, w, h) in enumerate(faces):

    # Draw rectangle around face
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # Crop face
    face_crop = image[y:y+h, x:x+w]

    # Save cropped face
    cv2.imwrite(f"images/cropped_face_{i}.jpg", face_crop)

# Show output image
cv2.imshow("Face Detection", image)

# Wait until key press
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()