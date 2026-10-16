
// ============================================
// FACEHELLO AI — COMPLETE SCRIPT.JS
// Face Detection + Recognition + Voice + Gallery
// ============================================

"use strict";

// --------------------------------------------
// 1. HTML ELEMENTS
// --------------------------------------------

const camera = document.getElementById("camera");
const placeholder = document.getElementById("placeholder");

const startButton = document.getElementById("startCamera");
const stopButton = document.getElementById("stopCamera");

const cameraStatus = document.getElementById("cameraStatus");
const cameraInfo = document.getElementById("cameraInfo");
const recognitionStatus = document.getElementById("recognitionStatus");
const personName = document.getElementById("personName");

const registerName = document.getElementById("registerName");
const registerButton = document.getElementById("registerFace");
const registrationStatus = document.getElementById("registrationStatus");

const photoGallery = document.getElementById("photoGallery");
const clearGalleryButton = document.getElementById("clearGallery");

const cameraFrame = document.querySelector(".camera-frame");

// --------------------------------------------
// 2. CONFIGURATION
// --------------------------------------------

// Official face-api.js model hosting path.
const MODEL_URL =
  "https://justadudewhohacks.github.io/face-api.js/models";

const PEOPLE_KEY = "faceHelloPeople";
const PHOTOS_KEY = "faceHelloPhotos";

const MAX_PHOTOS = 8;
const CAPTURE_INTERVAL = 5000;
const GREETING_COOLDOWN = 30000;

// Lower distance means a closer face match.
// Adjust cautiously if you get false matches.
const MATCH_THRESHOLD = 0.48;

// Tiny face detector settings.
// Lower input size / threshold can detect more faces,
// but may increase false detections or processing time.
const DETECTOR_OPTIONS = {
  inputSize: 320,
  scoreThreshold: 0.4
};

// --------------------------------------------
// 3. APP STATE
// --------------------------------------------

let people = [];
let photos = [];

let modelsLoaded = false;
let cameraStream = null;
let faceOverlay = null;

let detectionTimer = null;
let detectionBusy = false;
let isDetecting = false;

let latestDetections = [];
let lastCaptureTime = 0;

const lastGreetingTimes = new Map();

// --------------------------------------------
// 4. STORAGE HELPERS
// --------------------------------------------

function loadStoredData() {
  try {
    const savedPeople = JSON.parse(
      localStorage.getItem(PEOPLE_KEY) || "[]"
    );

    const savedPhotos = JSON.parse(
      localStorage.getItem(PHOTOS_KEY) || "[]"
    );

    people = Array.isArray(savedPeople)
      ? savedPeople.filter(
          (person) =>
            person &&
            typeof person.name === "string" &&
            Array.isArray(person.descriptor) &&
            person.descriptor.length === 128
        )
      : [];

    photos = Array.isArray(savedPhotos)
      ? savedPhotos.filter(
          (photo) =>
            photo &&
            typeof photo.src === "string" &&
            photo.src.startsWith("data:image/")
        ).slice(0, MAX_PHOTOS)
      : [];
  } catch (error) {
    console.error("Could not read saved data:", error);
    people = [];
    photos = [];
  }
}

function savePeople() {
  try {
    localStorage.setItem(PEOPLE_KEY, JSON.stringify(people));
    return true;
  } catch (error) {
    console.error("Could not save registered faces:", error);
    registrationStatus.textContent =
      "Storage is full or unavailable. Clear some browser data and try again.";
    return false;
  }
}

function savePhotos() {
  try {
    localStorage.setItem(PHOTOS_KEY, JSON.stringify(photos));
    return true;
  } catch (error) {
    console.error("Could not save photos:", error);

    // Remove older photos one at a time if storage is full.
    while (photos.length > 1) {
      photos.pop();

      try {
        localStorage.setItem(
          PHOTOS_KEY,
          JSON.stringify(photos)
        );

        renderGallery();

        recognitionStatus.textContent =
          "Photo saved. Older photos were removed to free storage.";

        return true;
      } catch {
        // Keep removing old photos until storage is available.
      }
    }

    recognitionStatus.textContent =
      "Browser storage is full. Clear the gallery and try again.";

    return false;
  }
}

// --------------------------------------------
// 5. LOAD FACE-API MODELS
// --------------------------------------------

async function loadModels() {
  if (modelsLoaded) {
    return true;
  }

  if (typeof faceapi === "undefined") {
    throw new Error(
      "face-api.js did not load. Check your internet connection and script tags."
    );
  }

  cameraStatus.textContent = "Loading AI models...";
  recognitionStatus.textContent =
    "Loading face detection and recognition models...";

  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL)
  ]);

  modelsLoaded = true;

  recognitionStatus.textContent =
    "AI models loaded. Start the camera to continue.";

  return true;
}

// --------------------------------------------
// 6. CAMERA OVERLAY
// --------------------------------------------

function createOverlay() {
  if (!cameraFrame) {
    throw new Error("Camera frame element was not found.");
  }

  if (faceOverlay) {
    faceOverlay.remove();
  }

  faceOverlay = document.createElement("canvas");
  faceOverlay.id = "faceOverlay";

  // Keep the canvas unmirrored; the drawing code mirrors
  // face boxes while keeping text readable.
  faceOverlay.style.transform = "none";

  cameraFrame.appendChild(faceOverlay);

  resizeOverlay();
}

function resizeOverlay() {
  if (!faceOverlay || !cameraFrame) {
    return;
  }

  const rect = cameraFrame.getBoundingClientRect();

  faceOverlay.width = Math.max(1, Math.round(rect.width));
  faceOverlay.height = Math.max(1, Math.round(rect.height));

  faceOverlay.style.width = "100%";
  faceOverlay.style.height = "100%";
}

function clearOverlay() {
  if (!faceOverlay) {
    return;
  }

  const context = faceOverlay.getContext("2d");

  if (context) {
    context.clearRect(
      0,
      0,
      faceOverlay.width,
      faceOverlay.height
    );
  }
}

// --------------------------------------------
// 7. START CAMERA
// --------------------------------------------

async function startCamera() {
  if (cameraStream) {
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    cameraStatus.textContent = "Camera unavailable";
    recognitionStatus.textContent =
      "Use HTTPS or localhost, and open this page in a supported browser.";
    return;
  }

  startButton.disabled = true;

  try {
    await loadModels();

    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    });

    camera.srcObject = cameraStream;
    camera.muted = true;
    camera.playsInline = true;

    await new Promise((resolve, reject) => {
      if (camera.readyState >= 2) {
        resolve();
        return;
      }

      camera.onloadedmetadata = () => resolve();
      camera.onerror = () => reject(
        new Error("Could not load the camera video.")
      );
    });

    await camera.play();

    createOverlay();

    placeholder.classList.add("hidden");
    placeholder.style.display = "none";

    cameraStatus.textContent = "Camera is running";
    if (cameraInfo) {
      cameraInfo.textContent = "Active";
    }

    recognitionStatus.textContent =
      "Looking for faces...";

    startButton.disabled = false;
    stopButton.disabled = false;

    isDetecting = true;
    runDetectionLoop();
  } catch (error) {
    console.error("Camera startup failed:", error);

    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
      cameraStream = null;
    }

    camera.srcObject = null;

    cameraStatus.textContent = "Camera could not start";
    if (cameraInfo) {
      cameraInfo.textContent = "Inactive";
    }

    recognitionStatus.textContent =
      error.name === "NotAllowedError"
        ? "Camera permission denied. Allow camera access in your browser."
        : error.name === "NotFoundError"
        ? "No camera was found on this device."
        : error.message ||
          "Unable to start camera. Check permissions and model loading.";

    startButton.disabled = false;
  }
}

// --------------------------------------------
// 8. FACE RECOGNITION
// --------------------------------------------

function euclideanDistance(descriptorA, descriptorB) {
  if (
    !descriptorA ||
    !descriptorB ||
    descriptorA.length !== descriptorB.length
  ) {
    return Infinity;
  }

  let sum = 0;

  for (let i = 0; i < descriptorA.length; i++) {
    const difference = descriptorA[i] - descriptorB[i];
    sum += difference * difference;
  }

  return Math.sqrt(sum);
}

function recognizeFace(descriptor) {
  let bestMatch = null;
  let smallestDistance = Infinity;

  for (const person of people) {
    const distance = euclideanDistance(
      descriptor,
      person.descriptor
    );

    if (distance < smallestDistance) {
      smallestDistance = distance;
      bestMatch = person;
    }
  }

  if (
    bestMatch &&
    smallestDistance <= MATCH_THRESHOLD
  ) {
    return {
      name: bestMatch.name,
      distance: smallestDistance,
      matched: true
    };
  }

  return {
    name: "Unknown person",
    distance: smallestDistance,
    matched: false
  };
}

// --------------------------------------------
// 9. DRAW FACE BOXES AND LABELS
// --------------------------------------------

function drawFaceBoxes(detections) {
  if (!faceOverlay) {
    return;
  }

  const context = faceOverlay.getContext("2d");

  if (!context) {
    return;
  }

  const width = faceOverlay.width;
  const height = faceOverlay.height;

  context.clearRect(0, 0, width, height);

  if (!detections.length) {
    return;
  }

  const resized = faceapi.resizeResults(detections, {
    width,
    height
  });

  resized.forEach((detection, index) => {
    const box = detection.detection.box;
    const result = recognizeFace(
      detections[index].descriptor
    );

    // Video is mirrored, so mirror the box horizontally.
    const x = width - box.x - box.width;
    const y = box.y;

    context.lineWidth = 2.5;
    context.strokeStyle = result.matched
      ? "#56f0e5"
      : "#ffbd69";

    context.shadowColor = context.strokeStyle;
    context.shadowBlur = 8;

    context.strokeRect(
      x,
      y,
      box.width,
      box.height
    );

    context.shadowBlur = 0;

    const label = result.matched
      ? result.name
      : "Unknown";

    context.font = "bold 14px Arial";

    const textWidth = context.measureText(label).width;
    const labelWidth = Math.min(
      width - 4,
      textWidth + 16
    );

    const labelHeight = 25;
    const labelX = Math.max(
      2,
      Math.min(x, width - labelWidth - 2)
    );

    const labelY = Math.max(
      0,
      y - labelHeight
    );

    context.fillStyle = result.matched
      ? "#56f0e5"
      : "#ffbd69";

    context.fillRect(
      labelX,
      labelY,
      labelWidth,
      labelHeight
    );

    context.fillStyle = "#071018";

    context.fillText(
      label,
      labelX + 8,
      labelY + 17,
      labelWidth - 12
    );
  });
}

// --------------------------------------------
// 10. DETECTION LOOP
// --------------------------------------------

async function runDetectionLoop() {
  if (!isDetecting || detectionBusy) {
    return;
  }

  if (
    !cameraStream ||
    camera.readyState < 2 ||
    !modelsLoaded
  ) {
    detectionTimer = setTimeout(runDetectionLoop, 300);
    return;
  }

  detectionBusy = true;

  try {
    const detections = await faceapi
      .detectAllFaces(
        camera,
        new faceapi.TinyFaceDetectorOptions(
          DETECTOR_OPTIONS
        )
      )
      .withFaceLandmarks()
      .withFaceDescriptors();

    if (!isDetecting) {
      return;
    }

    latestDetections = detections;

    resizeOverlay();
    drawFaceBoxes(detections);

    handleRecognition(detections);
  } catch (error) {
    console.error("Face detection error:", error);

    recognitionStatus.textContent =
      "Face detection error. Check that the models loaded correctly.";
  } finally {
    detectionBusy = false;

    if (isDetecting) {
      // Delay slightly to keep processing manageable.
      detectionTimer = setTimeout(runDetectionLoop, 120);
    }
  }
}

// --------------------------------------------
// 11. UPDATE RECOGNITION STATUS
// --------------------------------------------

function handleRecognition(detections) {
  if (!detections.length) {
    personName.textContent = "No face recognized";
    recognitionStatus.textContent =
      "No face detected. Look at the camera.";

    return;
  }

  if (detections.length > 1) {
    personName.textContent = "Multiple faces detected";
    recognitionStatus.textContent =
      "For accurate registration, keep only one face visible.";

    return;
  }

  const result = recognizeFace(
    detections[0].descriptor
  );

  if (result.matched) {
    personName.textContent = result.name;

    recognitionStatus.textContent =
      "Registered face recognized. Greeting enabled.";

    greetPerson(result.name);
    capturePhoto(result.name);
  } else {
    personName.textContent = "Unknown person";

    recognitionStatus.textContent =
      people.length
        ? "This face is not registered. Enter a name and click Register Face."
        : "No registered faces yet. Register a face to enable recognition.";
  }
}

// --------------------------------------------
// 12. VOICE GREETING
// --------------------------------------------

function greetPerson(name) {
  if (!("speechSynthesis" in window)) {
    return;
  }

  const now = Date.now();
  const lastTime = lastGreetingTimes.get(name) || 0;

  if (now - lastTime < GREETING_COOLDOWN) {
    return;
  }

  lastGreetingTimes.set(name, now);

  const message = new SpeechSynthesisUtterance(
    `Hello ${name}, welcome to FaceHello AI!`
  );

  message.lang = "en-US";
  message.rate = 1;
  message.pitch = 1;
  message.volume = 1;

  // Avoid a growing queue of greetings.
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(message);
}

// --------------------------------------------
// 13. REGISTER CURRENT FACE
// --------------------------------------------

function registerCurrentFace() {
  const name = registerName.value.trim();

  if (!name) {
    registrationStatus.textContent =
      "Please enter a name first.";
    registerName.focus();
    return;
  }

  if (!cameraStream || !isDetecting) {
    registrationStatus.textContent =
      "Start the camera before registering a face.";
    return;
  }

  if (latestDetections.length !== 1) {
    registrationStatus.textContent =
      latestDetections.length === 0
        ? "No face detected. Look at the camera and try again."
        : "Multiple faces detected. Only one person should be in the frame.";
    return;
  }

  const descriptor = Array.from(
    latestDetections[0].descriptor
  );

  const existingIndex = people.findIndex(
    (person) =>
      person.name.toLowerCase() === name.toLowerCase()
  );

  const newPerson = {
    name,
    descriptor,
    registeredAt: new Date().toISOString()
  };

  if (existingIndex >= 0) {
    const shouldReplace = window.confirm(
      `${name} is already registered. Replace their saved face?`
    );

    if (!shouldReplace) {
      return;
    }

    people[existingIndex] = newPerson;
  } else {
    people.push(newPerson);
  }

  if (!savePeople()) {
    return;
  }

  registrationStatus.textContent =
    `Successfully registered ${name}!`;

  personName.textContent = name;
  recognitionStatus.textContent =
    "Face registered successfully.";

  registerName.value = "";
}

// --------------------------------------------
// 14. AUTOMATIC PHOTO CAPTURE
// --------------------------------------------

function capturePhoto(name) {
  const now = Date.now();

  if (now - lastCaptureTime < CAPTURE_INTERVAL) {
    return;
  }

  if (
    !cameraStream ||
    camera.readyState < 2 ||
    camera.videoWidth === 0
  ) {
    return;
  }

  lastCaptureTime = now;

  const canvas = document.createElement("canvas");

  // Resize the image to reduce local storage usage.
  const maxWidth = 640;
  const scale = Math.min(
    1,
    maxWidth / camera.videoWidth
  );

  canvas.width = Math.round(camera.videoWidth * scale);
  canvas.height = Math.round(camera.videoHeight * scale);

  const context = canvas.getContext("2d");

  if (!context) {
    return;
  }

  // Save a mirrored selfie to match the on-screen preview.
  context.translate(canvas.width, 0);
  context.scale(-1, 1);

  context.drawImage(
    camera,
    0,
    0,
    canvas.width,
    canvas.height
  );

  let imageData;

  try {
    imageData = canvas.toDataURL("image/jpeg", 0.72);
  } catch (error) {
    console.error("Could not capture photo:", error);
    return;
  }

  const photo = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    name,
    src: imageData,
    capturedAt: new Date().toISOString()
  };

  photos.unshift(photo);
  photos = photos.slice(0, MAX_PHOTOS);

  if (savePhotos()) {
    renderGallery();
  }
}

// --------------------------------------------
// 15. RENDER PHOTO GALLERY
// --------------------------------------------

function renderGallery() {
  if (!photoGallery) {
    return;
  }

  photoGallery.replaceChildren();

  if (!photos.length) {
    const emptyMessage = document.createElement("p");

    emptyMessage.className = "gallery-empty";
    emptyMessage.textContent =
      "Your captured photos will appear here.";

    photoGallery.appendChild(emptyMessage);
    return;
  }

  photos.forEach((photo) => {
    const card = document.createElement("article");
    card.className = "photo-card";

    const image = document.createElement("img");
    image.src = photo.src;
    image.alt = `Photo of ${photo.name}`;
    image.loading = "lazy";

    const name = document.createElement("p");
    name.textContent = photo.name;

    const date = document.createElement("small");

    const capturedDate = new Date(photo.capturedAt);

    date.textContent = Number.isNaN(capturedDate.getTime())
      ? "Captured photo"
      : capturedDate.toLocaleString();

    card.appendChild(image);
    card.appendChild(name);
    card.appendChild(date);

    photoGallery.appendChild(card);
  });
}

// --------------------------------------------
// 16. CLEAR GALLERY
// --------------------------------------------

function clearGallery() {
  if (!photos.length) {
    return;
  }

  const confirmed = window.confirm(
    "Delete all captured photos? This cannot be undone."
  );

  if (!confirmed) {
    return;
  }

  const previousPhotos = photos;
  photos = [];

  try {
    localStorage.setItem(PHOTOS_KEY, JSON.stringify(photos));
    renderGallery();

    recognitionStatus.textContent =
      "Captured photo gallery cleared.";
  } catch (error) {
    photos = previousPhotos;

    console.error("Could not clear gallery:", error);

    recognitionStatus.textContent =
      "Unable to clear the gallery.";
  }
}

// --------------------------------------------
// 17. STOP CAMERA
// --------------------------------------------

function stopCamera() {
  isDetecting = false;

  if (detectionTimer) {
    clearTimeout(detectionTimer);
    detectionTimer = null;
  }

  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }

  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => {
      track.stop();
    });

    cameraStream = null;
  }

  camera.pause();
  camera.srcObject = null;

  latestDetections = [];
  detectionBusy = false;

  clearOverlay();

  if (faceOverlay) {
    faceOverlay.remove();
    faceOverlay = null;
  }

  placeholder.classList.remove("hidden");
  placeholder.style.display = "flex";

  cameraStatus.textContent = "Camera is off";

  if (cameraInfo) {
    cameraInfo.textContent = "Inactive";
  }

  personName.textContent = "No face recognized";

  recognitionStatus.textContent =
    "Camera stopped. Start it again to recognize faces.";

  startButton.disabled = false;
  stopButton.disabled = false;
}

// --------------------------------------------
// 18. BUTTON EVENT LISTENERS
// --------------------------------------------

startButton.addEventListener("click", startCamera);

stopButton.addEventListener("click", stopCamera);

registerButton.addEventListener(
  "click",
  registerCurrentFace
);

clearGalleryButton.addEventListener(
  "click",
  clearGallery
);

// Allow Enter to register a face.
registerName.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    registerCurrentFace();
  }
});

// Resize the overlay when the window changes size.
window.addEventListener("resize", resizeOverlay);

// Stop the camera when the page is being unloaded.
window.addEventListener("pagehide", () => {
  isDetecting = false;

  if (detectionTimer) {
    clearTimeout(detectionTimer);
  }

  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }

  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
});

// --------------------------------------------
// 19. INITIALIZE APP
// --------------------------------------------

loadStoredData();
renderGallery();

if (stopButton) {
  stopButton.disabled = false;
}

if (cameraInfo) {
  cameraInfo.textContent = "Inactive";
}

registrationStatus.textContent = people.length
  ? `${people.length} registered face(s) loaded from this browser.`
  : "Register a face to enable recognition.";

console.log("FaceHello AI initialized.");
console.log(`Registered faces: ${people.length}`);
console.log(`Saved photos: ${photos.length}`);
