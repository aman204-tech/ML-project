import cv2
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

class VideoPreprocessor:
    def __init__(self, 
                 face_cascade_path: str = 'haarcascade_frontalface_default.xml',
                 mouth_cascade_path: str = 'haarcascade_mcs_mouth.xml',
                 target_frames: int = 29,
                 target_size: Tuple[int, int] = (128, 128)):
        """Initialize video preprocessor with cascade classifiers."""
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
        self.target_frames = target_frames
        self.target_size = target_size

    def detect_mouth(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Detect mouth region in image.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (processed image, mouth coordinates or None)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        if len(faces) == 0:
            print("No faces detected.")
            return image, None

        (x, y, w, h) = faces[0]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        mouth_roi_y_start = int(y + 0.6 * h)
        mouth_roi = gray[mouth_roi_y_start: y + h, x: x + w]

        mouths = self.mouth_cascade.detectMultiScale(mouth_roi, scaleFactor=1.1, minNeighbors=3)

        if len(mouths) == 0:
            print("No mouth detected.")
            return image, None

        best_candidate = None
        best_y = -1
        for (mx, my, mw, mh) in mouths:
            absolute_y = mouth_roi_y_start + my
            if absolute_y > best_y:
                best_y = absolute_y
                best_candidate = (x + mx, mouth_roi_y_start + my, mw, mh)

        return image, best_candidate

    def extract_mouth_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract and resize mouth region from image."""
        _, mouth_coords = self.detect_mouth(image)
        
        if mouth_coords is None:
            return None
            
        x, y, w, h = mouth_coords
        mouth_region = image[y:y+h, x:x+w]
        mouth_region = cv2.resize(mouth_region, self.target_size)
        
        return mouth_region

    def process_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Process video file and extract mouth regions.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Array of shape (num_frames, height, width, channels) or None
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {video_path}")
            
        frames = []
        while len(frames) < self.target_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            mouth_region = self.extract_mouth_region(frame)
            if mouth_region is not None:
                frames.append(mouth_region)
                
        cap.release()
        
        if len(frames) == 0:
            return None
            
        # Pad or trim to target length
        if len(frames) < self.target_frames:
            last_frame = frames[-1]
            while len(frames) < self.target_frames:
                frames.append(last_frame)
        elif len(frames) > self.target_frames:
            indices = np.linspace(0, len(frames)-1, self.target_frames, dtype=int)
            frames = [frames[i] for i in indices]
            
        return np.stack(frames)

    def process_dataset(self, input_dir: str, output_dir: str):
        """
        Process entire dataset directory.
        
        Args:
            input_dir: Path to raw dataset
            output_dir: Path to save processed data
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Gather both .mpg and .mp4 files
        video_files = list(input_path.rglob('*.mpg')) + list(input_path.rglob('*.mp4'))
        
        for video_path in video_files:
            # Create output directory structure
            relative_path = video_path.relative_to(input_path)
            output_video_dir = output_path / relative_path.parent / video_path.stem
            output_video_dir.mkdir(parents=True, exist_ok=True)

            # Process video
            try:
                frames = self.process_video(str(video_path))
                if frames is not None:
                    # Save individual frames
                    for i, frame in enumerate(frames):
                        frame_path = output_video_dir / f"frame_{i:03d}.jpg"
                        cv2.imwrite(str(frame_path), frame)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")