# Import necessary libraries
import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import interp1d
import time
import os
import tempfile

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results.pose_landmarks if results.pose_landmarks else None

class DanceGenerator:
    def __init__(self):
        self.prev_moves = []
        self.style_memory = []
        self.rhythm_patterns = []

    def generate_dance_sequence(self, all_poses, mode, total_frames, frame_size):
        height, width = frame_size
        sequence = []

        if mode == "Sync Partner":
            sequence = self._generate_sync_sequence(all_poses, total_frames, frame_size)
        else:
            sequence = self._generate_creative_sequence(all_poses, total_frames, frame_size)

        return sequence

    def _generate_sync_sequence(self, all_poses, total_frames, frame_size):
        height, width = frame_size
        sequence = []
        
        # Enhanced rhythm analysis
        rhythm_window = 10  # Analyze chunks of frames for rhythm
        beat_positions = self._detect_dance_beats(all_poses, rhythm_window)
        
        pose_arrays = []
        for pose in all_poses:
            if pose is not None:
                pose_arrays.append(self._landmarks_to_array(pose))
            else:
                pose_arrays.append(None)
                
        for i in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            if pose_arrays[i] is not None:
                # Enhanced mirroring with rhythm awareness
                mirrored = self._mirror_movements(pose_arrays[i])
                
                # Apply rhythm-based movement enhancement
                if i in beat_positions:
                    mirrored = self._enhance_movement_on_beat(mirrored)
                
                if i > 0 and pose_arrays[i-1] is not None:
                    mirrored = self._smooth_transition(pose_arrays[i-1], mirrored, 0.3)
                
                frame = self._create_enhanced_dance_frame(
                    mirrored,
                    frame_size,
                    add_effects=True
                )
            
            sequence.append(frame)
        
        return sequence
        
    def _detect_dance_beats(self, poses, window_size):
        """Detect main beats in the dance sequence"""
        beat_positions = []
        
        if len(poses) < window_size:
            return beat_positions
            
        for i in range(window_size, len(poses)):
            if poses[i] is not None and poses[i-1] is not None:
                curr_pose = self._landmarks_to_array(poses[i])
                prev_pose = self._landmarks_to_array(poses[i-1])
                
                # Calculate movement magnitude
                movement = np.mean(np.abs(curr_pose - prev_pose))
                
                # Detect significant movements as beats
                if movement > np.mean(self.rhythm_patterns) + np.std(self.rhythm_patterns):
                    beat_positions.append(i)
                    
        return beat_positions
        
    def _enhance_movement_on_beat(self, pose):
        """Enhance movements during detected beats"""
        # Amplify movements slightly on beats
        center = np.mean(pose, axis=0)
        enhanced_pose = pose.copy()
        
        for i in range(len(pose)):
            # Amplify movement relative to center
            vector = pose[i] - center
            enhanced_pose[i] = center + vector * 1.2
            
        return enhanced_pose

    def _generate_creative_sequence(self, all_poses, total_frames, frame_size):
        """Generate creative dance sequence based on style"""
        height, width = frame_size
        sequence = []

        # Analyze style from all poses
        style_patterns = self._analyze_style_patterns(all_poses)

        # Generate new sequence using style patterns
        for i in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Generate new pose based on style
            new_pose = self._generate_style_based_pose(style_patterns, i/total_frames)

            if new_pose is not None:
                frame = self._create_enhanced_dance_frame(
                    new_pose,
                    frame_size,
                    add_effects=True
                )

            sequence.append(frame)

        return sequence

    def _analyze_style_patterns(self, poses):
        """Enhanced style analysis including rhythm and movement patterns"""
        patterns = []
        rhythm_data = []
        
        for i in range(1, len(poses)):
            if poses[i] is not None and poses[i-1] is not None:
                # Calculate movement speed and direction
                curr_pose = self._landmarks_to_array(poses[i])
                prev_pose = self._landmarks_to_array(poses[i-1])
                
                # Analyze movement velocity
                velocity = np.mean(np.abs(curr_pose - prev_pose), axis=0)
                rhythm_data.append(velocity)
                
                # Store enhanced pattern data
                pattern_info = {
                    'pose': curr_pose,
                    'velocity': velocity,
                    'acceleration': velocity if i == 1 else velocity - prev_velocity
                }
                patterns.append(pattern_info)
                prev_velocity = velocity
                
        self.rhythm_patterns = rhythm_data
        return patterns

    def _generate_style_based_pose(self, patterns, progress):
        """Generate new pose based on style patterns and progress"""
        if not patterns:
            return None

        # Create smooth interpolation between poses
        num_patterns = len(patterns)
        pattern_idx = int(progress * (num_patterns - 1))

        if pattern_idx < num_patterns - 1:
            t = progress * (num_patterns - 1) - pattern_idx
            # Extract pose arrays from pattern dictionaries
            pose1 = patterns[pattern_idx]['pose']
            pose2 = patterns[pattern_idx + 1]['pose']
            pose = self._interpolate_poses(pose1, pose2, t)
        else:
            pose = patterns[-1]['pose']

        return pose

    def _interpolate_poses(self, pose1, pose2, t):
        """Smoothly interpolate between two poses"""
        if isinstance(pose1, dict):
            pose1 = pose1['pose']
        if isinstance(pose2, dict):
            pose2 = pose2['pose']
        return pose1 * (1 - t) + pose2 * t

    def _create_enhanced_dance_frame(self, pose_array, frame_size, add_effects=True):
        """Create enhanced visualization frame with effects"""
        height, width = frame_size
        # Create transparent background
        frame = np.zeros((height, width, 3), dtype=np.uint8)  # Black background
        
        # Convert coordinates
        points = (pose_array[:, :2] * [width, height]).astype(int)
        
        # Draw enhanced skeleton with neon effects
        connections = self._get_pose_connections()
        
        # Define body parts and their colors
        body_parts = {
            'spine': [(11, 23), (23, 24), (11, 12)],  # Torso
            'right_arm': [(11, 13), (13, 15)],  # Right arm
            'left_arm': [(12, 14), (14, 16)],   # Left arm
            'right_leg': [(23, 25), (25, 27), (27, 29), (29, 31)],  # Right leg
            'left_leg': [(24, 26), (26, 28), (28, 30), (30, 32)],   # Left leg
            'face': [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8)]  # Face
        }
        
        colors = {
            'spine': (0, 255, 255),    # Cyan
            'right_arm': (0, 255, 0),  # Green
            'left_arm': (0, 255, 0),   # Green
            'right_leg': (255, 0, 255), # Magenta
            'left_leg': (255, 0, 255),  # Magenta
            'face': (255, 255, 0)      # Yellow
        }
        
        # Draw body parts with glow effects
        for part, connections_list in body_parts.items():
            color = colors[part]
            for connection in connections_list:
                start_idx, end_idx = connection
                if start_idx < len(points) and end_idx < len(points):
                    if add_effects:
                        self._draw_glowing_line(
                            frame,
                            points[start_idx],
                            points[end_idx],
                            color,
                            thickness=3
                        )
                    else:
                        cv2.line(frame,
                                tuple(points[start_idx]),
                                tuple(points[end_idx]),
                                color, 3)

        # Draw enhanced joints with glow
        for i, point in enumerate(points):
            if add_effects:
                # Different colors for different body parts
                if i in [0,1,2,3,4,5,6,7,8]:  # Face points
                    color = (255, 255, 0)  # Yellow
                elif i in [11,12,23,24]:  # Torso points
                    color = (0, 255, 255)  # Cyan
                elif i in [13,14,15,16]:  # Arms points
                    color = (0, 255, 0)    # Green
                else:  # Legs points
                    color = (255, 0, 255)  # Magenta
                    
                self._draw_glowing_point(frame, point, color, radius=4)
            else:
                cv2.circle(frame, tuple(point), 4, (255, 255, 255), -1)

        return frame

    def _draw_glowing_line(self, frame, start, end, color, thickness=3):
        """Draw a line with enhanced neon glow effect"""
        # Draw outer glow
        for i in range(3):
            alpha = 0.3 - i * 0.1
            thick = thickness + (i * 2)
            blur_color = tuple([int(c * alpha) for c in color])
            cv2.line(frame, tuple(start), tuple(end),
                    blur_color, thick)
        
        # Draw main line
        cv2.line(frame, tuple(start), tuple(end), color, thickness)

    def _draw_glowing_point(self, frame, point, color, radius=4):
        """Draw a point with enhanced neon glow effect"""
        # Draw outer glow
        for i in range(3):
            alpha = 0.3 - i * 0.1
            r = radius + (i * 2)
            blur_color = tuple([int(c * alpha) for c in color])
            cv2.circle(frame, tuple(point), r,
                      blur_color, -1)
        
        # Draw main point
        cv2.circle(frame, tuple(point), radius, color, -1)

    def _landmarks_to_array(self, landmarks):
        """Convert MediaPipe landmarks to numpy array"""
        points = []
        for landmark in landmarks.landmark:
            points.append([landmark.x, landmark.y, landmark.z])
        return np.array(points)

    def _mirror_movements(self, landmarks):
        """Mirror the input movements"""
        mirrored = landmarks.copy()
        mirrored[:, 0] = 1 - mirrored[:, 0]  # Flip x coordinates
        return mirrored

    def _update_style_memory(self, landmarks):
        """Update memory of dance style"""
        self.style_memory.append(landmarks)
        if len(self.style_memory) > 30:  # Keep last 30 frames
            self.style_memory.pop(0)

    def _generate_style_based_moves(self):
        """Generate new moves based on learned style"""
        if not self.style_memory:
            return np.zeros((33, 3))  # Default pose shape

        # Simple implementation: interpolate between stored poses
        base_pose = self.style_memory[-1]
        if len(self.style_memory) > 1:
            prev_pose = self.style_memory[-2]
            t = np.random.random()
            new_pose = t * base_pose + (1-t) * prev_pose
        else:
            new_pose = base_pose

        return new_pose

    def _create_dance_frame(self, pose_array):
        """Create visualization frame from pose array"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Convert normalized coordinates to pixel coordinates
        points = (pose_array[:, :2] * [640, 480]).astype(int)

        # Draw connections between joints
        connections = self._get_pose_connections()
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame,
                        tuple(points[start_idx]),
                        tuple(points[end_idx]),
                        (0, 255, 0), 2)

        # Draw joints
        for point in points:
            cv2.circle(frame, tuple(point), 4, (0, 0, 255), -1)

        return frame

    def _get_pose_connections(self):
        """Define connections between pose landmarks"""
        return [
            (0, 1), (1, 2), (2, 3), (3, 7),  # Face
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15),  # Arms
            (12, 14), (14, 16),
            (11, 23), (12, 24),  # Torso
            (23, 24), (23, 25), (24, 26),  # Legs
            (25, 27), (26, 28), (27, 29), (28, 30),
            (29, 31), (30, 32)
        ]

    def _smooth_transition(self, prev_pose, current_pose, smoothing_factor=0.3):
        """Create smooth transition between poses"""
        if prev_pose is None or current_pose is None:
            return current_pose

        # Interpolate between previous and current pose
        smoothed_pose = (1 - smoothing_factor) * prev_pose + smoothing_factor * current_pose

        # Ensure the smoothed pose maintains proper proportions
        # Normalize joint positions relative to hip center
        hip_center_idx = 23  # Index for hip center landmark

        prev_hip = prev_pose[hip_center_idx]
        current_hip = current_pose[hip_center_idx]
        smoothed_hip = smoothed_pose[hip_center_idx]

        # Adjust positions relative to hip center
        for i in range(len(smoothed_pose)):
            if i != hip_center_idx:
                # Calculate relative positions
                prev_relative = prev_pose[i] - prev_hip
                current_relative = current_pose[i] - current_hip

                # Interpolate relative positions
                smoothed_relative = (1 - smoothing_factor) * prev_relative + smoothing_factor * current_relative

                # Update smoothed pose
                smoothed_pose[i] = smoothed_hip + smoothed_relative

        return smoothed_pose

class AIDancePartner:
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.dance_generator = DanceGenerator()

    def process_video(self, video_path, mode="Sync Partner"):
        # Create a temporary directory for output
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, 'output_dance.mp4')
        
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps,
                            (frame_width * 2, frame_height))

        # Pre-process video to extract all poses
        all_poses = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pose_landmarks = self.pose_detector.detect_pose(frame)
            all_poses.append(pose_landmarks)
            frame_count += 1

        # Generate AI dance sequence
        ai_sequence = self.dance_generator.generate_dance_sequence(
            all_poses,
            mode,
            total_frames,
            (frame_height, frame_width)
        )

        # Reset video capture and create final video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get corresponding AI frame
            ai_frame = ai_sequence[frame_count]

            # Combine frames side by side
            combined_frame = np.hstack([frame, ai_frame])

            # Write frame to output video
            out.write(combined_frame)
            frame_count += 1

        # Release resources
        cap.release()
        out.release()

        return output_path
