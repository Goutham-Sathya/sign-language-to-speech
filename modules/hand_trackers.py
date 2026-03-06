"""
This module handles hand detection using MediaPipe
what it does :
1. Capture hand landmarks from a frame
2. Draw hand skeleton on the frame
3. Return the detected landmarks for gesture logic
"""

import cv2
import mediapipe as mp


class HandTracker:

    def __init__(self,
                 max_hands=2,
                 detection_confidence=0.7,
                 tracking_confidence=0.7):

        # MediaPipe Hands initialization
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        self.mp_draw = mp.solutions.drawing_utils


    def find_hands(self, frame):
        """
        Detect hands and draw landmarks.
        """

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        return frame, results


    def get_landmarks(self, results):
        """
        Returns landmarks for each detected hand.
        """

        all_hands = []

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                hand_points = []

                for lm in hand_landmarks.landmark:
                    hand_points.append((lm.x, lm.y))

                all_hands.append(hand_points)

        return all_hands