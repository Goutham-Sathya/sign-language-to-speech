"""
tts_engine.py

This module handles Text-To-Speech (TTS) functionality.

It converts text messages into spoken audio using the
pyttsx3 library. The speech is played through the
system speakers.

"""

import pyttsx3


class TTSEngine:
    """
    TTSEngine is responsible for converting text
    into speech output.
    """

    def __init__(self, rate=150, volume=1.0):
        """
        Initialize the TTS engine.

        volume : float
            Volume level (0.0 to 1.0)
        """

        # Initialize the speech engine
        self.engine = pyttsx3.init()

        # Set speaking speed
        self.engine.setProperty("rate", rate)

        # Set volume level
        self.engine.setProperty("volume", volume)

    def speak(self, text):
        """
        Convert text into speech.

        """

        # Queue the text to be spoken
        self.engine.say(text)

        # Run the speech engine
        self.engine.runAndWait()