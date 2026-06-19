"""
text_to_speech.py
=================
Phase 5 — Text-to-Speech Module

Converts Sinhala text predictions into audio using Google Text-to-Speech (gTTS).
Uses Pygame for non-blocking asynchronous playback so that the video 
feed does not freeze while the audio is speaking.

Usage:
    from src.live.text_to_speech import SinhalaTTS
    tts = SinhalaTTS()
    tts.speak("ආයුබෝවන්")
"""

import logging
import os
import tempfile
import threading
from pathlib import Path

from gtts import gTTS

# Suppress Pygame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame

logger = logging.getLogger(__name__)


class SinhalaTTS:
    """
    Sinhala Text-to-Speech Synthesizer with background playback.
    """

    def __init__(self):
        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.init()
            self._mixer_initialized = True
            logger.info("[OK] TTS Engine (Pygame Mixer) initialized")
        except Exception as e:
            logger.error(f"Failed to initialize audio mixer: {e}")
            self._mixer_initialized = False

        # Keep track of the currently playing temp file
        self._current_temp_file = None

    def speak(self, text: str) -> None:
        """
        Speak the given Sinhala text. Does not block the main thread.
        
        Args:
            text: The Sinhala word or phrase to speak.
        """
        if not text or text.strip() == "":
            return
            
        if not self._mixer_initialized:
            logger.warning("Audio mixer is not initialized. Skipping TTS.")
            return

        # If audio is already playing, let it finish or stop it.
        # We stop the current one so the new word can be spoken immediately.
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            self._cleanup_temp_file()

        # Run TTS API call and playback in a background thread to avoid blocking video stream
        threading.Thread(target=self._synthesize_and_play, args=(text,), daemon=True).start()

    def _synthesize_and_play(self, text: str) -> None:
        """Internal method to generate mp3 and play it."""
        try:
            # 1. Generate audio using gTTS (language='si' for Sinhala)
            tts = gTTS(text=text, lang="si", slow=False)
            
            # 2. Save to a temporary file
            # NamedTemporaryFile requires delete=False on Windows for Pygame to load it
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_path = temp_file.name
            temp_file.close()
            
            tts.save(temp_path)
            self._current_temp_file = temp_path
            
            # 3. Play audio
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
            # Note: We do not delete the file immediately because pygame needs it 
            # while it plays. The _cleanup_temp_file() method will handle cleanup
            # on the next speak() call or on application exit.
            
        except Exception as e:
            logger.error(f"TTS Synthesis error for text '{text}': {e}")

    def _cleanup_temp_file(self) -> None:
        """Deletes the previous temporary audio file to save disk space."""
        if self._current_temp_file and os.path.exists(self._current_temp_file):
            try:
                os.remove(self._current_temp_file)
                self._current_temp_file = None
            except Exception:
                pass  # File might still be locked by pygame

    def close(self) -> None:
        """Cleanup resources on exit."""
        if self._mixer_initialized:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        self._cleanup_temp_file()


if __name__ == "__main__":
    # Quick test
    import time
    
    logging.basicConfig(level=logging.INFO)
    tts = SinhalaTTS()
    
    logger.info("Testing Sinhala TTS: ආයුබෝවන් (Ayubowan)")
    tts.speak("ආයුබෝවන්")
    
    # Wait for audio to finish
    time.sleep(3)
    
    logger.info("Testing Sinhala TTS: ස්තුතියි (Thank you)")
    tts.speak("ස්තුතියි")
    
    time.sleep(3)
    tts.close()
