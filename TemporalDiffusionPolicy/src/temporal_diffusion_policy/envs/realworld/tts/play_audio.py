import pyaudio
import wave
import threading
import os
from queue import Queue

# https://ttsmp3.com
# AI Voice
# Shimmer (female)
# download mp3, then use cloudconvert to wav
class AudioPlayer:
    def __init__(self):
        self.queue = Queue()
        self.thread = threading.Thread(target=self._play_queue, daemon=True)
        self.thread.start()
    
    def _play_audio(self, filename):
        """Play a single audio file"""
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        if not os.path.exists(filepath):
            print(f"Error: File not found - {filepath}")
            return
        
        try:
            with wave.open(filepath, 'rb') as wf:
                p = pyaudio.PyAudio()
                
                stream = p.open(
                    format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )
                
                chunk = 1024
                data = wf.readframes(chunk)
                while data:
                    stream.write(data)
                    data = wf.readframes(chunk)
                
                stream.stop_stream()
                stream.close()
                p.terminate()
        except Exception as e:
            print(f"Error playing {filename}: {e}")
    
    def _play_queue(self):
        """Background thread: continuously play from queue"""
        while True:
            filename = self.queue.get()
            self._play_audio(filename)
            self.queue.task_done()
    
    def queue_sound(self, filename):
        """Add sound to playback queue (non-blocking)"""
        self.queue.put(filename)
    


# Usage
if __name__ == "__main__":
    # Global player instance
    player = AudioPlayer()
    # Queue sounds - they'll play one after the other on background thread
    player.queue_sound("tts_go.wav")
    player.queue_sound("tts_lift.wav")
    player.queue_sound("tts_return.wav")
    
    print("Sounds queued! Main thread continues...")
    
    
    input()
    # Wait for all sounds to finish
    print("All sounds finished!")
