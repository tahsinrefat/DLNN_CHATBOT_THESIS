import mutagen
from mutagen.mp3 import MP3
audio = MP3("output.mp3")
print(audio.info.length)