import os
import speech_recognition as sr
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment
import time

class AudioTranscriber:

    def __init__(self):
        self.recognizer = sr.Recognizer()

    def convert_mp3_to_wav(self, mp3_path):
        audio = AudioSegment.from_mp3(mp3_path)
        wav_path = mp3_path.replace(".mp3", ".wav")
        audio.export(wav_path, format="wav")
        return wav_path

    def get_transcription_from_source(self):
        pass

    def convert_video_to_wav(video_path, wav_path):
        video_clip = VideoFileClip(video_path)
        audio_temp_path = "temp_audio.wav"
        video_clip.audio.write_audiofile(audio_temp_path, codec="pcm_s16le")
        os.rename(audio_temp_path, wav_path)
        return wav_path

    def transcribe_audio(self, audio_path):
        if audio_path.endswith(".mp3"):
            audio_path = self.convert_mp3_to_wav(audio_path)

        for intento in range(5):
            try:
                with sr.AudioFile(audio_path) as source:
                    audio_data = self.recognizer.record(source)
                transcription = self.recognizer.recognize_google(audio_data, language='en')
                return transcription
            except Exception as e:
                print(f"Intento {intento + 1}: Error al transcribir el audio: {e}. Repitiendo en 1 segundo...")
                time.sleep(1)  # Espera 1 segundo antes de reintentar

        return "-"


    def transcribe_mp3(self, mp3_path):
        wav_path = self.convert_mp3_to_wav(mp3_path)
        transcription = self.transcribe_audio(wav_path)
        os.remove(wav_path)
        return transcription

    def transcribe(self, path):
        file_type = path[-3:].lower()
        if file_type in ['mp3', 'wav']:
            return self.transcribe_audio(path)
        elif file_type in ['mp4']:
            return self.transcribe_video(path)
        else:
            return "Formato de archivo no compatible."

    #TODO: si lo vemos necesario implementar
    def transcribe_video(self, video_path):
        print("ERROR: No esta implementada la transcripción de video")
        return "ESTO AL MENOS LLEGA"

