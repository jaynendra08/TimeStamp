# converts videos to mp3(audio)
import os
import subprocess

files = os.listdir("videos")
for file in files:
    tutorial_number = int(file.split(" ")[1]) - 10
    # number = int(tutorial_number)-10
    print(file)
    file_name = file.split(".")[0]
    print(tutorial_number, file_name) 
    # subprocess.run(["ffmpeg", "-i",f"videos/{file}", f"audios/{tutorial_number}_{file_name}.mp3"])