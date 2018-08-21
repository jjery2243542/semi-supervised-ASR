import sys
from pydub import AudioSegment

'''
usage: python3 convert.py [XXX.flac] [XXX.wav] [sample rate]
'''


src_path = sys.argv[1]
tar_path = sys.argv[2]
sr = int(sys.argv[3])

audio = AudioSegment.from_file(src_path, 'flac', frame_rate=sr)
audio.export(tar_path, format='wav')
