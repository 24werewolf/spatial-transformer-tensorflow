import os
import scipy.io
import re

BaseDir = '/home/lazycal/workspace/qudou/output/'
regexp = re.compile(r'(\d+).mp4.avi')

def fetch(video_name, frame_id):
    video_name = regexp.match(video_name).group(1)
    path = os.path.join(BaseDir, video_name, '{:04d}'.format(frame_id))
    mat = scipy.io.loadmat(path)
    print('Read {}. Shape={}'.format(path, mat['res'].shape))
    return mat['res']