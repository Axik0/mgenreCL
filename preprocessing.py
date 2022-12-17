import os
import librosa
import audioread.ffdec
import soundfile as sf
import math
import json

import time


def timer(func):
    def wrapper(*args,  **kwargs):
        tic = time.perf_counter()
        func(*args, **kwargs)
        print(f"Time passed: {round(time.perf_counter()-tic, 2)}s")
    return wrapper


SR = 22050
DURATION = 30
SAMPLES_TOTAL = SR * DURATION

NG_PATH = r"C:\Users\ks\Desktop\postpunk"
DS_PATH = r"C:\Users\ks\Downloads\Data\genres_original"
RES_PATH = 'extr_data.json'

# this script requires ffmpeg.exe decoder app in a root folder

@timer
def split_raw_files(input_path, block_size):
    """creates folder in input path with genre name and puts all audio chunks inside"""
    result_folder_name = 'result'
    # prepare list of all files
    file_paths = [os.path.join(input_path, f) for (input_path, dirnames, filenames) in os.walk(input_path) for f in filenames]
    # create directory or pass if it already exists
    try:
        os.mkdir(os.path.join(input_path, result_folder_name))
    except FileExistsError as error:
        print('This folder already exists')
    fc = len(file_paths)
    tot_splits = 0
    for fp in file_paths[:5]:
        with audioread.audio_open(fp) as m:
            signal, sr = librosa.load(m, sr=SR)
            ns_block = block_size*sr
            num_splits = int(len(signal)/ns_block)
            splits = []
            tot_splits += num_splits
            for s in range(num_splits):
                splits.append(signal[s*ns_block:(s+1)*ns_block])
                fname = input_path.split("\\")[-1]+"_"+str(fc)+"_"+str(s)+".wav"
                sf.write(os.path.join(input_path, result_folder_name, fname), splits[s], sr)
    print(f"{fc} tracks have been splitted onto {tot_splits} parts.")


@timer
def gen_mfcc(dataset_path, result_path, n_mfcc=13, n_fft=2048, hop_length=512, n_segments=5):
    """loop through all folders and files, get genres as folder names,
    then get all files, split and generate mfcc for each segment"""
    data = {'genres': [], 'mfcc': [], }
    # each segment must obtain exact and equal quantity of mfcc vectors:
    ns_p_segment = int(SAMPLES_TOTAL/n_segments)
    lim_nmfcc_p_segment = math.ceil(ns_p_segment / hop_length)

    for i, (dp, dn, fn) in enumerate(os.walk(dataset_path)):
        # scan all levels omitting the root
        if dp != dataset_path:
            # gets genre label from current path (implies that genre is a directory name)
            dirpath_splitted = dp.split('\\')
            genre = dirpath_splitted[-1]
            data['genres'].append(genre)
            print(f"Processing {len(fn)} samples of {genre}")
            # for each audio file in this folder (fn = list of all filenames), open it and split onto n_segments
            for f in fn[:2]:
                restored_fp = os.path.join(dp, f)
                with audioread.audio_open(restored_fp) as m:
                    signal, sr = librosa.load(m, sr=SR)
                    for s in range(n_segments):
                        s_start = ns_p_segment * s
                        s_finish = ns_p_segment * (s+1)
                        mfcc = librosa.feature.mfcc(y=signal[s_start:s_finish],
                                                    n_fft=n_fft,
                                                    hop_length=hop_length,
                                                    sr=SR,
                                                    n_mfcc=n_mfcc)
                        # each mfcc is a matrix, i.e. numpy 2d array of size n_mfcc(13) * #samples of fft for segment
                        # print(mfcc.shape, ns_p_segment/hop_length)
                        # if we have enough mfccs for that segment, use (store) it
                        if mfcc.shape[1] == lim_nmfcc_p_segment:
                            # we want each mfcc vector to be a row instead of column
                            mfcc = mfcc.T
                            data['mfcc'].append(mfcc.tolist())
                            # retrieves python list of lists (mfcc vectors) here
                            # data['labels'].append(i - 1)
                            print(f"{f}, segment:{s + 1}")
    # save results, uses json module for convenient data serialization
    with open(result_path, 'w') as rp:
        json.dump(data, rp, indent=4)


if __name__ == "__main__":
    # split_raw_files(NG_PATH, DURATION)
    gen_mfcc(DS_PATH, RES_PATH)
