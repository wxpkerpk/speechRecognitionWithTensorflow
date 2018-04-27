import os

import numpy
import numpy as np
import librosa
from collections import Counter
import wave
# import extensions as xx
from random import shuffle
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

init = False
data_dir = "E:\data_thchs30\wx"

CHUNK = 4096
def get_wav_files(wav_path=data_dir):
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith(".wav") or filename.endswith(".WAV"):
                filename_path = os.sep.join([dirpath, filename])
                if os.stat(filename_path).st_size < 240000:
                    continue
                wav_files.append(filename_path)

    return wav_files


wav_files = get_wav_files()


def get_wav_lable(wav_files=wav_files, label_file=data_dir):
    labels_dict = {}
    for (dirpath, dirnames, filenames) in os.walk(label_file):
        for filename in filenames:
            if filename.endswith('.trn') or filename.endswith('.TRN'):
                filename_path = os.sep.join([dirpath, filename])
                fd = open(filename_path, 'r')
                text = fd.readline().strip('\n')
                label_id = filename.split('.', 1)[0]
                if len(text) > 0:
                    labels_dict[label_id] = text

    labels = []
    new_wav_files = []
    for wav_file in wav_files:
        wav_id = os.path.basename(wav_file).split('.')[0]
        if wav_id in labels_dict:
            labels.append(labels_dict[wav_id])
            new_wav_files.append(wav_file)

    return new_wav_files, labels



def load_wav_file(name):
  f = wave.open(name, "rb")
  # print("loading %s"%name)
  chunk = []
  data0 = f.readframes(CHUNK)
  while data0:  # f.getnframes()
    # data=numpy.fromstring(data0, dtype='float32')
    # data = numpy.fromstring(data0, dtype='uint16')
    data = numpy.fromstring(data0, dtype='uint8')
    data = (data + 128) / 255.  # 0-1 for Better convergence
    # chunks.append(data)
    chunk.extend(data)
    data0 = f.readframes(CHUNK)
  # finally trim:
  chunk = chunk[0:CHUNK * 2]  # should be enough for now -> cut
  chunk.extend(numpy.zeros(CHUNK * 2 - len(chunk)))  # fill with padding 0's
  # print("%s loaded"%name)
  return chunk

def get_wav_files(wav_path=data_dir):
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith(".wav") or filename.endswith(".WAV"):
                filename_path = os.sep.join([dirpath, filename])
                if os.stat(filename_path).st_size < 240000:
                    continue
                wav_files.append(filename_path)

    return wav_files


def getWavName(wavName):
    var = wavName.split('.', 1)[0]
    return var


def mfcc_batch_generator(batch_size=10):
    batch_features = []
    labels = []

    ########################################
    files = get_wav_files(data_dir)
    wav_files, labeles = get_wav_lable(files, data_dir)
    all_words = []
    for label in labeles:
        # 字符分解
        all_words += [word for word in label]

    counter = Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    words, _ = zip(*count_pairs)
    words_size = len(words)
    print(u"词汇表大小：", words_size)

    word_num_map = dict(zip(words, range(len(words))))

    # 当字符不在已经收集的words中时，赋予其应当的num，这是一个动态的结果
    to_num = lambda word: word_num_map.get(word, len(words))

    # 将单个file的标签映射为num 返回对应list,最终all file组成嵌套list
    labels_vector = [list(map(to_num, label)) for label in labeles]
    label_max_len = np.max([len(label) for label in labels_vector])

    wav_feature = {}
    for i in range(len(wav_files)):
        name = wav_files[i]
        feature = labels_vector[i]
        wav_feature[name] = feature

    while True:
        print("loaded batch of %d files" % len(files))
        shuffle(files)
        for wav in files:
            if not wav.endswith(".wav"): continue
            wave, sr = librosa.load(wav, mono=True)
            id = wav
            label = wav_feature[id]
            # 取零补齐
            # mfcc 默认的计算长度为20(n_mfcc of mfcc) 作为channel length
            mfcc = librosa.feature.mfcc(wave, sr)

            labels.append(label)
            # print(np.array(mfcc).shape)
            mfcc = np.pad(mfcc, ((0, 0), (0, 80 - len(mfcc[0]))), mode='constant', constant_values=0)
            batch_features.append(np.array(mfcc))
            if len(batch_features) >= batch_size:
                for mfcc in batch_features:
                    while len(mfcc) < wav_max_len:
                        mfcc.append([0] * 20)
                for label in labels:
                    while len(label) < label_max_len:
                        label.append(0)
                # print(np.array(batch_features).shape)
                # yield np.array(batch_features), labels
                yield batch_features, labels  # basic_rnn_seq2seq inputs must be a sequence
                batch_features = []  # Reset for next batch
                labels = []
