import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2
import re
import unicodedata
import json
import io

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data
finished_files_dir = "../pointer_bert/finished_files"
chunks_dir = os.path.join(finished_files_dir, "chunked")

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')
  
def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z]+", " ", w)
    
    w = w.rstrip().strip()
    
    return w

def chunk_file(set_name):
  in_file = 'finished_files/%s.bin' % set_name
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['train', 'val', 'test']:
    print "Splitting %s data into chunks..." % set_name
    chunk_file(set_name)
  print "Saved chunked data in %s" % chunks_dir

if __name__ == '__main__':
  
    if not os.path.exists(finished_files_dir):
        os.makedirs(finished_files_dir)

    # generate eval tar
    txt_file = 'gs://bytecup2018/bytecup2018/bytecup.corpus.eval.txt'
    out_file =  os.path.join(finished_files_dir, "val.bin")
    with tf.gfile.Open(txt_file, "r") as f:
        with open(out_file, 'wb') as writer:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                print "Writing story %i of %i; %.2f percent done" % (idx, len(lines), float(idx)*100.0/float(len(lines)))
                article = preprocess_sentence(json.loads(line)['content'])
                abstract = preprocess_sentence(json.loads(line)['title'])
                abstract = SENTENCE_START+abstract+SENTENCE_END
                # Write to tf.Example
                tf_example = example_pb2.Example()
                tf_example.features.feature['article'].bytes_list.value.extend([article])
                tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, tf_example_str))
    print "Finished writing file %s\n" % out_file

    # generate test tar
    txt_file = 'gs://bytecup2018/bytecup2018/bytecup.corpus.test.txt'
    out_file =  os.path.join(finished_files_dir, "test.bin")
    with tf.gfile.Open(txt_file, "r") as f:
        with open(out_file, 'wb') as writer:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                print "Writing story %i of %i; %.2f percent done" % (idx, len(lines), float(idx)*100.0/float(len(lines)))
                article = preprocess_sentence(json.loads(line)['content'])
                abstract = preprocess_sentence(json.loads(line)['title'])
                abstract = SENTENCE_START+abstract+SENTENCE_END
                # Write to tf.Example
                tf_example = example_pb2.Example()
                tf_example.features.feature['article'].bytes_list.value.extend([article])
                tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, tf_example_str))
    print "Finished writing file %s\n" % out_file  

    # generate train tar and vocab file
    train_files_size = 2
    out_file =  os.path.join(finished_files_dir, "train.bin")
    vocab_counter = collections.Counter()
    with open(out_file, 'wb') as writer:
        for i in range(train_files_size):
            txt_file = 'gs://bytecup2018/bytecup2018/bytecup.corpus.train.{}.txt'.format(i)
            with tf.gfile.Open(txt_file, "r") as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                  print "Writing story %i of %i; %.2f percent done" % (idx, len(lines), float(idx)*100.0/float(len(lines)))
                  article = preprocess_sentence(json.loads(line)['content'])
                  abstract = preprocess_sentence(json.loads(line)['title'])
                  abstract = SENTENCE_START+abstract+SENTENCE_END
                  # Write to tf.Example
                  tf_example = example_pb2.Example()
                  tf_example.features.feature['article'].bytes_list.value.extend([article])
                  tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
                  tf_example_str = tf_example.SerializeToString()
                  str_len = len(tf_example_str)
                  writer.write(struct.pack('q', str_len))
                  writer.write(struct.pack('%ds' % str_len, tf_example_str))

                  # Write the vocab to file, if applicable
                  art_tokens = article.split(' ')
                  abs_tokens = abstract.split(' ')
                  abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
                  tokens = art_tokens + abs_tokens
                  tokens = [t.strip() for t in tokens] # strip
                  tokens = [t for t in tokens if t!=""] # remove empty
                  vocab_counter.update(tokens)
    print "Finished writing file %s\n" % out_file

    # write vocab to file
    print "Writing vocab file..."
    with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print "Finished writing vocab file"

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    chunk_all()
