# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import collections
import random
import tokenization
import tensorflow as tf
import spacy

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")

nlp = spacy.load('en_core_web_lg')


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next, direction):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

        self.direction = direction

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "direction: %s\n" % self.direction
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
        while len(segment_ids) < max_seq_length:
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
        while len(masked_lm_ids) < max_predictions_per_seq:
            masked_lm_ids.append(0)
        while len(masked_lm_weights) < max_predictions_per_seq:
            masked_lm_weights.append(0.0)

        next_sentence_label = 1 if instance.is_random_next else 0
        sentence_direction = instance.direction

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        features["next_sentence_labels"] = create_int_feature([next_sentence_label])
        features["sentence_direction"] = create_int_feature([sentence_direction])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 2:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)
            doc = nlp(' '.join(doc_tokens))
            tmp = []
            for sent in doc.sents:
                if sent.text:
                    if len(sent.text.split(" ")) > 4:
                        tmp.append(sent.text)
            examples.append(tmp)
            # if len(examples) > 1:
            #     break

    return examples


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_training_instances_squad(input_files, tokenizer, max_seq_length,
                                    dupe_factor, short_seq_prob, masked_lm_prob,
                                    max_predictions_per_seq, rng):
    """Create `TrainingInstance`s from raw text."""
    # all_documents = []
    # Input file format:
    # json file from SQUAD2
    all_documents = read_squad_examples(input_files[0], is_training=True)
    # json file from SQUAD1
    if (len(input_files) > 1):
        all_documents += read_squad_examples(input_files[1], is_training=True)
        if (len(input_files) > 2):
            with tf.gfile.GFile(input_files[2], "r") as reader:
                while True:
                    line = tokenization.convert_to_unicode(reader.readline())
                    if not line:
                        break
                    line = line.strip()

                    # Empty lines are used as document delimiters

                    doc = nlp(line)
                    tmp = []
                    for sent in doc.sents:
                        if sent.text:
                            if len(sent.text.split(" ")) > 4:
                                tmp.append(sent.text)
                    all_documents.append(tmp)

    # Remove empty documents and at least 8 words
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)
    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document_squad(
                    all_documents, document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng, tokenizer))

    rng.shuffle(instances)
    return instances


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for input_file in input_files:
        with tf.gfile.GFile(input_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(tokens)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document_squad(
                    all_documents, document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

    rng.shuffle(instances)
    return instances


def create_instances_from_document_squad(
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng, tokenizer):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]
    # Account for [CLS],  2 [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(10, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    # case = -1
    first_segment_length = 0
    # times1 = 0
    # times2 = 0
    while i < len(document):
        segment = document[i].split(" ")
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                # forward =1
                # unrelated=2
                # backward=0
                direction = 1
                # reduced .3 to .2 as we have direction
                if len(document) == 1 or len(current_chunk) == 1 or rng.random() < 0.2:
                    is_random_next = True
                    direction = 2
                    target_b_length = target_seq_length - len(tokens_a)
                    # times1 += 1
                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.

                    random_document_index = rng.randint(0, len(all_documents) - 1)
                    while random_document_index == document_index:
                        random_document_index = rng.randint(0, len(all_documents) - 1)

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b += (random_document[j].split(" "))
                        if len(tokens_b) >= target_b_length:
                            break

                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    direction = 1
                    for j in range(a_end, len(current_chunk)):
                        tokens_b += (current_chunk[j])
                    # times2 += 1
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 2

                assert len(tokens_b) >= 2
                # only go backward if forward
                if direction == 1:
                    # 40% case do backward
                    if rng.random() < 0.4:
                        tmp_token = tokens_a
                        tokens_a = tokens_b
                        tokens_b = tmp_token
                        direction = 0

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    if token:
                        tokens.append(token)
                        segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    if token:
                        tokens.append(token)
                        segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                assert len(segment_ids) == len(tokens)
                (tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, tokenizer)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    direction=direction,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0

        i += 1
    # print("*******times******")
    # print(times1, times2)
    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng, tokenizer):
    """Creates the predictions for the masked LM objective."""
    output_tokens = []
    cand_indexes_tmp = []
    text = ""
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            output_tokens.append(token)
            continue
        cand_indexes_tmp.append(i)
        output_tokens.append(token.lower())
        text += token + " "

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    ## NER part
    cand_indexes_final = []
    cand_indexes_final1 = []
    cand_indexes_final2 = []
    ner = nlp(text)
    word_type = dict()
    ents = []
    nouns = []
    for token in ner:
        word_type[token.text.lower()] = token
        if token.pos_ == "NOUN" and len(token) > 2:
            nouns.append(token.text.lower())
    for entities in ner.ents:
        for word in tokenizer.tokenize(entities.text):
            if word in word_type:
                if not word_type[word].is_punct and word_type[word].pos_ != "CCONJ" and \
                        word_type[word].pos_ != "DET" and len(word) > 2:
                    ents.append(word)

    # if ents.__len__() < max_predictions_per_seq // 3:
    #     for token in ner:
    #         word = token.lower()
    #         if rng.random() > 0.5 and word not in ents :
    #             ents.append(word)

    for i in cand_indexes_tmp:
        if tokens[i].lower() in ents:
            cand_indexes_final1.append(i)

            if i - 1 not in cand_indexes_final1 and tokens[i - 1] not in ['[CLS]', '[SEP]'] and rng.random() > .5:
                cand_indexes_final1.append(i - 1)
            if i + 1 not in cand_indexes_final1 and tokens[i - 1] not in ['[CLS]', '[SEP]'] and rng.random() > .5:
                cand_indexes_final1.append(i + 1)
        elif ents.__len__() < max_predictions_per_seq // 3 and tokens[i].lower() in nouns:
            if rng.random() > 0.75 and i not in cand_indexes_final1:
                cand_indexes_final1.append(i)

        if i not in cand_indexes_final1:
            cand_indexes_final2.append(i)
    rng.shuffle(cand_indexes_final2)
    rng.shuffle(cand_indexes_final1)
    cand_indexes_final = cand_indexes_final1 + cand_indexes_final2

    masked_lms = []
    covered_indexes = []
    for index in cand_indexes_final:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.append(index)

        masked_token = None

        probab = 0.8
        if index in cand_indexes_final1:
            probab = 0.5
        # 50/30% of the time, keep original
        if rng.random() > probab:
            masked_token = "[MASK]"
        else:
            # 70% of the time, keep original
            if rng.random() > 0.3:
                masked_token = tokens[index].lower()
            # 30% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)].lower()

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index].lower()))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label.lower())

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    # CaseSensitivetokenizer = tokenization.FullTokenizer(
    #     vocab_file=FLAGS.vocab_file, do_lower_case=False)
    # CaseSensitivetokenizer.tokenize()
    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)

    rng = random.Random(FLAGS.random_seed)
    instances = create_training_instances_squad(
        input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
        FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
        rng)

    output_files = FLAGS.output_file.split(",")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
