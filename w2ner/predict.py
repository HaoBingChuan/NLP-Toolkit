import os.path
from deepke.name_entity_re.standard.w2ner import (
    List,
    InputExample,
    NerProcessor,
    Vocabulary,
    fill_vocab,
    Model,
    AutoTokenizer,
    dis2idx,
    decode,
)
import numpy as np
import torch
import sys

sys.path.append(os.getcwd())
from logConfig import logger
from config import DEVICE


class ParaConfig:
    data_dir = "w2ner/data/people's-daily"
    save_path = "save_model/w2ner"
    learning_rate = 0.001
    max_seq_len = 200
    use_bert_last_4_layers = True
    bert_name = "save_model/bert-base-chinese"
    dist_emb_size = 20
    type_emb_size = 20
    lstm_hid_size = 512
    conv_hid_size = 96
    bert_hid_size = 768
    biaffine_size = 512
    ffnn_hid_size = 288
    dilation = [1, 2, 3]
    emb_dropout = 0.5
    conv_dropout = 0.5
    out_dropout = 0.33


def trans_dataset(config, examples: List[InputExample]):
    D = []
    for example in examples:
        span_infos = []
        sentence, label, d = example.text_a.split(" "), example.label, []
        if len(sentence) > config.max_seq_len:
            continue
        assert len(sentence) == len(label)
        i = 0
        while i < len(sentence):
            flag = label[i]
            if flag[0] == "B":
                start_index = i
                i += 1
                while i < len(sentence) and label[i][0] == "I":
                    i += 1
                d.append([start_index, i, flag[2:]])
            elif flag[0] == "I":
                start_index = i
                i += 1
                while i < len(sentence) and label[i][0] == "I":
                    i += 1
                d.append([start_index, i, flag[2:]])
            else:
                i += 1
        for s_e_flag in d:
            start_span, end_span, flag = s_e_flag[0], s_e_flag[1], s_e_flag[2]
            span_infos.append(
                {"index": list(range(start_span, end_span)), "type": flag}
            )
        D.append({"sentence": sentence, "ner": span_infos})

    return D


def preload(config: ParaConfig):

    logger.info("*********Building the vocabulary(Need Your Training Set!!!!)*********")

    processor = NerProcessor()

    train_examples = processor.get_train_examples(config.data_dir)
    train_data = trans_dataset(config, train_examples)

    vocab = Vocabulary()
    train_ent_num = fill_vocab(vocab, train_data)

    logger.info("Building Done!\n")

    config.label_num = len(vocab.label2id)

    logger.info("*********Loading the Final preload_model*********")
    preload_model = Model(config)
    if torch.cuda.is_available():
        preload_model = preload_model.cuda(device=DEVICE)
        preload_model.load_state_dict(
            torch.load(
                os.path.join(
                    config.save_path,
                    "pytorch_model.bin",
                )
            )
        )
    else:
        preload_model = preload_model.cpu()
        preload_model.load_state_dict(
            torch.load(
                os.path.join(
                    config.save_path,
                    "pytorch_model.bin",
                ),map_location=torch.device('cpu')
            )
        )

    tokenizer = AutoTokenizer.from_pretrained(config.bert_name)
    logger.info("Loading Done!\n")
    return preload_model, tokenizer, vocab


def w2ner(text, preload_model, tokenizer, vocab):
    entities = []
    entity_types = []
    for i in range(len(text) // 500 + 1):
        s = i * 500
        e = (i + 1) * 500
        sentence = text[s:e]
        if sentence:
            entity_text = sentence  # 待提取实体文本
            length = len([word for word in entity_text])
            tokens = [tokenizer.tokenize(word) for word in entity_text]
            pieces = [piece for pieces in tokens for piece in pieces]

            bert_inputs = tokenizer.convert_tokens_to_ids(pieces)

            bert_inputs = np.array(
                [tokenizer.cls_token_id] + bert_inputs + [tokenizer.sep_token_id]
            )
            pieces2word = np.zeros((length, len(bert_inputs)), dtype=np.bool)
            grid_mask2d = np.ones((length, length), dtype=np.bool)
            dist_inputs = np.zeros((length, length), dtype=np.int)
            sent_length = length

            if tokenizer is not None:
                start = 0
                for i, pieces in enumerate(tokens):
                    if len(pieces) == 0:
                        continue
                    pieces = list(range(start, start + len(pieces)))
                    pieces2word[i, pieces[0] + 1 : pieces[-1] + 2] = 1
                    start += len(pieces)

            for k in range(length):
                dist_inputs[k, :] += k
                dist_inputs[:, k] -= k

            for i in range(length):
                for j in range(length):
                    if dist_inputs[i, j] < 0:
                        dist_inputs[i, j] = dis2idx[-dist_inputs[i, j]] + 9
                    else:
                        dist_inputs[i, j] = dis2idx[dist_inputs[i, j]]
            dist_inputs[dist_inputs == 0] = 19

            result = []
            with torch.no_grad():
                if torch.cuda.is_available():
                    bert_inputs = torch.tensor([bert_inputs], dtype=torch.long).cuda(
                        device=DEVICE
                    )
                    grid_mask2d = torch.tensor([grid_mask2d], dtype=torch.bool).cuda(
                        device=DEVICE
                    )
                    dist_inputs = torch.tensor([dist_inputs], dtype=torch.long).cuda(
                        device=DEVICE
                    )
                    pieces2word = torch.tensor([pieces2word], dtype=torch.bool).cuda(
                        device=DEVICE
                    )
                    sent_length = torch.tensor([sent_length], dtype=torch.long).cuda(
                        device=DEVICE
                    )
                else:
                    bert_inputs = torch.tensor([bert_inputs], dtype=torch.long)
                    grid_mask2d = torch.tensor([grid_mask2d], dtype=torch.bool)
                    dist_inputs = torch.tensor([dist_inputs], dtype=torch.long)
                    pieces2word = torch.tensor([pieces2word], dtype=torch.bool)
                    sent_length = torch.tensor([sent_length], dtype=torch.long)
                preload_model.eval()
                outputs = preload_model(
                    bert_inputs,
                    grid_mask2d,
                    dist_inputs,
                    pieces2word,
                    sent_length,
                )

                length = sent_length
                grid_mask2d = grid_mask2d.clone()
                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = decode(
                    outputs.cpu().numpy(), entity_text, length.cpu().numpy()
                )

                decode_entities = decode_entities[0]

                input_sentence = [word for word in entity_text]
                for ner in decode_entities:

                    ner_indexes, ner_label = ner
                    entity = "".join(
                        [input_sentence[ner_index] for ner_index in ner_indexes]
                    )
                    entity_label = vocab.id2label[ner_label]
                    result.append((entity, entity_label))

            entity_map = {"loc": "LOCATION", "org": "ORGANIZATION", "per": "PERSON"}
            for entity, entity_label in result:
                entities.append(entity)
                entity_types.append(entity_map[entity_label])
    info = {
        "text": text,
        "entities": entities,
        "entity_types": entity_types,
    }
    return info


if __name__ == "__main__":
    preload_model, tokenizer, vocab = preload(config=ParaConfig)
    text = "秦始皇兵马俑位于陕西省西安市，1961年被国务院公布为第一批全国重点文物保护单位，是世界八大奇迹之一。"
    print(w2ner(text, preload_model, tokenizer, vocab))
