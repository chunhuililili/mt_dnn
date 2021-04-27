import json
import numpy as np

from data_utils.task_def import TaskType, DataFormat
import tasks

def load_data(file_path, task_def):
    data_format = task_def.data_type
    task_type = task_def.task_type
    label_dict = task_def.label_vocab
    if task_type == TaskType.Ranking:
        assert data_format == DataFormat.PremiseAndMultiHypothesis

    rows = []
    for line in open(file_path, encoding="utf-8"):
        #print(line)
        fields = line.strip("\n").split("\t")
        if data_format == DataFormat.PremiseOnly:
            assert len(fields) == 3
            row = {"uid": fields[0], "label": fields[1], "premise": fields[2]}
        elif data_format == DataFormat.PremiseAndOneHypothesis:
            assert len(fields) == 4
            row = {
                "uid": fields[0],
                "label": fields[1],
                "premise": fields[2],
                "hypothesis": fields[3]}
        elif data_format == DataFormat.PremiseAndMultiHypothesis:
            assert len(fields) > 5
            row = {"uid": fields[0], "ruid": fields[1].split(","), "label": fields[2], "premise": fields[3],
                   "hypothesis": fields[4:]}
        elif data_format == DataFormat.Seqence:
            row = {"uid": fields[0], "label": eval(fields[1]),  "premise": eval(fields[2])}

        elif data_format == DataFormat.MRC:
            row = {
                "uid": fields[0],
                "label": fields[1],
                "premise": fields[2],
                "hypothesis": fields[3]}
        elif data_format == DataFormat.SimPair:
            if len(fields)<4:
                continue
            row = {
                "uid": fields[0],
                "label": fields[1],
                "text_a": fields[2],
                "text_b": fields[3]}
        else:
            raise ValueError(data_format)

        task_obj = tasks.get_task_obj(task_def)
        if task_obj is not None:
            row["label"] = task_obj.input_parse_label(row["label"])
        elif task_type == TaskType.Ranking:
            labels = row["label"].split(",")
            if label_dict is not None:
                labels = [label_dict[label] for label in labels]
            else:
                labels = [float(label) for label in labels]
            row["label"] = int(np.argmax(labels))
            row["olabel"] = labels
        elif task_type == TaskType.Span:
            pass  # don't process row label
        elif task_type == TaskType.SeqenceLabeling:
            assert type(row["label"]) is list
            row["label"] = [label_dict[label] for label in row["label"]]

        rows.append(row)
    return rows

def load_clue_data(file_path, task_def):
    print("task_def={}".format(task_def))
    data_format = task_def.data_type
    task_type = task_def.task_type
    label_dict = task_def.label_vocab
    if task_type == TaskType.Ranking:
        assert data_format == DataFormat.PremiseAndMultiHypothesis

    rows = []
    uid = 0
    for line in open(file_path, encoding="utf-8"):
        record = json.loads(line)
        if task_def.name == "iflytek":
            label = record.get("label_des")
        elif task_def.name == "tnews":
            label = record.get("label_desc")
        else:
            label = record.get("label")
        if label is None:
            label = "0"
        if data_format == DataFormat.PremiseOnly:
            if task_def.name == "wsc":
                premise = read_wsc(record)
            else:
                premise = record["sentence"]
            row = {"premise": premise}
        elif data_format == DataFormat.PremiseAndOneHypothesis:
            if task_def.name == "csl":
                premise = record["abst"]
                hypothesis = ' '.join(record["keyword"])
            else:
                premise = record["sentence1"]
                hypothesis = record["sentence2"]
            row = {
                "premise": premise,
                "hypothesis": hypothesis}
        else:
            raise ValueError("not implemented yet")
        task_obj = tasks.get_task_obj(task_def)
        row["label"] = label
        if task_obj is not None:
            row["label"] = task_obj.input_parse_label(row["label"])
            if row["label"] is None:
                continue
        row["uid"] = f"{uid}"
        rows.append(row)
        uid += 1
    return rows


def read_wsc(line: dict) -> str:
    text_a = line['text']
    text_a_list = list(text_a)
    target = line['target']
    query = target['span1_text']
    query_idx = target['span1_index']
    pronoun = target['span2_text']
    pronoun_idx = target['span2_index']
    assert text_a[pronoun_idx: (
            pronoun_idx + len(pronoun))] == pronoun, "pronoun: {}".format(pronoun)
    assert text_a[query_idx: (query_idx + len(query))
           ] == query, "query: {}".format(query)
    if pronoun_idx > query_idx:
        text_a_list.insert(query_idx, "_")
        text_a_list.insert(query_idx + len(query) + 1, "_")
        text_a_list.insert(pronoun_idx + 2, "[")
        text_a_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
    else:
        text_a_list.insert(pronoun_idx, "[")
        text_a_list.insert(pronoun_idx + len(pronoun) + 1, "]")
        text_a_list.insert(query_idx + 2, "_")
        text_a_list.insert(query_idx + len(query) + 2 + 1, "_")
    text_a = "".join(text_a_list)
    return text_a


def load_qianyan_data(file_path, task_def):
    data_format = task_def.data_type

    rows = []
    for line in open(file_path, encoding="utf-8"):
        fields = line.strip("\n").split("\t")
        assert data_format == DataFormat.PremiseAndOneHypothesis
        if len(fields) == 4:
            row = {
                "uid": fields[0],
                "label": fields[1],
                "premise": fields[2],
                "hypothesis": fields[3]}
        elif len(fields) == 3:  # test
            row = {
                "uid": fields[0],
                "label": "0",
                "premise": fields[1],
                "hypothesis": fields[2]}
        else:
            raise ValueError(f"invalid line found: {line}")

        task_obj = tasks.get_task_obj(task_def)
        if task_obj is not None:
            row["label"] = task_obj.input_parse_label(row["label"])

        rows.append(row)
    return rows


def load_score_file(score_path, n_class):
    sample_id_2_pred_score_seg_dic = {}
    score_obj = json.loads(open(score_path, encoding="utf-8").read())
    assert (len(score_obj["scores"]) % len(score_obj["uids"]) == 0) and \
           (len(score_obj["scores"]) / len(score_obj["uids"]) == n_class), \
        "scores column size should equal to sample count or multiple of sample count (for classification problem)"

    scores = score_obj["scores"]
    score_segs = [scores[i * n_class: (i+1) * n_class] for i in range(len(score_obj["uids"]))]
    for sample_id, pred, score_seg in zip(score_obj["uids"], score_obj["predictions"], score_segs):
        sample_id_2_pred_score_seg_dic[sample_id] = (pred, score_seg)
    return sample_id_2_pred_score_seg_dic

