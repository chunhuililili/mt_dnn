from transformers import *
#from transformers import BertConfig, BertModel, BertTokenizer
#from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from module.san_model import SanModel
from modeling_nezha import BertForSequenceClassification
MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
#    "nezha": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    "xlm": (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer),
    "san": (BertConfig, SanModel, BertTokenizer),
    "electra": (ElectraConfig, ElectraModel, ElectraTokenizer),
    "t5": (T5Config, T5EncoderModel, T5Tokenizer),
    "deberta": (DebertaConfig, DebertaModel, DebertaTokenizer)
}
