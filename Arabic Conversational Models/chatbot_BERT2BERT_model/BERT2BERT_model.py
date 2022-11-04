from transformers import EncoderDecoderModel, AutoTokenizer
from datasets import load_dataset
from arabert.preprocess import ArabertPreprocessor
from torch.utils.data.dataloader import DataLoader
from transformers import default_data_collator
from torch.utils.data.sampler import SequentialSampler
import torch
from tqdm.notebook import tqdm


from transformers import EncoderDecoderModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tareknaous/bert2bert-empathetic-response-msa")
model = EncoderDecoderModel.from_pretrained("tareknaous/bert2bert-empathetic-response-msa")

#model.to("cuda")
model.eval()

arabert_prep = ArabertPreprocessor(model_name="bert-large-arabertv02-twitter", keep_emojis=False)


def BERT_response(text):
  text_clean = arabert_prep.preprocess(text)
  inputs = tokenizer.encode_plus(text_clean,return_tensors='pt')

  outputs = model.generate(input_ids = inputs.input_ids,
                   attention_mask = inputs.attention_mask,
                   do_sample = True,
                   min_length=2,
                   max_length=50 ,
                   top_k = 0,
                   top_p = 0.9,
                   temperature = 0.5)

  preds = tokenizer.batch_decode(outputs)
  response = str(preds)
  response = response.replace("\'", '')
  response = response.replace("[[CLS]", '')
  response = response.replace("[SEP]]", '')
  response = str(arabert_prep.desegment(response))
  return response









#final
'''
from flask import Flask, request, jsonify


app = Flask(__name__)


@app.route('/chat', methods=['GET', 'POST'])
def chatBot():
    chatInput = request.json['chatInput']
    print(chatInput)
    return jsonify(chatBotReply= BERT_response(chatInput))


if __name__ == '__main__':
    app.run(host='0.0.0.0')

'''



