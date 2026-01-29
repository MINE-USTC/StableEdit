from sklearn.metrics import matthews_corrcoef, f1_score
# matthews_corrcoef: 用于计算二分类问题的 Matthews/马修斯 相关系数（MCC），它是一个衡量二分类模型预测结果与实际标签之间相关性的指标。MCC 的值范围在 -1 到 1 之间，值越接近 1 表示模型预测越准确，值0表示随机预测，相当于抛硬币，值越接近 -1 表示模型预测越差。
# 马修斯相关系数的计算基于混淆矩阵，综合考虑了四个指标：MCC=(TP×TN - FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]可以处理类别不平衡的情形，该情况下其比准确率更可靠：假设数据集：90% 正面，10% 负面，一个"偷懒"的模型：全部预测为正面，此时准确率 = 90%（看起来很好），但MCC ≈ 0（实际上是随机预测水平）
# 总之，MCC 是一个比准确率更可靠的二分类评估指标，特别适合处理类别不平衡的情况，在情感分析任务中用来衡量模型的整体分类质量。
# 而F1 score: 用于计算二分类或多分类问题，它是精确率（Precision）和召回率（Recall）的调和平均数。F1-score 的值范围在 0 到 1 之间，值越高表示模型的分类性能越好。
# F1 score也能处理类别不平衡的问题中(使用加权F1 score)，但它更关注于正类的预测性能，尤其是在正类样本较少时（比如在疾病检测，垃圾邮件识别中，而MCC中正负类是平等的）。F1 score 的计算公式为：F1 = 2 × (Precision × Recall) / (Precision + Recall)，其中 Precision 是正确预测的正类样本占所有预测为正类样本的比例，Recall 是正确预测的正类样本占所有实际为正类样本的比例。
from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from glue_eval.useful_functions import load_data, load_data_split, MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP
import time
import torch
import numpy as np

MAX_NUMBER_OF_FEW_SHOTS = 100


class SSTEval():
    def __init__(self, model, tokenizer, number_of_tests = None, number_of_few_shots = 0, eval_split = 'validation'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {number_of_few_shots}"
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.few_shots, self.eval_dataset = load_data_split('glue_eval/dataset/sst2.pkl', number_of_few_shots, number_of_tests)
        # print(f"the few shot in SSTEval is {self.few_shots}\n")
        # the few shot in SSTEval is [](这里是因为number_of_few_shots = 0)
        # print(f"the eval_dataset in SSTEval is {self.eval_dataset}\n") 
        # e.g., [..., {'sentence': "it 's one of those baseball pictures where the hero is stoic , the wife is patient , the kids are as cute as all get-out and the odds against success are long enough to intimidate , but short enough to make a dream seem possible . ", 'label': 1, 'idx': 104}, ...]
        # self.eval_dataset的长度是由number_of_tests决定的，如果number_of_tests = 100, 那么self.eval_dataset的长度就是100；但如果number_of_tests=None, 那么self.eval_dataset的长度就是所有的数据集长度-100(few_shots的长度)

        self._initialize_prompts()


    def _initialize_prompts(self):
        self.prefix_prompt = 'Review :' # 我猜这里的冒号前面的空格（包括下面的sentiment后的冒号前面有一个空格）可能会帮助模型更好的识别这是一个提示标签而非普通文本的一部分。
        self.postfix_prompt = '\nSentiment :'
        self.few_shot_context = []
        for _, few_shot in enumerate(self.few_shots):
            self.few_shot_context.append(f"{self.prefix_prompt} {few_shot['sentence']}{self.postfix_prompt} {'positive' if few_shot['label']==1 else 'negative'}\n")

    def _create_prompt(self, example, gen_len):
        question = self.prefix_prompt + example['sentence'] + self.postfix_prompt
        question_token_length = len(self.tokenizer(question)["input_ids"])
        # 下面的代码是确定可以用于 few-shot 示例的 token 长度
        # 上下文窗口是指模型在一次处理过程中可以考虑的最大 token 数量。对于语言模型来说，上下文窗口的大小决定了模型在生成文本时可以参考的前后文长度。上下文窗口越大，模型可以处理的上下文信息就越多，从而生成更连贯和一致的文本。
        remaining_token_length = MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP[self.model.config._name_or_path.lower().split('/')[-1]] - question_token_length - gen_len
        actual_few_shot = ""
        for few_shot in self.few_shot_context:
            few_shot_token_length = len(self.tokenizer(few_shot)["input_ids"])
            remaining_token_length -= few_shot_token_length
            if remaining_token_length < 0:
                break 
            actual_few_shot += few_shot
        input_prompt = actual_few_shot + question
        return input_prompt, example['sentence'], example['label']


    def _get_answer(self, generated_text):
        # split('Sentiment :') 方法将 generated_text 按照 'Sentiment :' 这个字符串进行分割，返回一个字符串列表：['Review :( serry ) wants to blend politics and drama, an admirable ambition. ', 'Positive \n Review']
        # .strip()方法用于去除字符串两端的空白字符（包括空格、换行符等），事实上调用一次就足够了
        answer_text = generated_text.split('Sentiment :')[-1].strip().strip()
        # print(f"answer_text: {answer_text}") # answer_text: Positive \n Review
        if 'positive' in answer_text.lower():
            return 1
        elif 'negative' in answer_text.lower():
            return 0

        return -1


    def evaluate(self, gen_len = 3, print_logs = False):
        pos_tok, neg_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['positive', 'negative'])
        # 这里之所以使用self.tokenizer对文本进行tokenize之后获取key=‘input_ids’对应的值为一维列表是因为代码中并未设置return_tensors参数，设置return_tensors='pt'返回的结果是二维张量，设置return_tensors='np'返回的结果是numpy数组
        # print(f"pos_tok: {pos_tok}, neg_tok: {neg_tok}")
        # 在使用的模型是gpt-j-6b时，pos_tok: [3967], neg_tok: [4633]，和明显和下面的llama模型得到的结果不同，所以下面专门记录了关于llama模型的结果
        # 在使用的模型是llama时，pos_tok: [128000, 6928], neg_tok: [128000, 8389]，这里的结果pos_tok 和 neg_tok 是通过使用 self.tokenizer 对字符串 positive 和 negative 进行编码后的 token IDs
        # 这些 token IDs 是将单词或子词转换为模型能够理解的数字表示。其中128000是BOS token id
        # f" {n}" 是将 positive 和 negative 加上空格（在词前）进行处理。这有时是为了处理模型对首字母空格的要求（如 BERT、GPT 等）???
        # ["input_ids"] 提取的是 tokenizer 返回字典中的 input_ids 字段，它是一个数字列表，表示文本对应的 token IDs。
        if 'llama' in self.model.config._name_or_path.lower() or "mistral" in self.model.config._name_or_path.lower():
            # Remove the bos token
            # LLaMA Tokenizer 在 Tokenize 词语时，会在最前面额外添加一个 BOS（Beginning of Sentence）Token。由于 BOS Token 在大多数 NLP 任务（分类、嵌入计算等）中是不需要的，所以代码 pos_tok[1:] 会 去掉 BOS Token，只保留 "positive" 和 "negative" 的实际 token ID
            pos_tok = pos_tok[1:] # pos_tok: [6928]
            neg_tok = neg_tok[1:] # neg_tok: [8389]


        
        pos_len, neg_len = (len(n) for n in [pos_tok, neg_tok]) # pos_len: 1, neg_len: 1
        suffixes = {0: ['positive', pos_tok, pos_len], 1: ['negative', neg_tok, neg_len]}

        correct = 0
        incorrect = 0
        invalid = 0

        pos_correct = 0 
        neg_correct = 0 
        pos_incorrect = 0 
        neg_incorrect = 0

        predictions = []
        labels = []
        predictions_new = []
        stored_generations = []
        start = time.time()

        for s, example in enumerate(self.eval_dataset):
            # if s == 0:
            #     print(f"example: {example}")
                # example: {'sentence': '( serry ) wants to blend politics and drama , an admirable ambition . ', 'label': 1, 'idx': 853}

            input_prompt, sentence, label = self._create_prompt(example, gen_len)
            # print(f"input_prompt: {input_prompt}") # input_prompt: Review :( serry ) wants to blend politics and drama , an admirable ambition . \nSentiment :
            labels.append(label)
            # self.tokenizer.encode()方法将输入的文本编码为 token IDs，返回的是一个 PyTorch Tensor
            input_prompt_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to('cuda')
            
            # 使用 encode_plus 方法获取 input_ids 和 attention_mask
            # self.tokenizer.pad_token = self.tokenizer.eos_token
            # encoding = self.tokenizer.encode_plus(input_prompt, return_tensors='pt', padding=True)
            # input_prompt_ids = encoding['input_ids'].to('cuda')
            # attention_mask = encoding['attention_mask'].to('cuda')

            # print(f"input_prompt_ids: {input_prompt_ids}") # e.g., tensor([[128000, 19997,..,3904, 551]], device='cuda:0')
            # print(f"the shape of input_prompt_ids: {input_prompt_ids.shape}") # torch.Size([1, 21])
            # self.tokenizer.decode()方法将token IDs解码为文本，参数skip_special_tokens=True表示跳过特殊 token,如[CLS], [SEP], [PAD]等
            input_prompt_text = self.tokenizer.decode(input_prompt_ids[0], skip_special_tokens=True)
            # print(f"input_prompt_text: {input_prompt_text}") # input_prompt_text: Review :( serry ) wants to blend politics and drama, an admirable ambition. \nSentiment :
            
        
            prefix_tok_len = len(self.tokenizer(input_prompt)["input_ids"])

            if 'llama' in self.model.config._name_or_path.lower() or "mistral" in self.model.config._name_or_path.lower():
                prefix_tok_len = prefix_tok_len - 1

            max_len = input_prompt_ids.shape[1] + gen_len
            # self.model.generate()方法用于生成文本，返回的是一个 PyTorch Tensor，最大长度为24个token，do_sample=False 表示不使用采样方法生成文本，而是使用贪婪搜索（greedy search）方法生成文本。而如果想要采用采样方法生成文本，需要将 do_sample 设置为 True，并且可以设置top_p参数(核采样)，这个参数表示采样时的概率阈值，它通过选择概率质量的前 p 百分比的 token（选取token的时候要先将token的概率分布进行排序） 来随机生成下一个 token。
            # 另外一个与采样过程相关的参数是 temperature, 用于控制生成文本的多样性。 它通过调整模型的输出概率分布来影响生成的token。（temperature 参数会对模型输出的 logits（此时还未进行softmax,仅仅是一些没有确定数值关系的数） 进行缩放，从而改变生成的概率分布: logits/temperature）temperature 是一个浮点数，通常取值范围在 (0, 1],当 temperature 越低，生成的文本越确定性（即更倾向于选择概率最高的 token），生成的文本会更加保守和重复。当 temperature 越高，生成的文本越多样化（即更倾向于选择概率较低的 token），生成的文本会更加随机和多样。
            # 这里的max_length参数控制的整个输出序列的最大长度，包含输入的input_prompt_ids和生成的文本长度。
            output = self.model.generate(input_prompt_ids, max_length = max_len, do_sample = False)
            # output = self.model.generate(input_prompt_ids, max_length=max_len, do_sample=False, attention_mask=attention_mask, pad_token_id=self.tokenizer.eos_token_id)
            # output.shape: torch.Size([1, 24]) ,包含输入的23个token和生成的1个token
            
            
            # print(output) # e.g., tensor([[128000, 19997,..., 3904, 551, 45003, 198, 19997]], device='cuda:0')
            # print(output.shape) # torch.Size([1, 24])
            
            # /data2/forest/anaconda3/envs/alphaedit/lib/python3.8/site-packages/transformers/generation/configuration_utils.py:590: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.6` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
            # warnings.warn(/data2/forest/anaconda3/envs/alphaedit/lib/python3.8/site-packages/transformers/generation/configuration_utils.py:595: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
            # 下面的第一个警告是因为在调用 self.model.generate 方法时，没有设置 attention_mask 和 pad_token_id，这可能会导致生成文本时出现意外行为。具体来说，模型在生成文本时需要知道哪些位置是实际的输入，哪些位置是填充（padding），以便正确地处理输入序列。因此，需要传递 attention_mask 参数来确保模型只关注实际的输入。另外，还需要传递 pad_token_id 参数，以便模型知道填充 token 的 ID。如果不传递 attention_mask 参数，模型会尝试从输入中推断 attention_mask，但这可能会导致意外行为。如果不传递 pad_token_id 参数，模型会将 pad_token_id 设置为 eos_token_id，这可能会导致意外行为（模型无法区分填充和实际结束的位置）。因此，建议传递 attention_mask 和 pad_token_id 参数以获得可靠的结果。
            # The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
            # Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
            # The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # print(f"generated_text: {generated_text}") # generated_text: Review :( serry ) wants to blend politics and drama, an admirable ambition. \nSentiment : positive \n Review
            answer = self._get_answer(generated_text)
            predictions.append(answer)

            probs = [0 for _ in suffixes.keys()]
            gen_texts = [0 for _ in suffixes.keys()]

            for i in range(len(suffixes.keys())):
                # suffixes = {0: ['positive', pos_tok, pos_len], 1: ['negative', neg_tok, neg_len]}
                prompt_tok = self.tokenizer([f"{input_prompt} {suffixes[i][0]}"], return_tensors="pt").to('cuda')

                with torch.no_grad():
                    logits = self.model(**prompt_tok).logits
                    # logits: [batch_size, sequence_length, vocab_size]，logits 是模型在最后一个线性层的输出结果，通常没有经过 softmax 操作。
                if 'llama' in self.model.config._name_or_path.lower() or "mistral" in self.model.config._name_or_path.lower():
                    logits = logits[:, 1:, :]



                cur_len = suffixes[i][2]

                for j in range(cur_len):
                    cur_tok = suffixes[i][1][j]
                    probs[i] += -torch.nn.functional.log_softmax(
                    logits[0, prefix_tok_len + j - 1, :], dim=0
                    )[cur_tok].item()
                probs[i] /= cur_len # -log(p('positive')) or -log(p('negative'))
                gen_texts[i] = self.tokenizer.decode(logits[0, prefix_tok_len - 1 : prefix_tok_len + cur_len - 1, :].argmax(dim = -1))


            prob_pos = np.exp(-probs[0])
            prob_neg = np.exp(-probs[1])

            print(f"prob_positive: {prob_pos}, prob_negative: {prob_neg}")

            # 这里的转换可能是为了和情感计算中的标准约定想匹配：0 → negative（负面情感），1 → positive（正面情感）
            answer_new = 1 if prob_pos > prob_neg else 0
            predictions_new.append(answer_new)
            print(f"prediction: {answer}, true: {label}")

            if answer == -1:
                invalid += 1
            else:

                if answer == label:
                    correct += 1

                    if label == 1:
                        pos_correct += 1
                    elif label == 0:
                        neg_correct += 1

                else:
                    incorrect += 1

                    if label == 1:
                        pos_incorrect += 1
                    elif label == 0:
                        neg_incorrect += 1

            exp_temp_dict = {
                'sentence': sentence,
                'input_prompt': input_prompt_text,
                'true_answer': 'positive' if label == 1 else 'negative',  
                'generated_text': generated_text.replace(input_prompt_text, ''),
                'answer': answer,
                'correct': answer == label,
                'prob_positive': prob_pos,
                'prob_negative': prob_neg,
                'highest_probability_answer': 'positive' if answer_new == 1 else 'negative',
                'correct_new': answer_new == label,
            }
            stored_generations.append(exp_temp_dict)

            if print_logs:
                # 是这里计算马修斯相关系数时：mcc = matthews_corrcoef(labels, predictions)，labels中的元素取值只有两种，因为这是二分类问题，但是predictions的取值有三种，比前者多了个-1，那这个时候马修斯相关系数该如何计算，f1-score又该如何计算
                # 下面的两行代码的处理方式会将 -1 视为第三个类别，可能这会导致错误或不准确的结果，因为变成了多分类问题
                mcc = matthews_corrcoef(labels, predictions)
                f1 = f1_score(labels, predictions, average='weighted')
                print(generated_text)
                print(correct, incorrect, invalid, s+1, '|', pos_correct, neg_correct, '|', pos_incorrect, neg_incorrect, '|ACC: ', correct / (correct + incorrect + invalid), '|MCC:', mcc, '|F1:', f1)
                print('--'*50)


        end = time.time()
        mcc = matthews_corrcoef(labels, predictions)
        # average='weighted'：计算每个类别的 F1-score，然后根据每个类别的支持（即该类别的样本数量）加权平均这些 F1-score。这样可以考虑类别不平衡的情况，使得样本数量较多的类别对最终的 F1-score 影响更大。
        # 对于labels 列表只包含 0 和 1 两个元素，而 predictions 列表包含 -1、0 和 1 三个元素这样的情况，在计算 F1-score 之前，可以将 predictions 中的 -1 元素移除或替换为一个有效的类别。
        # 我个人觉得比较合理的做法是过滤无效预测
        f1 = f1_score(labels, predictions, average='weighted')
        f1_new = f1_score(labels, predictions_new, average='weighted')
        result_dict = {
            'correct': correct,
            'incorrect': incorrect,
            'invalid': invalid,
            'total': s+1,
            'f1': f1,
            'f1_new': f1_new,
            'mcc': mcc,
            'time': end-start,
        }

        return result_dict, stored_generations

if __name__ == '__main__':
    # Load the tokenizer and model
    # sst_eval = SSTEval(None, None)
    # exit()
    # model_name = 'EleutherAI/gpt-j-6b'
    model_name  = 'meta-llama/Meta-Llama-3-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    sst_eval = SSTEval(model, tokenizer)
    correct, incorrect, invalid, total = sst_eval.evaluate(print_logs=False)