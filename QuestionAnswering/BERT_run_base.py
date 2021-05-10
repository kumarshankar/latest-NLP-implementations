
# import torch and transformers packages
import torch
from transformers import BertForQuestionAnswering, BertTokenizer

class modelHandler:
    '''
    This class handles Bert based QnA model initialitisation and related NLP preprocessing steps
    '''
    def __init__(self):
        self.model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")


    def get_answer(self, question, context_text):
        '''
        function to tokenize inputs, create segments and get answer outputs based on bert logits
        '''
        tokenizer = self.tokenizer
        model = self.model
        input_ids = tokenizer.encode(question, context_text)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        sep_index = input_ids.index(tokenizer.sep_token_id)
        num_seg_A = sep_index + 1
        num_seg_B = len(input_ids) - num_seg_A
        seg_ids = [0]* num_seg_A + [1] * num_seg_B
        # Lets run our model through an example
        outputs = model(torch.tensor([input_ids]), #tokens representing our input text
                                    token_type_ids = torch.tensor([seg_ids]),  #The segment ids to differentiate question from context text
                                    return_dict=True)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # Finding the tokens with highest start and end scores
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)

        # Combine the scores in the answers and print out
        answer = ' '.join(tokens[answer_start:answer_end+1])

        # Detokenize Bert word pieces
        if ' ##' in answer:
            answer = answer.replace(' ##','')

        return answer

# test the class
if __name__ == "__main__":
    # initialise model  handler with BERT
    model = modelHandler()
    # Lets ask three questions to the model

    examples = {"question1": "What is the color of the sky?",
                "context1": "The sky ranges from horizon to horizon. The blue colored sky appears beautiful",
                "question2": "Who runs faster than an elephant?",
                "context2": "There are several animals that are quicker than elephant. Cheetah for example sprints faster than an elephant",
                "question3": "What is the capital of India?",
                "context3": "The capital of India shifted from Calcutta to New Delhi in early twentieth century"}

    for i in range(1,4):
        print(model.get_answer(examples["question" + str(i)], examples["context" + str(i)]))


