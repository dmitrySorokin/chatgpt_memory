import openai 
import asyncio
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset
import numpy as np
import tiktoken
import time
from argparse import ArgumentParser
import os
from hashlib import md5


names = ['Mary', 'John', 'Daniel', 'Sandra']
actions = ['moved', 'went', 'went back', 'journeyed', 'travelled']
places = ['bathroom', 'hallway', 'garden', 'office', 'bedroom', 'kitchen']
choices_dict = {'names': names, 'actions': actions, 'places': places}


class MemoryDataset:
    def __init__(self, num_samples, choices_dict=choices_dict, num_facts=1, split='train', dataset='quality'):
        self.choices_dict = choices_dict
        self.dataset = load_dataset('tau/scrolls', dataset)[split]
        self.num_facts = num_facts
        self.num_samples = num_samples

    def __getitem__(self, ind):
        if self.num_samples:
            ind = np.random.randint(len(self.dataset))
        sample = self.dataset[ind]
        sample['fact'], sample['question'], sample['answer'] = self.generate_qa() 
        return sample
    
    def __len__(self):
        return self.num_samples

    def generate_qa(self):
        names, actions, places = self.choices_dict['names'], self.choices_dict['actions'], self.choices_dict['places']

        np.random.shuffle(names)
        facts, questions, answers = [], [], []
        for fact_num, name in zip(range(self.num_facts), names):
            action, place = np.random.choice(actions), np.random.choice(places)

            facts.append(f'{name} {action} to the {place}')
            questions.append(f'Where is {name}?')
            answers.append(place)

        facts = ', '.join(facts) + '.'
        questions = ' '.join(questions)
        answers = ', '.join(answers)
        
        return facts, questions, answers
    

def collate_fn(data, fact_position, message_len):
    fact = tokenizer.encode(data['fact'] + ' ')

    noize_mult = np.ceil(message_len / len(tokenizer.encode(data['input'])))
    message = ' '.join([data['input']] * int(noize_mult))
    message = tokenizer.encode(message)[:message_len]

    print(data['fact'], len(message))

    if fact_position is None:
        fact_position = np.random.randint(0, message_len - len(fact))

    message[fact_position: fact_position + len(fact)] = fact
    
    return tokenizer.decode(message), data['answer']



if __name__ == '__main__': 
    parser = ArgumentParser()

    parser.add_argument('--fact', type=int, default=0, help='position of fact in percents of msg length')
    parser.add_argument('--msg', type=int, default=1000, help='length of message in tokens')
    parser.add_argument('--model', type=str, choices=['gpt-4-1106-preview', 'gpt-4'], default='gpt-4-1106-preview')
    parser.add_argument('--sleep', type=int, default=10)
    parser.add_argument('--tot', type=int, default=1)
    args = parser.parse_args()

    msg_len = args.msg
    fact_pos = int(args.fact * msg_len)

    outdir = f'results/msg_{msg_len}'
    os.makedirs(outdir, exist_ok=True)


    np.random.seed(123)
    tokenizer = tiktoken.encoding_for_model(args.model)
    loop = asyncio.get_event_loop()

    with open('token') as inp:
        key = inp.read().strip()
    client = openai.AsyncOpenAI(api_key=key)


    outfile = outdir + f'/fact_{args.fact}.csv'
    if os.path.isfile(outfile):
        df = pd.read_csv(outfile, index_col=[0])
    else:
        df = pd.DataFrame({
            'answer': [],
            'gpt4answer': [],
            'result': [],
            'md5': [],
        })

    for i, item in enumerate(MemoryDataset(num_samples=1)):
        if i == args.tot:
            break

        query, true_answer = collate_fn(item, fact_pos, msg_len)
        messages = [ 
            {
                "role": "system",
                "content": "You are a intelligent assistant."
            },
            {
                "role": "user", 
                "content": 
                    "I give you a fact hidden in some random text and a question. "
                    "You need to answer the question based only on the information from the fact. "
                    "The answer should be exactly one word.\n"
                    f"{query}\n"
                    f"QUESTION: \"{item['question']}\""
            }
        ]

        md5msg = md5(messages[1]['content'].encode('utf8')).hexdigest()

        if len(df) > i:
            assert df.loc[i]['md5'] == md5msg
            continue

        completion = client.chat.completions.create(model=args.model, messages=messages)

        response = loop.run_until_complete(completion)
        gpt_answer = response.choices[0].message.content.strip().lower()

        if gpt_answer.endswith('.'):
            gpt_answer = gpt_answer[:-1]

        print(i, true_answer, gpt_answer)

        df.loc[len(df)] = [true_answer, gpt_answer, true_answer == gpt_answer, md5msg]

        df.to_csv(outfile)

        time.sleep(args.sleep)
