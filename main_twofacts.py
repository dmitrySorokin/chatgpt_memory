import openai 
import asyncio
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    loop = asyncio.get_event_loop()

    with open('token') as inp:
        key = inp.read().strip()

    df = pd.read_csv('two_facts_reasoning.csv')
    results = []
    answers = []

    for (row_id, row) in df.iterrows():        
        messages = [ 
            {
                "role": "system",
                "content": "You are a intelligent assistant."
            },
            {
                "role": "user", 
                "content": 
                    "I give you two facts and a question. "
                    "You need to answer the question based only on the information from the facts. The answer should be exactly one word.\n"
                    f"Fact 1: \"{row.fact1}\"\n"
                    f"Fact 2: \"{row.fact2}\"\n"
                    f"Question: \"{row.question}\""
            }
        ]

        client = openai.AsyncOpenAI(api_key=key)
        completion = client.chat.completions.create(model="gpt-4-1106-preview", messages=messages)
        response = loop.run_until_complete(completion)
        gpt_answer = response.choices[0].message.content.strip().lower()
        true_answer = row.answer

        if gpt_answer[-1] == '.':
            gpt_answer = gpt_answer[:-1]

        answers.append(gpt_answer)
        results.append(true_answer == gpt_answer)

        print(row_id, true_answer, gpt_answer)

    df.insert(4, "gpt4answer", answers, True)
    df.insert(5, "result", results, True)

    df.to_csv('two_facts_v11.csv')
