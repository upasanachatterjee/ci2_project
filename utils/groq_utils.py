from groq import Groq
import os
import json
from dotenv import load_dotenv
from utils.file_utils import load_json_file
load_dotenv()  # Load variables from .env into the environment

api_key = os.getenv("groq")

def get_groq_client():
    client = Groq(api_key=api_key)
    return client


def run_groq(client, prompt, max_tokens, system_prompt, response_format=False, model="llama-3.3-70b-versatile"):
    format = {"type": "json_object"} if response_format else None
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=1,
        max_completion_tokens=max_tokens,
        top_p=1,
        stream=False,
        stop=None,
        response_format=format,
    )
    return completion.choices[0].message.content


def generate_text_groq(idx, path, groq_client):
    json_data = load_json_file(path)
    word_count = len(json_data["original_text"].split())

    print("generating for id = ", idx, "word count = ", word_count)

    system_prompt = "Continue the following text, keeping the same tone"

    output = run_groq(
        groq_client, json_data["original_text"], word_count * 2, system_prompt
    )
    print("generated sequence of length = ", len("".join(output).split()))
    return output


def generate_bias_groq(prompt, client):
    system_prompt = ("classify the following text with one of the following biases:"
    + "left, center, right. Do this in the following json format {'bias' : 'text'}")
    output = run_groq(client, prompt, 1024, system_prompt, True)
    return json.loads(output)["bias"]


def generate_topic_similarity_groq(client, entities, topics, model="llama-3.3-70b-versatile"):
    system_prompt = "Do not hallucinate. Do not make up information. Do not consider one off events. Do not consider outlier relationships. Do not consider one off major news events. If the entity is a phrase, select the most relevant words for the topic."
    prompt = f"Given a group of topics and a group of entities, score from 0 to 1 for each topic-entity pair, where 0 means no similarity and 1 means very similar. Format the response as a json of the form {{ topic : {{ entity : score }} }}. Sort the results so that entities with higher scores are earlier. The topics are: {topics}. The entities are: {entities}."
        

    output = run_groq(client, prompt, 1024, system_prompt, True, model=model)

    return json.loads(output)

def generate_entites_sentiments_groq(client, text, model="llama-3.3-70b-versatile"):
    system_prompt = "Return responses in json format as follows {{entity1: sentiment1, entity2: sentiment2}}. Named entities should be present in the text. Do not hallucinate. Sentiment scores should be real numbers between -1 and +1 where -1 is strongly negative and +1 is strongly positive. Sentiment scores should reflect the author's perspective."

    prompt = f"Given the following text, identify the named entities and their sentiments. The text is: {text}"

    output = run_groq(client, prompt, 1024, system_prompt, True, model=model)
    return json.loads(output)

def generate_topic_groq(client, loaded_json):
    system_prompt = ("Are the following two texts on roughly the same topic?"
                     + "Use the following JSON format: {'same_topic': boolean}")
    prompt = json.dumps(
        {
            "text_1": loaded_json["generated_text"],
            "text_2": loaded_json["original_text"],
        }
    )
    output = run_groq(client, prompt, 1024, system_prompt, True)
    return json.loads(output)["same_topic"]


def generate_bias_groq_with_samples(prompt, sample, client):
    system_prompt = (
        "classify the following text with one of the following biases: "
        + "left, center, right. Do this in the following json format {'bias' : 'text'}. "
        + "Here are some samples "
        + sample
    )
    output = run_groq(client, prompt, 1024, system_prompt, True)
    return json.loads(output)["bias"]


def generate_summary_groq(client, word_count, original_text, model="llama-3.3-70b-versatile"):
    system_prompt = (f"Generate an exactly {word_count} word summary of the following article." +
                      "Return the response as a json following the format {'article': 'text'}"
                    )

    try:
        output = run_groq(client=client, 
                        prompt=f"{system_prompt}\n{original_text}", 
                        max_tokens=2048, 
                        system_prompt="", 
                        response_format=True,
                        model=model)
        
        summary = json.loads(output)["article"]
        print("generated sequence of length = ", len("".join(summary).split()))
    except Exception as e:
        print(f"error! returning empty")
        summary = ""
    return summary