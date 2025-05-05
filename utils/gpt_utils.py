from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env into the environment

client = OpenAI(
  api_key=os.getenv("openai")
)

def create_chatgpt_tags(word_count, instruction, example):
    article = example["text"]
    response = client.responses.create(model="gpt-4o-mini",
        store=True,
        input=f"Generate a list of at most {word_count} topics for the following article. {instruction}. \n {article}",
        text= {
        "format" : {
            "name": "response_type",
            "schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "response": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                    }
                },
                "required": ["response"],
                "additionalProperties": False
            },
            "type": "json_schema" 
            }
        },
    )
    return response.output_text

def create_chatgpt_summary(word_count, instruction, example, model="gpt-4o-mini"):
  article = example["text"]
  completion = client.responses.create(
    model=model,
    store=True,
    input=f"Generate an exactly {word_count} word summary of the following article. {instruction}. \n {article}",
    text={
      "format" : {
        "name": "response_type",
        "schema": {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "properties": {
            "article": {
              "type": "string"
            }
          },
          "required": ["article"],
          "additionalProperties": False
        },
        "type": "json_schema" 
        }
      }
  )
  message = completion.output_text
  print("generated sequence of length = ", len("".join(message).split()))
  #print("message = ", message)

  return message
