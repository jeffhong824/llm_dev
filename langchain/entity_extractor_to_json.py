"""
Module Description:

This module is designed to identify specified entities within a given sentence and output the results in JSON format. 
It supports command-line arguments for flexible configuration and execution.

Features:
- Identify entities in an input sentence based on a user-defined list.
- Save extracted entities and their values in a structured JSON format.
- Provide usage examples and argument descriptions for easy execution.

Usage:
To execute the module:
    python entity_extractor_to_json.py --text "<input_sentence>" --entities <entity1> <entity2> ...

Arguments:
- text (str): The input sentence where entities will be identified.
- entities (List[str]): List of entities to be extracted from the sentence.

Example:
    python entity_extractor_to_json.py --text "This is a sample sentence" --entities Bacteria Virus Disease

Output: The extracted entities with their values in a JSON file.

Execution Flow:
1. Define and parse command-line arguments.
2. Implement main logic to extract entities based on input arguments.
3. Output results in JSON format.

Author: Ting-Chun Hung
Version: 1.0.0
Date: 2024-11-11
"""


from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import os
import argparse
import json
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys and model information
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')
TEXT_GENERATION_MODEL = os.getenv('TEXT_GENERATION_MODEL')

def create_response_schemas(entity_list: List[str]) -> List[ResponseSchema]:
    """Generate response schemas based on the list of entities."""
    return [ResponseSchema(name=entity, description=f"識別句子中{entity}實體") for entity in entity_list]

def run_llm_chain(text: str, entities: List[str]) -> Dict:
    """Run the LLM chain to identify specified entities in the text and return structured JSON output."""
    # Create prompt template based on entity list
    ner_prompt = "請從以下句子中識別出以下實體: " + ", ".join(entities) + "\n句子: {input_sentence}\n請以JSON格式輸出結果。"
    prompt_template = PromptTemplate.from_template(ner_prompt)
    
    # Initialize LLM with Azure configuration
    # llm = ChatOpenAI(
    #     openai_api_key=,
    #     model_name=TEXT_GENERATION_MODEL,
    #     temperature=0.8
    # )
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=TEXT_GENERATION_MODEL,
        api_version="2024-05-01-preview",
        temperature=0.8
    )

    chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
    
    # Run chain with provided text
    res = chain.run({"input_sentence": text})
    print("Model Response:", res)  # Debugging line to observe the model's raw response
    
    # Attempt to parse as JSON, with fallback error handling
    try:
        response_schemas = create_response_schemas(entities)
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        res_dict = output_parser.parse(res)
    except json.JSONDecodeError as e:
        print("JSON Decode Error:", e)
        res_dict = {"error": "Invalid JSON format from model output", "raw_output": res}
    return res_dict

def save_to_json(data: Dict, filename: str) -> None:
    """Save the data to a JSON file."""
    with open(filename, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="A script to identify specified entities in a text.")
    parser.add_argument("--text", "-t", type=str, required=True, help="Input text for entity recognition")
    parser.add_argument("--entities", "-e", type=str, nargs='+', required=True, help="List of entities to recognize")
    parser.add_argument("--output", "-o", type=str, default="output.json", help="Output JSON file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Run the entity recognition
    results = run_llm_chain(args.text, args.entities)
    
    # Save results to JSON file
    save_to_json(results, args.output)
    print(f"Results saved to {args.output}")


# python entity_extractor_to_json.py --text "這是一個測試句子，涉及疾病、病毒和細菌的資訊。" --entities Virus Disease Bacteria --output output_multiple.json

