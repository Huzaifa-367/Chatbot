from __future__ import annotations
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from typing import Any
import openai
import os

# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
prompt_file = "prompt_template.txt"


class ProductDescGen(LLMChain):
    """LLM Chain specifically for generating multi-paragraph rich text product description using emojis."""

    @classmethod
    def from_llm(
        cls, llm: BaseLanguageModel, prompt: str, **kwargs: Any
    ) -> ProductDescGen:
        """Load ProductDescGen Chain from LLM."""
        return cls(llm=llm, prompt=prompt, **kwargs)


def product_desc_generator(product_name, keywords, openai_api_key):
    with open(prompt_file, "r") as file:
        prompt_template = file.read()

    PROMPT = PromptTemplate(
        input_variables=["product_name", "keywords"], template=prompt_template
    )
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        # openai_api_key=OPENAI_API_KEY,
        openai_api_key=openai_api_key,
    )

    ProductDescGen_chain = ProductDescGen.from_llm(llm=llm, prompt=PROMPT)
    ProductDescGen_query = ProductDescGen_chain.apply_and_parse(
        [{"product_name": product_name, "keywords": keywords}]
    )
    return ProductDescGen_query[0]["text"]

def main():
    st.title("Product Description Generator")
    st.write(
        "Generate multi-paragraph rich text product descriptions for your products instantly!"
        " Provide the product name and keywords related to the product."
    )
    openai_api_key = st.text_input("OpenAI API Key", "your_openai_api_key_here")
    product_name = st.text_input("Product Name", "Nike Shoes")
    keywords = st.text_input(
        "Keywords (separated by commas)",
        "black shoes, leather shoes for men, water resistant"
    )

    if st.button("Generate Description"):
        if openai_api_key:
            description = product_desc_generator(product_name, keywords, openai_api_key)
            st.subheader("Product Description:")
            st.text(description)
        else:
            st.warning("Please provide your OpenAI API Key.")

if __name__ == "__main__":
    main()
