from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from operator import itemgetter
from dotenv import load_dotenv
import argparse
load_dotenv()

parser = argparse.ArgumentParser()

parser.add_argument("--task", default="Return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()


code_prompt = PromptTemplate(
    template = "Write a very short {language} function that will {task}",
    input_variables=["language", "task"]

)

test_prompt = PromptTemplate(
    template = "Write a test for the following {language} code\n{code}",
    input_variables=["language", "code"]
)

llm = OpenAI()

code_chain = code_prompt | llm | StrOutputParser()

test_chain = test_prompt | llm | StrOutputParser()


complete_chain = ({
    "code": code_chain,
    "language": itemgetter("language"),
    }
    | RunnablePassthrough.assign(test=test_chain)
)

result = complete_chain.invoke({
    "language": args.language,
    "task": args.task
})
 

print(result['code'])
print(result['language'])
print(result['test'])
