from langchain_huggingface import HuggingFaceEndpoint
import os
from langchain import PromptTemplate, LLMChain

# Set HuggingFace API token (replace `your_hf_token` with your actual token or retrieve securely)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
repo_id = "microsoft/Phi-3-mini-4k-instruct"

# Initialize the LLM
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=512,  # Adjusted max length for input/output
    temperature=0.7,
)

# Define the prompt template
template = """
Extract the following fields from the given patient data: name, gender, age, weight, height, BMI, and chief medical complaint.
Provide the output as a JSON object with these exact keys: name, gender, age, weight, height, BMI, chief_medical_complaint.

Patient data: {patient_text}
"""
prompt = PromptTemplate(template=template, input_variables=["patient_text"])

# Initialize the LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Example patient data
patient_data = """John R. Whitaker, a 52-year-old male, stands 5'10" (70 inches) tall and weighs 198 lbs. 
Mr. Whitaker has a history of hypertension and type 2 diabetes, both diagnosed in his mid-40s, and recently began experiencing 
worsening peripheral neuropathy in his lower extremities. He also reports chronic lower back pain, which he attributes to years 
of heavy lifting in his previous occupation as a construction worker. Over the past six months, John has developed shortness of breath 
during mild exertion, prompting concerns about potential early-stage congestive heart failure. Additionally, he struggles with obesity-related 
sleep apnea, contributing to fatigue and cognitive fog throughout the day. Despite his conditions, Mr. Whitaker maintains a generally 
positive outlook but admits to inconsistent medication adherence and difficulty following a healthy diet."""

# Run the LLM chain
response = llm_chain.run(patient_text=patient_data)

# Print the output
print(response)
