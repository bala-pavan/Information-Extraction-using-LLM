Write a Python script that takes patient data in the form of free text and extracts the following
properties using an LLM: name, gender, age, weight, height, BMI, and chief medical
complaint. For each execution, the script must generate a Python dictionary containing
exactly those seven fields.
- Even without access to a GPU, models like phi3 should run on a local laptop. It will be
slow, but the task does not require a lot of output tokens.
- Resources for LLM execution: HuggingFace in server mode
- Three example patients you should use are shown
