from openai import OpenAI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class JobSummary(BaseModel):
    """Model to represent the job summary response from the LLM."""
    job_title: str
    score: int
    company: str
    summary: str

class LLMService:
    """Service to interact with OpenAI LLMs."""

    def __init__(self, model: str):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = model

    def get_model_name(self) -> str:
        """Return the name of the model."""
        return self.model

    async def perform_task(self, job_description: str, 
                           resume: str, 
                           response_model: type, 
                           max_tokens: int = 5000) -> JobSummary:
        """Send a resume, job description, and prompt to the LLM and return structured output."""
        
        # Construct the prompt, using the user provided resume and job description
        prompt = f"""is this job description a good fit? 
    Use my resume to identify skills and experiemces that are relevant to the job description. 
    Provide a score between 1 and 5 with 5 being the best fit, and a sort explinationf why the score waselected.
    Job Description: {job_description}
    Resume: {resume}"""
        
        
        messages = [
        {"role": "developer", "content": """you are a helpful but critical assistant that reviews job descriptons, identifying those that are most relevant 
        based on the skills and experiences in a user provided resume. For each job desription, provide a score between 1 and 5, with 5 being the best fit. 
        Provide a brief summary of why the score was selected, including any relevant skills and experiences the resume contains or is missing in the job description. 
        The user can only apply to a limited number of jobs, so give lower scores whenever there is not a perfect fit 
        Jobs that require some skills the user does not have should be rated a 3, and jobs that are not a good fit are rated a 2 or lower. 
        Treat generative AI and LLMs as a skill seperate from ML. Focus on the domain of the job (eg agriculture, logistics. finance, etc), comparing to domains the user discusses in their resume.
        If the user has not worked in the domain of the job, or if the job has no domain listed, give a lower score. 
        """},
        {"role": "user", "content": prompt}
        ]

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            response_format=response_model
        )
        return response.choices[0].message.parsed
    
async def evaluate_job(llm_agent, job_description: dict, resume: str) -> dict:
    """Main function to run the LLM service."""

    # convert the job description to a JSON string
    job_str = json.dumps(job_description)

    result = await llm_agent.perform_task(job_str, resume, response_model=JobSummary)
    return result.__dict__

if __name__ == "__main__":
    # Evaluate relevance of jobs from LinkedIn using an LLM to compare job descriptions to a resume
    # and provide a score and summary of the fit

    import json
    import pandas as pd

    # helper functions to read data files
    def read_json_file(file_path) -> dict:
        """Reads a JSON file and returns the data as a Python dictionary."""
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def read_text_file(file_path) -> str:
        """Reads a text file and returns the content as a string."""
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    
    # Load the job descriptions and resume
    jobs = read_json_file("data/jobs.json")
    resume = read_text_file("data/shriver_resume.txt")

    llm_service = LLMService(model="gpt-4o-mini")

    job_scores = []

    for j in jobs:
        # evaluate each job, append results to a list
        result = evaluate_job(llm_service, j, resume)
        job_scores.append(result)

    df = pd.DataFrame(job_scores)
    df.to_csv("data/job_scores.csv", index=False)
    print("Job scores saved to data/job_scores.csv")