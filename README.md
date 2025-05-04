# Job Agent
## Using LLMs to determine job posting relevnce 

Simple AI agent to review job postings:
- Pass job posting information sourced from LinkedIn to the LLM
    - Parsed to extract jop title, description, location, etc (`scripts/get_jobs.py`)
    - Data stored as JSON (`data/jobs.jspon`)
- LLM agent evaluates exh job description, determines relevance for a user based on a resume
    - User supplied resume provides information about prior erxperiences, skills, etc
    - Agent provides a relevance score, along with a summary of why the score is assigned
    - Results are saved as a CSV
