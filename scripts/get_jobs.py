### LinkedIn Job Scraper
# This script scrapes job postings from LinkedIn using BeautifulSoup and requests.
# It retrieves job URLs from a search page and then scrapes details from each job posting.
# Code heavily inspired by tutorial here: https://brightdata.com/blog/how-tos/linkedin-scraping-guide

from bs4 import BeautifulSoup
import requests
import json
import time

def retrieve_job_urls(job_search_url):
    # Make an HTTP GET request to get the HTML of the page
    response = requests.get(job_search_url)

    # Access the HTML and parse it
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    # Where to store the scraped data
    job_urls = []

    # Scraping logic
    job_url_elements = soup.select("[data-tracking-control-name=\"public_jobs_jserp-result_search-card\"]")
    for job_url_element in job_url_elements:
      # Extract the job page URL and append it to the list
      job_url = job_url_element["href"]
      job_urls.append(job_url)

    return job_urls

def scrape_job(job_url):
    # Send an HTTP GET request to fetch the page HTML
    response = requests.get(job_url)

    # Access the HTML text from the response and parse it
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    # Scraping logic
    title_element = soup.select_one("h1")
    title = title_element.get_text().strip()

    company_element = soup.select_one("[data-tracking-control-name=\"public_jobs_topcard-org-name\"]")
    company_name = company_element.get_text().strip()
    company_url = company_element["href"]

    location_element = soup.select_one(".topcard__flavor--bullet")
    location = location_element.get_text().strip()

    applicants_element = soup.select_one(".num-applicants__caption")
    applicants = applicants_element.get_text().strip()

    salary_element = soup.select_one(".salary")
    if salary_element is not None:
      salary = salary_element.get_text().strip()
    else:
       salary = None

    description_element = soup.select_one(".description__text .show-more-less-html")
    description = description_element.get_text().strip()

    criteria = []
    criteria_elements = soup.select(".description__job-criteria-list li")
    for criteria_element in criteria_elements:
        name_element = criteria_element.select_one(".description__job-criteria-subheader")
        name = name_element.get_text().strip()

        value_element = criteria_element.select_one(".description__job-criteria-text")
        value = value_element.get_text().strip()

        criteria.append({
            "name": name,
            "value": value
        })

    # Collect the scraped data and return it
    job = {
        "url": job_url,
        "title": title,
        "company": {
            "name": company_name,
            "url": company_url
        },
        "location": location,
        "applications": applicants,
        "salary": salary,
        "description": description,
        "criteria": criteria
    }
    return job

# The public URL of the LinkedIn Jobs search page
public_job_search_url = "https://www.linkedin.com/jobs/search?keywords=Director%20Of%20Data%20Science&location=United%20States&geoId=103644278&f_TPR=r604800&f_WT=2&position=1&pageNum=0"
print("Starting job retrieval from LinkedIn search URL...")

# Retrieving the single URLs for each job on the page
job_urls = retrieve_job_urls(public_job_search_url)

print(f"Retrieved {len(job_urls)} job URLs\n")

# Scrape data from each job position page
jobs = []
for job_url in job_urls:

    try:
        job = scrape_job(job_url)
        jobs.append(job)
    
    except:
       continue

    # pause to give the server a break
    if len(jobs) % 10 == 0:
        time.sleep(5)

print(f"\nExporting {len(jobs)} scraped jobs to JSON")

file_name = "jobs.json"
with open(file_name, "w", encoding="utf-8") as file:
    json.dump(jobs, file, indent=4, ensure_ascii=False)

print(f"Jobs successfully saved to \"{file_name}\"\n")


# Export the scraped data to CSV
