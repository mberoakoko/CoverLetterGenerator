# Cover Letter Generator

CoverLetterGeneratorA powerful, AI-driven tool designed to assist job seekers in creating highly tailored, professional, and impactful cover letters.
This project leverages advanced Large Language Models (LLMs) to streamline the cover letter writing process, ensuring your application stands out.<br>
✨ FeaturesMulti-Context Integration: Ingests diverse input documents including:Job DescriptionOriginal Cover Letter (for reference and improvement)Internship CertificatesRecommendation Letters (First and Second)Intelligent Context Refinement: A primary LLM processes all input contexts, extracting and synthesizing the most relevant information into a structured format.<br>
This refined context highlights key skills, achievements, and unique selling points, ensuring perfect alignment with the target job.Advanced Cover Letter Generation: A secondary, specialized LLM utilizes the refined context to draft a high-standard cover letter. It adheres to modern professional communication standards, with a particular focus on requirements for Fortune 500 and German companies (e.g., conciseness, quantification, formal tone, company fit).<br>
Sentence Structure Enhancement: An additional refinement step is applied to improve sentence diversity, avoid repetitive pronoun-led sentences, and strategically balance active and passive voice for optimal readability and impact.<br><br>🚀 InstallationTo get started with the CoverLetterGenerator, follow these steps:Clone the repository:git clone https://github.com/mberoakoko/CoverLetterGenerator.git
cd CoverLetterGenerator
Create a virtual environment (recommended):python -m venv venv
### On Windows:
#### venv\Scripts\activate
### On macOS/Linux:
#### source venv/bin/activate
Install the required dependencies:pip install -r requirements.txt
(Note: Ensure you have your LLM API keys configured as environment variables or within a configuration file as per the project's internal setup.)💡 UsageOnce installed, you can run the CoverLetterGenerator by executing the main script and providing your input documents.python main.py --job_description_path path/to/job_description.txt \
               --original_cover_letter_path path/to/original_cover_letter.txt \
               --internship_certificate_path path/to/internship_certificate.txt \
               --recommendation_letter_1_path path/to/rec_letter_1.txt \
               --recommendation_letter_2_path path/to/rec_letter_2.txt \
<br>
The generated cover letter will be outputted to the console or saved to a specified file.

<br>🤝 ContributingContributions are welcome! Please feel free to open issues or submit pull requests.
<br>📄 LicenseThis project is licensed under the Apache2.0.