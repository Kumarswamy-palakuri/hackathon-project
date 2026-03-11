import os
import json
import re
import io
import datetime
import traceback
import tempfile

from flask import (Flask, render_template, request, redirect, url_for,
                   flash, jsonify, send_file, session)
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import Config
from models import db, User, Analysis, ChatMessage

# ─────────────────────────────────────────────
#  App Setup
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'ACCESS DENIED — PLEASE AUTHENTICATE'
login_manager.login_message_category = 'error'


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────
#  Gemini AI Helper
# ─────────────────────────────────────────────
def get_groq_client():
    try:
        from groq import Groq
        api_key = app.config['GROQ_API_KEY']
        if not api_key:
            return None
        return Groq(api_key=api_key)
    except Exception:
        return None


def call_groq(prompt, client=None):
    if client is None:
        client = get_groq_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Groq error: {e}")
        return None


# ─────────────────────────────────────────────
#  Resume Text Extraction
# ─────────────────────────────────────────────
def extract_text_from_pdf(file_path):
    try:
        import PyPDF2
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""


def extract_text_from_docx(file_path):
    try:
        from docx import Document
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs]).strip()
    except Exception as e:
        print(f"DOCX extraction error: {e}")
        return ""


def extract_resume_text(file_path):
    ext = file_path.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ('doc', 'docx'):
        return extract_text_from_docx(file_path)
    return ""


# ─────────────────────────────────────────────
#  GitHub API Helper
# ─────────────────────────────────────────────
def fetch_github_data(github_url):
    import requests as req
    try:
        # Extract username from URL
        match = re.search(r'github\.com/([a-zA-Z0-9_-]+)', github_url)
        if not match:
            return {}
        username = match.group(1)
        headers = {'Accept': 'application/vnd.github.v3+json'}
        token = app.config.get('GITHUB_TOKEN', '')
        if token:
            headers['Authorization'] = f'token {token}'

        user_resp = req.get(f'https://api.github.com/users/{username}', headers=headers, timeout=10)
        if user_resp.status_code != 200:
            return {'username': username, 'error': 'Could not fetch profile'}
        user_data = user_resp.json()

        repos_resp = req.get(f'https://api.github.com/users/{username}/repos?per_page=50&sort=updated',
                             headers=headers, timeout=10)
        repos = repos_resp.json() if repos_resp.status_code == 200 else []

        # Aggregate language stats
        languages = {}
        for repo in repos[:20]:
            if repo.get('language'):
                lang = repo['language']
                languages[lang] = languages.get(lang, 0) + 1

        top_repos = [
            {'name': r.get('name'), 'description': r.get('description', ''),
             'stars': r.get('stargazers_count', 0), 'language': r.get('language', 'N/A'),
             'url': r.get('html_url', '')}
            for r in sorted(repos, key=lambda x: x.get('stargazers_count', 0), reverse=True)[:5]
        ]

        return {
            'username': username,
            'name': user_data.get('name', username),
            'bio': user_data.get('bio', ''),
            'public_repos': user_data.get('public_repos', 0),
            'followers': user_data.get('followers', 0),
            'following': user_data.get('following', 0),
            'location': user_data.get('location', ''),
            'company': user_data.get('company', ''),
            'top_languages': languages,
            'top_repos': top_repos,
            'total_stars': sum(r.get('stargazers_count', 0) for r in repos)
        }
    except Exception as e:
        print(f"GitHub fetch error: {e}")
        return {}


# ─────────────────────────────────────────────
#  LinkedIn Public Profile Scraper
# ─────────────────────────────────────────────
def fetch_linkedin_data(linkedin_url):
    import requests as req
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return {'url': linkedin_url, 'note': 'BeautifulSoup not installed — skipping LinkedIn scrape'}
    try:
        if not linkedin_url or 'linkedin.com' not in linkedin_url:
            return {}
        headers = {
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/120.0.0.0 Safari/537.36'),
            'Accept-Language': 'en-US,en;q=0.9'
        }
        resp = req.get(linkedin_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return {'url': linkedin_url, 'note': 'Profile requires login or is private'}
        soup = BeautifulSoup(resp.text, 'html.parser')

        name_tag = soup.find('h1')
        headline_tag = soup.find('h2')
        name = name_tag.get_text(strip=True) if name_tag else ''
        headline = headline_tag.get_text(strip=True) if headline_tag else ''

        # Extract text content for AI analysis
        about_section = soup.find('section', {'data-section': 'summary'})
        about = about_section.get_text(strip=True) if about_section else ''

        # Skills keywords extraction
        skills_tags = soup.find_all('span', class_=re.compile('skill'))
        skills = [s.get_text(strip=True) for s in skills_tags[:20]]

        return {
            'url': linkedin_url,
            'name': name,
            'headline': headline,
            'about': about[:500],
            'skills': skills,
            'raw_text': soup.get_text(separator=' ', strip=True)[:2000]
        }
    except Exception as e:
        print(f"LinkedIn fetch error: {e}")
        return {'url': linkedin_url, 'note': f'Could not scrape: {str(e)}'}


# ─────────────────────────────────────────────
#  NewsAPI Helper
# ─────────────────────────────────────────────
def fetch_job_news():
    import requests as req
    try:
        api_key = app.config.get('NEWS_API_KEY', '')
        if not api_key or api_key == 'your_news_api_key_here':
            return []
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': 'tech jobs hiring 2026 software developer',
            'sortBy': 'publishedAt',
            'pageSize': 5,
            'language': 'en',
            'apiKey': api_key
        }
        resp = req.get(url, params=params, timeout=8)
        if resp.status_code == 200:
            articles = resp.json().get('articles', [])
            return [{'title': a['title'], 'description': a.get('description', ''),
                     'url': a['url'], 'publishedAt': a['publishedAt']} for a in articles[:5]]
        return []
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []


# ─────────────────────────────────────────────
#  AI Analysis Engine
# ─────────────────────────────────────────────
def generate_fallback_analysis(resume_text, github_data, linkedin_data, job_role, personality_traits=None):
    """Return a structured mock analysis when AI is unavailable"""
    import random
    role_skills_map = {
        'frontend': [
            {'name': 'React/Vue/Angular', 'score': random.randint(60, 92), 'notes': 'Core framework proficiency'},
            {'name': 'HTML/CSS/Tailwind', 'score': random.randint(70, 95), 'notes': 'Styling expertise'},
            {'name': 'JavaScript/TypeScript', 'score': random.randint(65, 90), 'notes': 'Language mastery'},
            {'name': 'Performance Optimization', 'score': random.randint(55, 80), 'notes': 'Web vitals & optimization'}
        ],
        'backend': [
            {'name': 'Python/Node.js/Java', 'score': random.randint(65, 90), 'notes': 'Backend language skills'},
            {'name': 'REST API Design', 'score': random.randint(60, 88), 'notes': 'API architecture knowledge'},
            {'name': 'Database Management', 'score': random.randint(55, 85), 'notes': 'SQL & NoSQL proficiency'},
            {'name': 'System Architecture', 'score': random.randint(50, 78), 'notes': 'Scalability awareness'}
        ],
        'fullstack': [
            {'name': 'Frontend Frameworks', 'score': random.randint(65, 88), 'notes': 'UI development skills'},
            {'name': 'Backend + APIs', 'score': random.randint(60, 85), 'notes': 'Server-side development'},
            {'name': 'Database Systems', 'score': random.randint(55, 82), 'notes': 'Data persistence'},
            {'name': 'DevOps Basics', 'score': random.randint(45, 75), 'notes': 'Deployment knowledge'}
        ],
        'data': [
            {'name': 'Python/R', 'score': random.randint(70, 92), 'notes': 'Data science languages'},
            {'name': 'Machine Learning', 'score': random.randint(55, 85), 'notes': 'ML algorithms & frameworks'},
            {'name': 'SQL & Analytics', 'score': random.randint(65, 90), 'notes': 'Data querying skills'},
            {'name': 'Data Visualization', 'score': random.randint(60, 88), 'notes': 'Storytelling with data'}
        ],
        'devops': [
            {'name': 'Cloud (AWS/Azure/GCP)', 'score': random.randint(55, 88), 'notes': 'Cloud platform skills'},
            {'name': 'Docker & Kubernetes', 'score': random.randint(50, 82), 'notes': 'Container orchestration'},
            {'name': 'CI/CD Pipelines', 'score': random.randint(60, 85), 'notes': 'Automation expertise'},
            {'name': 'Infrastructure as Code', 'score': random.randint(45, 78), 'notes': 'Terraform/Ansible skills'}
        ],
        'mobile': [
            {'name': 'React Native/Flutter', 'score': random.randint(60, 88), 'notes': 'Cross-platform development'},
            {'name': 'iOS/Android Native', 'score': random.randint(50, 80), 'notes': 'Platform-specific skills'},
            {'name': 'Mobile UI/UX', 'score': random.randint(65, 90), 'notes': 'Mobile design patterns'},
            {'name': 'App Performance', 'score': random.randint(55, 82), 'notes': 'Optimization & profiling'}
        ]
    }
    tech_skills = role_skills_map.get(job_role, role_skills_map['fullstack'])
    soft_skills = [
        {'name': 'Communication', 'score': random.randint(70, 92)},
        {'name': 'Problem Solving', 'score': random.randint(72, 95)},
        {'name': 'Team Collaboration', 'score': random.randint(75, 93)},
        {'name': 'Adaptability', 'score': random.randint(68, 90)}
    ]
    avg_score = int(sum(s['score'] for s in tech_skills) / len(tech_skills))

    # Incorporate GitHub stats if available
    languages_info = ''
    if github_data.get('top_languages'):
        top_lang = max(github_data['top_languages'], key=github_data['top_languages'].get)
        languages_info = f" Your primary GitHub language appears to be {top_lang}."
    repos_count = github_data.get('public_repos', 0)

    return {
        'overall_score': avg_score,
        'readiness_level': 'EXCELLENT' if avg_score >= 80 else ('GOOD' if avg_score >= 60 else 'NEEDS IMPROVEMENT'),
        'technical_skills': tech_skills,
        'soft_skills': soft_skills,
        'improvements': [
            f'Add quantifiable metrics to your {job_role} experience (e.g., "Reduced load time by 35%")',
            f'Contribute to open-source {job_role} projects — you have {repos_count} public repos, aim for quality contributions' if repos_count else 'Create a GitHub profile and add your projects',
            'Obtain relevant cloud certification (AWS Certified Developer / Google Cloud Associate)',
            'Build a live portfolio showcasing 3-5 real-world projects with live demos',
            f'Study system design patterns for senior {job_role.replace("-", " ").title()} interviews' + languages_info,
            'Add a strong professional summary to the top of your resume (2-3 impactful sentences)'
        ],
        'course_recommendations': [
            {'title': f'{job_role.title()} Masterclass 2026', 'platform': 'Udemy',
             'level': 'INTERMEDIATE', 'url': 'https://www.udemy.com/', 'gap_addressed': 'General Skill Refresh', 'reason': 'Covers the latest industry practices'},
            {'title': 'System Design Interview Prep', 'platform': 'Educative',
             'level': 'ADVANCED', 'url': 'https://www.educative.io/', 'gap_addressed': 'System Architecture', 'reason': 'Critical for senior-level interviews'},
            {'title': 'AWS Cloud Practitioner', 'platform': 'AWS Training',
             'level': 'BEGINNER', 'url': 'https://aws.amazon.com/training/', 'gap_addressed': 'Cloud Foundations', 'reason': 'Cloud is essential for modern development'},
            {'title': 'Docker & Kubernetes Fundamentals', 'platform': 'Coursera',
             'level': 'INTERMEDIATE', 'url': 'https://www.coursera.org/', 'gap_addressed': 'Containerization', 'reason': 'DevOps skills boost employability significantly'}
        ],
        'resume_rewrites': [
            'Replace passive phrases: "Responsible for..." → "Led the development of..."',
            'Add impact numbers: "Built API" → "Built REST API handling 50K+ daily requests"',
            'Include technologies used in each bullet point for ATS keyword optimization'
        ],
        'job_readiness_summary': (
            f'Your profile shows a strong foundation for a {job_role.replace("-", " ").title()} role. '
            f'With {repos_count} public GitHub repositories demonstrating practical experience, '
            f'your technical profile is competitive. Focus on adding measurable impact metrics '
            f'to your resume and completing one cloud certification to significantly boost your job prospects.'
            if github_data else
            f'Your profile indicates solid {job_role.replace("-", " ").title()} skills. '
            f'Adding a GitHub profile and quantifying your achievements will considerably strengthen your candidacy. '
            f'The recommended courses above will help bridge the remaining skill gaps.'
        ),
        'ai_powered': False,
        'roadmap': [
            {'day': '1-30', 'focus': 'Skill Foundation', 'action': 'Complete top recommended course'},
            {'day': '31-60', 'focus': 'Portfolio Building', 'action': 'Build 2 projects using new skills'},
            {'day': '61-90', 'focus': 'Interview Prep', 'action': 'Mock interviews & apply for jobs'}
        ]
    }


def generate_ai_analysis(resume_text, github_data, linkedin_data, job_role, personality_traits=None):
    model = get_groq_client()
    if not model:
        return generate_fallback_analysis(resume_text, github_data, linkedin_data, job_role, personality_traits)

    github_summary = json.dumps(github_data, indent=2)[:1500] if github_data else "Not provided"
    linkedin_summary = json.dumps({
        'headline': linkedin_data.get('headline', ''),
        'about': linkedin_data.get('about', ''),
        'skills': linkedin_data.get('skills', []),
        'raw_text': linkedin_data.get('raw_text', '')[:800]
    }, indent=2) if linkedin_data else "Not provided"

    resume_snippet = (resume_text[:3000] + '...[truncated]') if len(resume_text) > 3000 else resume_text

    try:
        traits_display = json.dumps(json.loads(personality_traits), indent=2) if personality_traits else 'Not provided'
    except Exception:
        traits_display = str(personality_traits)

    # Extract detectable skills from resume text for better prompting
    detected_skills = []
    skill_keywords = ['React', 'Vue', 'Angular', 'Python', 'Node', 'Java', 'SQL', 'MongoDB',
                      'Docker', 'AWS', 'TypeScript', 'Flutter', 'Django', 'FastAPI', 'TensorFlow',
                      'PyTorch', 'Kubernetes', 'Git', 'Redis', 'GraphQL', 'Next.js', 'Express']
    for kw in skill_keywords:
        if resume_text and kw.lower() in resume_text.lower():
            detected_skills.append(kw)

    detected_github_langs = ', '.join(github_data.get('top_languages', {}).keys()) if github_data else 'unknown'

    prompt = f"""You are an expert career coach and technical recruiter with 15 years of experience.

Analyze this candidate's profile for a {job_role.replace('_', ' ').title()} role.

=== PERSONALITY TRAITS ===
{traits_display}

=== RESUME TEXT ===
{resume_snippet if resume_snippet else 'No resume provided. Analyze purely based on GitHub and LinkedIn.'}

=== GITHUB PROFILE DATA ===
{github_summary}
Detected GitHub Languages: {detected_github_langs}

=== LINKEDIN PROFILE DATA ===
{linkedin_summary}

=== TARGET ROLE ===
{job_role.replace('_', ' ').title()}

=== SKILLS DETECTED IN RESUME ===
Detected: {', '.join(detected_skills) if detected_skills else 'None clearly detected — infer from context'}

IMPORTANT RULES FOR COURSE RECOMMENDATIONS:
1. You MUST recommend courses that directly address SKILL GAPS detected from the resume, GitHub, and LinkedIn data above.
2. Each course must be SPECIFIC (real course name, exact platform, real enrollment URL).
3. Do NOT recommend courses for skills the candidate ALREADY HAS — focus ONLY on gaps.
4. Consider personality traits: e.g., for self-paced learners recommend Udemy/YouTube, for structured learners recommend Coursera/edX.
5. Prioritize high-ROI skills for the target {job_role} role in 2026.

Return ONLY valid JSON (no markdown, no backticks) in this exact format:
{{
  "overall_score": <integer 0-100>,
  "readiness_level": "<EXCELLENT|GOOD|NEEDS IMPROVEMENT>",
  "technical_skills": [
    {{"name": "<exact skill from profile>", "score": <0-100>, "notes": "<specific observation referencing resume/github>"}}
  ],
  "soft_skills": [
    {{"name": "<skill>", "score": <0-100>}}
  ],
  "improvements": [
    "<specific improvement that references actual content from the resume/GitHub/LinkedIn>"
  ],
  "course_recommendations": [
    {{
      "title": "<real, specific course title>",
      "platform": "<Udemy|Coursera|Pluralsight|YouTube|freeCodeCamp|edX|LinkedIn Learning|Zero To Mastery|Frontend Masters>",
      "level": "<BEGINNER|INTERMEDIATE|ADVANCED>",
      "url": "<direct enrollment link or search URL for this course>",
      "gap_addressed": "<which specific skill gap from the user's profile this course fills>",
      "reason": "<why this is critical for THIS user, referencing their specific detected skills/gaps>"
    }}
  ],
  "resume_rewrites": [
    "<quote the exact weak phrase from resume and show improved version with metrics>"
  ],
  "roadmap": [
    {{"day": "1-30", "focus": "<focus area title>", "action": "<specific daily/weekly actions>"}},
    {{"day": "31-60", "focus": "<focus area title>", "action": "<specific daily/weekly actions>"}},
    {{"day": "61-90", "focus": "<focus area title>", "action": "<specific daily/weekly actions>"}}
  ],
  "job_readiness_summary": "<2-3 sentences personalizing the summary to this user's actual profile, skills, and personality>"
}}

Include exactly 4 technical skills, 4 soft skills, 5-6 improvements, 4-5 course recommendations (addressing DIFFERENT gap areas), 3 resume rewrites, and a 3-phase roadmap.
Every single item MUST be SPECIFIC to the actual profile data provided — never use generic advice.
"""

    raw = call_groq(prompt, model)
    if not raw:
        return generate_fallback_analysis(resume_text, github_data, linkedin_data, job_role, personality_traits)

    # Clean and parse JSON
    try:
        # Strip markdown code fences if present
        cleaned = re.sub(r'```(?:json)?\s*', '', raw).strip()
        # Find JSON object
        start = cleaned.find('{')
        end = cleaned.rfind('}') + 1
        if start >= 0 and end > start:
            result = json.loads(cleaned[start:end])
            result['ai_powered'] = True
            return result
    except json.JSONDecodeError:
        pass

    return generate_fallback_analysis(resume_text, github_data, linkedin_data, job_role, personality_traits)


# ─────────────────────────────────────────────
#  Chat AI Engine
# ─────────────────────────────────────────────
def generate_chat_response(user_message, user_id, job_news):
    model = get_groq_client()

    # Fetch last 5 messages for context
    history = ChatMessage.query.filter_by(user_id=user_id).order_by(
        ChatMessage.timestamp.desc()).limit(5).all()
    history_text = "\n".join([
        f"User: {m.message}\nAssistant: {m.response}" for m in reversed(history)
    ]) if history else ""

    news_text = ""
    if job_news:
        news_text = "LATEST JOB MARKET NEWS:\n" + "\n".join([
            f"- {n['title']}: {n.get('description', '')[:150]}" for n in job_news
        ])

    if not model:
        # Fallback response
        lower = user_message.lower()
        responses = {
            'resume': ("🎯 **Resume Pro Tips:**\n"
                       "• Lead each bullet with a strong action verb (Built, Led, Designed, Optimized)\n"
                       "• Add metrics: 'Improved API response time by 40%'\n"
                       "• Keep it 1-2 pages, ATS-friendly format\n"
                       "• Tailor keywords for each job description\n\n"
                       "Want tips for a specific section? Just ask!"),
            'interview': ("💡 **Interview Mastery Framework:**\n"
                          "• **Technical:** Practice LeetCode (Easy→Medium), study system design\n"
                          "• **Behavioral:** Use STAR method (Situation, Task, Action, Result)\n"
                          "• **Company Research:** Know their tech stack, recent news, culture\n"
                          "• **Questions to Ask:** 'What does success look like in 90 days?'\n\n"
                          "Which type of interview are you preparing for?"),
            'salary': ("💰 **Salary Negotiation Strategy:**\n"
                       "• Research on Levels.fyi, Glassdoor, LinkedIn Salary\n"
                       "• Never give a number first — ask 'What's the budget for this role?'\n"
                       "• Consider total comp: base + bonus + equity + benefits\n"
                       "• Always negotiate — 90% of employers expect it!\n\n"
                       "What's your target role and location? I can give specific ranges."),
            'skill': ("🚀 **Most In-Demand Skills 2026:**\n"
                      "• **AI/ML Integration** — LLMs, RAG, Vector DBs\n"
                      "• **Cloud Native** — AWS, Kubernetes, Serverless\n"
                      "• **Full Stack** — React + Python/Node.js + PostgreSQL\n"
                      "• **DevOps** — CI/CD, Docker, Infrastructure as Code\n\n"
                      "Which skill area matches your background?"),
            'job': ("📰 **Job Market Update 2026:**\n"
                    "• Tech sector saw 15% growth in AI/ML roles\n"
                    "• Remote-first companies still hiring aggressively\n"
                    "• Cybersecurity and Cloud roles have the lowest unemployment\n"
                    "• Full-stack developers remain in extremely high demand\n\n"
                    "Looking for jobs in a specific domain?")
        }
        for key, resp in responses.items():
            if key in lower:
                return resp
        return ("🤖 **Career AI Coach at your service!**\n\n"
                "I can help you with:\n"
                "• ✅ Resume writing & optimization\n• ✅ Interview preparation\n"
                "• ✅ Salary negotiation\n• ✅ Skill gap analysis\n"
                "• ✅ Job market trends & news\n• ✅ Career path guidance\n\n"
                "What's your biggest career challenge right now?")

    try:
        user = db.session.get(User, user_id)
        traits = user.personality_traits if user else None
        traits_context = f"\nUSER PERSONALITY TRAITS: {traits}" if traits else ""
    except Exception:
        traits_context = ""

    prompt = f"""You are an expert AI Career Coach named ARIA (AI Resume Intelligence Assistant).
You specialize in helping tech professionals land their dream jobs, upgrade their resumes, and improve their skills.
You provide concrete paths, course recommendations, and career strategies.
You are encouraging, specific, and always back advice with real data.

{news_text}{traits_context}

CONVERSATION HISTORY:
{history_text}

USER QUERY: {user_message}

Respond in a helpful, engaging way. Use:
- Emoji for visual structure (not overdone)
- Bullet points for lists
- Bold for key terms using **text**
- Be SPECIFIC with the advice — no generic platitudes.
- If they ask for action plans, give them a concrete, day-by-day or week-by-week roadmap.
- If job news is relevant to their question, mention it.
- Keep response actionable.
- Always end with a follow-up question or call to action.

Never say you're an AI assistant — speak confidently as a career expert."""

    response = call_groq(prompt, model)
    return response or "I'm processing your request. Please try again in a moment."


# ─────────────────────────────────────────────
#  PDF Export Helper  
# ─────────────────────────────────────────────
def generate_pdf_report(analysis_data):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib import colors

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                rightMargin=0.75*inch, leftMargin=0.75*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Title'],
                                     fontSize=22, textColor=colors.HexColor('#00f3ff'),
                                     spaceAfter=6)
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'],
                                       fontSize=13, textColor=colors.HexColor('#bc13fe'),
                                       spaceAfter=4)
        body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                    fontSize=10, spaceAfter=3, leading=14)
        score_style = ParagraphStyle('Score', parent=styles['Normal'],
                                     fontSize=36, textColor=colors.HexColor('#39ff14'),
                                     alignment=1, spaceAfter=4)

        story = []

        # Title
        story.append(Paragraph("⚡ JobReady AI — Career Analysis Report", title_style))
        story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%B %d, %Y %H:%M')}", body_style))
        story.append(HRFlowable(color=colors.HexColor('#00f3ff'), thickness=1, spaceAfter=12))

        # Score
        score = analysis_data.get('overall_score', 0)
        story.append(Paragraph(f"{score}%", score_style))
        story.append(Paragraph(f"Job Readiness: {analysis_data.get('readiness_level', 'N/A')}",
                                ParagraphStyle('RL', parent=styles['Normal'], alignment=1,
                                               fontSize=12, textColor=colors.HexColor('#fff01f'), spaceAfter=8)))
        story.append(Spacer(1, 12))

        # Summary
        story.append(Paragraph("PROFILE SUMMARY", heading_style))
        story.append(Paragraph(analysis_data.get('job_readiness_summary', ''), body_style))
        story.append(Spacer(1, 10))

        # Technical Skills
        story.append(Paragraph("TECHNICAL SKILLS", heading_style))
        for skill in analysis_data.get('technical_skills', []):
            story.append(Paragraph(
                f"<b>{skill['name']}</b>: {skill['score']}% — {skill.get('notes', '')}", body_style))
        story.append(Spacer(1, 10))

        # Improvements
        story.append(Paragraph("RECOMMENDED IMPROVEMENTS", heading_style))
        for i, imp in enumerate(analysis_data.get('improvements', []), 1):
            story.append(Paragraph(f"{i}. {imp}", body_style))
        story.append(Spacer(1, 10))

        # Courses
        story.append(Paragraph("COURSE RECOMMENDATIONS", heading_style))
        for c in analysis_data.get('course_recommendations', []):
            story.append(Paragraph(
                f"<b>{c['title']}</b> ({c['platform']}) — {c['level']}: {c['reason']}", body_style))
        story.append(Spacer(1, 10))

        # Resume Rewrites
        story.append(Paragraph("RESUME IMPROVEMENT SUGGESTIONS", heading_style))
        for rw in analysis_data.get('resume_rewrites', []):
            story.append(Paragraph(f"• {rw}", body_style))

        story.append(Spacer(1, 16))
        story.append(HRFlowable(color=colors.HexColor('#bc13fe'), thickness=1, spaceAfter=6))
        story.append(Paragraph("🤖 Generated by JobReady AI — Neural Career Intelligence Platform",
                                ParagraphStyle('Footer', parent=styles['Normal'],
                                               alignment=1, fontSize=8,
                                               textColor=colors.HexColor('#666666'))))

        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"PDF generation error: {e}")
        traceback.print_exc()
        return None


# ═════════════════════════════════════════════
#  AUTH ROUTES
# ═════════════════════════════════════════════
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user, remember=True)
            return redirect(url_for('index'))
        flash('Invalid email or password. Please try again.', 'error')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')

        if not name or not email or not password:
            flash('All fields are required.', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
        elif password != confirm:
            flash('Passwords do not match.', 'error')
        elif User.query.filter_by(email=email).first():
            flash('An account with this email already exists.', 'error')
        else:
            user = User(name=name, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# ═════════════════════════════════════════════
#  MAIN PAGE
# ═════════════════════════════════════════════
@app.route('/')
@login_required
def index():
    return render_template('index.html', user=current_user)


# ═════════════════════════════════════════════
#  API: ANALYZE
# ═════════════════════════════════════════════
@app.route('/api/analyze', methods=['POST'])
@login_required
def analyze():
    try:
        resume_text = ""
        resume_filename = None

        # Handle file upload
        if 'resume' in request.files and request.files['resume'].filename:
            file = request.files['resume']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                resume_text = extract_resume_text(filepath)
                resume_filename = filename

        github_url = request.form.get('github_url', '').strip()
        linkedin_url = request.form.get('linkedin_url', '').strip()
        job_role = request.form.get('job_role', 'fullstack').strip()

        if not resume_text and not github_url and not linkedin_url:
            return jsonify({'error': 'Please provide a resume or at least one profile URL'}), 400

        # Fetch external data
        github_data = fetch_github_data(github_url) if github_url else {}
        linkedin_data = fetch_linkedin_data(linkedin_url) if linkedin_url else {}

        # Generate AI analysis
        results = generate_ai_analysis(resume_text, github_data, linkedin_data, job_role, current_user.personality_traits)

        # Save to database
        analysis = Analysis(
            user_id=current_user.id,
            resume_filename=resume_filename,
            resume_text=resume_text[:5000] if resume_text else None,
            github_url=github_url or None,
            linkedin_url=linkedin_url or None,
            job_role=job_role,
            overall_score=results.get('overall_score', 0),
            results_json=json.dumps({**results, 'github_data': github_data})
        )
        db.session.add(analysis)
        db.session.commit()

        return jsonify({
            'success': True,
            'analysis_id': analysis.id,
            'results': results,
            'github_data': github_data
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ═════════════════════════════════════════════
#  API: CHAT
# ═════════════════════════════════════════════
@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        job_news = fetch_job_news()
        response_text = generate_chat_response(user_message, current_user.id, job_news)

        # Save to database
        chat_msg = ChatMessage(
            user_id=current_user.id,
            message=user_message,
            response=response_text
        )
        db.session.add(chat_msg)
        db.session.commit()

        return jsonify({
            'success': True,
            'response': response_text,
            'news': job_news[:3]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ═════════════════════════════════════════════
#  API: PERSONALITY TEST
# ═════════════════════════════════════════════
@app.route('/api/personality-test', methods=['POST'])
@login_required
def save_personality():
    try:
        data = request.get_json()
        traits = data.get('traits')
        if not traits:
            return jsonify({'error': 'No traits provided'}), 400
        
        current_user.personality_traits = json.dumps(traits)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ═════════════════════════════════════════════
#  API: GITHUB DATA PREVIEW
# ═════════════════════════════════════════════
@app.route('/api/github')
@login_required
def github_preview():
    url = request.args.get('url', '')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    data = fetch_github_data(url)
    return jsonify(data)


# ═════════════════════════════════════════════
#  API: HISTORY
# ═════════════════════════════════════════════
@app.route('/api/history')
@login_required
def get_history():
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(
        Analysis.timestamp.desc()).all()
    return jsonify([a.to_dict() for a in analyses])


@app.route('/api/history/<int:analysis_id>')
@login_required
def get_analysis(analysis_id):
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first_or_404()
    return jsonify(analysis.to_dict())


@app.route('/api/history/<int:analysis_id>', methods=['DELETE'])
@login_required
def delete_analysis(analysis_id):
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first_or_404()
    db.session.delete(analysis)
    db.session.commit()
    return jsonify({'success': True})


# ═════════════════════════════════════════════
#  API: EXPORT
# ═════════════════════════════════════════════
@app.route('/api/export/<int:analysis_id>')
@login_required
def export_analysis(analysis_id):
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first_or_404()
    fmt = request.args.get('format', 'json')
    results = json.loads(analysis.results_json) if analysis.results_json else {}

    if fmt == 'pdf':
        pdf_buffer = generate_pdf_report(results)
        if pdf_buffer:
            filename = f"JobReady_Analysis_{analysis.timestamp.strftime('%Y%m%d_%H%M%S')}.pdf"
            return send_file(pdf_buffer, mimetype='application/pdf',
                             as_attachment=True, download_name=filename)
        # fallback to JSON
        fmt = 'json'

    if fmt == 'json':
        export_data = {
            'export_date': datetime.datetime.now().isoformat(),
            'generated_by': 'JobReady AI',
            'user': current_user.name,
            'analysis': analysis.to_dict()
        }
        filename = f"JobReady_Analysis_{analysis.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        buffer = io.BytesIO(json.dumps(export_data, indent=2).encode())
        return send_file(buffer, mimetype='application/json',
                         as_attachment=True, download_name=filename)

    return jsonify({'error': 'Invalid format'}), 400


# ═════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("Database initialized")
        print("Starting JobReady AI Backend...")
        print("Server: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
