import streamlit as st
import re
from io import BytesIO
from fpdf import FPDF
from PyPDF2 import PdfReader
import spacy
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv
import en_core_web_sm



# Load .env file
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))



# Load SpaCy model
nlp = en_core_web_sm.load()

# ---- Helpers ----
def clean_split_skills(text):
    
    if text is None:
        return set()
    text = text.lower()
    parts = re.split(r'[,\n;]+', text)
    parts = [p.strip() for p in parts if p.strip()]
    return set(parts)

def extract_text_from_pdf_bytes(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))
    full = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        full.append(txt)
    return "\n".join(full)

def parse_uploaded_resume(uploaded_file):
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    content = uploaded_file.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(content)
    elif name.endswith(".txt"):
        try:
            return content.decode("utf-8", errors="ignore")
        except:
            return str(content)
    else:
        return ""


       # ---- ATS CV ANALYSIS OPENAI AND PROMPT ----


def analyze_cv_with_gemini(cv_text):
    
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")

        prompt = f"""
        You are an ATS CV expert and career advisor. Analyze the following resume and return a JSON object with:
        - score: integer 0–100 (ATS optimization score)
        - highlights: 1–3 short and brief positive points about the resume
        - improvements: 3 short and brief suggestions to improve ATS matching
        - matching_jobs: 3 short job titles that best fit this candidate’s skills and background (e.g., "Data Analyst", "Python Developer", "Machine Learning Engineer")

        Return **only** valid JSON in this structure:
        {{
            "score": 0–100,
            "highlights": ["...", "..."],
            "improvements": ["...", "...", "..."],
            "matching_jobs": ["...", "...", "..."]
        }}

        Resume Text:
        \"\"\"{cv_text[:4000]}\"\"\"
        """

        response = model.generate_content(prompt)
        text = response.text.strip()

        # Try to extract JSON
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end+1]

        result = json.loads(text)
        return result

    except Exception as e:
        return {"error": f"Gemini request failed: {str(e)}"}





# ----  PDF Creator ----
def create_pdf(data):
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 5, data.get("name", "").upper(), ln=True, align="C")
    pdf.ln(2)

    # --- Contact Info ---
    pdf.set_font("Helvetica", "", 12)
    contact = []
    if data.get("email"): contact.append("Email: " + data["email"])
    if data.get("linkedin"): contact.append("LinkedIn: " + data["linkedin"])
    if data.get("github"): contact.append("GitHub: " + data["github"])
    if data.get("phone"): contact.append("Phone: " + data["phone"])
    if data.get("location"): contact.append("Location: " + data["location"])
    if contact:
        pdf.multi_cell(0, 10, " | ".join(contact), align="C")
        pdf.ln(10)

    def add_section(title, text):
        if not text:
            return
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(0, 8, text)
        pdf.ln(6)

    # --- Profile Summary ---
    summary_text = data.get("summary", "")
    if summary_text:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Profile Summary", ln=True)
        line_y = pdf.get_y()
        pdf.set_draw_color(0, 0, 0)  
        pdf.line(10, line_y, 200, line_y)  
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(0, 8, summary_text)
        pdf.ln(6)


        # --- Experience ---
    experience_data = data.get("experience", [])
    if experience_data:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Experience", ln=True)
        line_y = pdf.get_y()
        pdf.set_draw_color(0, 0, 0)  
        pdf.line(10, line_y, 200, line_y)  
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 12)
        for exp in experience_data:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, f"{exp['title']}", ln=0)
            pdf.set_font("Helvetica", "I", 12)
            pdf.cell(0, 6, f"({exp['start_date']} - {exp['end_date']})", align="R", ln=True)
            pdf.set_font("Helvetica", "", 12)
            pdf.cell(0, 8, f"{exp['company']}", ln=True)
            if exp.get("description"):
                pdf.multi_cell(0, 6, exp["description"])
            pdf.ln(4)

    # --- Education ---
    education_list = data.get("education", [])
    if education_list:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Education", ln=True)
        line_y = pdf.get_y()
        pdf.set_draw_color(0, 0, 0)  
        pdf.line(10, line_y, 200, line_y)  
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 12)
        for edu in education_list:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, f"{edu['degree']}", ln=0)
            pdf.set_font("Helvetica", "I", 12)
            pdf.cell(0, 8, f"({edu['start_date']} - {edu['end_date']})", align="R", ln=True)
            pdf.set_font("Helvetica", "", 12)
            pdf.cell(0, 6, f"{edu['university']}", ln=True)
            if edu.get("cgpa"):
                pdf.cell(0, 6, f"CGPA/Percentage: {edu['cgpa']}", ln=True)
            if edu.get("description"):
                pdf.multi_cell(0, 6, edu["description"])
            pdf.ln(4)

    # --- Skills ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Skills", ln=True)

    line_y = pdf.get_y()
    pdf.set_draw_color(0, 0, 0)  
    pdf.line(10, line_y, 200, line_y)
    pdf.ln(2)  

    skills = data.get("skills", [])
    if skills:
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(0, 8, ", ".join(skills))
    pdf.ln(4)


    # --- Projects ---
    projects_list = data.get("projects", [])
    if projects_list:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Projects", ln=True)
        line_y = pdf.get_y()
        pdf.set_draw_color(0, 0, 0)  # black color
        pdf.line(10, line_y, 200, line_y)  # (x1, y1, x2, y2)
        pdf.ln(2)

        for proj in projects_list:
            # Title and Dates
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, f"{proj['title']}", ln=False)
            pdf.set_font("Helvetica", "I", 11)
            pdf.cell(0, 8, f"   {proj['start_date']} - {proj['end_date']}",align="R", ln=True)

            # Company and Budget
            pdf.set_font("Helvetica", "", 12)
            if proj.get("company"):
                pdf.cell(0, 6, f"Company: {proj['company']}", ln=True)
            if proj.get("budget"):
                pdf.cell(0, 6, f"Budget: {proj['budget']}", ln=True)

            # Description
            if proj.get("description"):
                pdf.multi_cell(0, 6, proj['description'])

            pdf.ln(4)

    # --- Extra Courses & Certificates ---
    certificates_list = data.get("certificates", [])
    if certificates_list:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Extra Courses & Certificates", ln=True)
        line_y = pdf.get_y()
        pdf.set_draw_color(0, 0, 0)
        pdf.line(10, line_y, 200, line_y)
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 12)
        for cert in certificates_list:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, f"{cert['title']}", ln=0)
            pdf.set_font("Helvetica", "I", 12)
            pdf.cell(0, 8, f"({cert['date']})", align="R", ln=True)
            pdf.set_font("Helvetica", "", 12)
            pdf.cell(0, 4, f"{cert['organization']}", ln=True)
            pdf.ln(4)

        
    # --- Additional Information ---
    if data.get("languages") or data.get("interests"):
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Additional Information", ln=True)
        line_y = pdf.get_y()
        pdf.set_draw_color(0, 0, 0)  # black color
        pdf.line(10, line_y, 200, line_y)  # (x1, y1, x2, y2)
        pdf.ln(2)

        if data.get("languages"):
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(25, 6, "Languages:", ln=False)
            pdf.set_font("Helvetica", "", 12)
            pdf.cell(0, 6, ", ".join(data["languages"]), ln=True)
        pdf.ln(1)
        if data.get("interests"):
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(20, 6, "Interests:", ln=False)
            pdf.set_font("Helvetica", "", 12)
            pdf.cell(0, 6, ", ".join(data["interests"]), ln=True)

        pdf.ln(4)

    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return pdf_bytes

# ---- Streamlit App ----
st.set_page_config(page_title="AI Resume Builder", layout="centered")
st.title("AI Resume Builder & Job Matching")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Resume Builder"):
        st.session_state['mode'] = 'builder'
with col2:
    if st.button("Upload Your Resume & View Matching Jobs"):
        st.session_state['mode'] = 'upload'

mode = st.session_state.get('mode', 'builder')

# -------------------- Resume Builder --------------------
if mode == 'builder':
    st.header("📝 Resume Builder")
    tabs = st.tabs(["Personal Information", "Experience", "Education Details", "Skills", "Projects", "Courses & Certificates", "Additional Information"])


    personal = {}

    with tabs[0]:
        personal['name'] = st.text_input("Full Name", placeholder="Full Name")
        personal['title'] = st.text_input("Professional Title", placeholder="Python Developer")
        personal['email'] = st.text_input("Email", placeholder="abc@gmail.com")
        personal['linkedin'] = st.text_input("LinkedIn ID", placeholder="linkedin.com/in/username")
        personal['github'] = st.text_input("GitHub ID", placeholder="github.com/username")
        personal['phone'] = st.text_input("Phone")
        personal['location'] = st.text_input("Location (City, Country)")
        personal['summary'] = st.text_area("Short Professional Summary", height=120, placeholder="A short summary about you...")

            
            # ---- SKILLS TAB -------
    with tabs[3]:
        
        if "skills" not in st.session_state:
            st.session_state["skills"] = []

        skill_input_key = "skill_input"

        def add_skill_callback():
            new_skill = st.session_state.get(skill_input_key, "").strip()
            if new_skill:
                st.session_state["skills"].append(new_skill)
                # clear the input box
                st.session_state[skill_input_key] = ""
                st.session_state["_last_added"] = new_skill

        # text input and button with callback
        st.text_input("Add a Skill", placeholder="e.g. Python", key=skill_input_key)
        st.button("Add Skill", on_click=add_skill_callback)

        # display current skills
        if st.session_state["skills"]:
            st.write("### Added Skills:")
            st.write(", ".join(st.session_state["skills"]))


        # ---- PROJECTS TAB ---- ----
    with tabs[4]:
        st.subheader("Add Your Projects")

        if "projects" not in st.session_state:
            st.session_state["projects"] = []

        # unique keys for each input field
        proj_title_key = "proj_title"
        proj_start_key = "proj_start"
        proj_end_key = "proj_end"
        proj_company_key = "proj_company"
        proj_budget_key = "proj_budget"
        proj_desc_key = "proj_desc"

        # ---- callback to add project ----
        def add_project_callback():
            title = st.session_state.get(proj_title_key, "").strip()
            description = st.session_state.get(proj_desc_key, "").strip()
            start_date = st.session_state.get(proj_start_key)
            end_date = st.session_state.get(proj_end_key)
            company = st.session_state.get(proj_company_key, "")
            budget = st.session_state.get(proj_budget_key, "")

            if title:
                st.session_state["projects"].append({
                    "title": title,
                    "description": description,
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "company": company,
                    "budget": budget
                })
                # clear fields safely
                st.session_state[proj_title_key] = ""
                st.session_state[proj_desc_key] = ""
                st.session_state[proj_company_key] = ""
                st.session_state[proj_budget_key] = ""
                st.session_state["_last_added_project"] = title
            else:
                st.session_state["_proj_warning"] = True

        # ---- Input Fields ----
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Project Title", placeholder="Project Name",key=proj_title_key)
            st.date_input("Start Date", key=proj_start_key)
            st.text_input("Project Budget", placeholder="e.g. $5000", key=proj_budget_key)
            
        with col2:
            st.text_input("Company / Institute", key=proj_company_key)
            st.date_input("End Date", key=proj_end_key)

        st.text_area("Project Description", key=proj_desc_key, placeholder="Describe what you did in this project...")

        # Add button with callback
        st.button("➕ Add Project", on_click=add_project_callback)

        # Warnings or messages
        if st.session_state.get("_proj_warning"):
            st.warning("Please fill in at least Project Title and Description.")
            st.session_state["_proj_warning"] = False
        if st.session_state.get("_last_added_project"):
            st.success(f"Added Project: {st.session_state['_last_added_project']}")
            st.session_state["_last_added_project"] = None

        # ---- Show Added Projects ----
        if st.session_state["projects"]:
            st.write("### Added Projects:")
            for i, p in enumerate(st.session_state["projects"], 1):
                st.markdown(
                    f"**{i}. {p['title']}** ({p['start_date']} to {p['end_date']})  \n"
                    f"**Company:** {p['company']}  \n"
                    f"**Budget:** {p['budget']}  \n"
                    f"**Project description:** {p['description']}"
                )
                st.divider()


 # ---- EXTRA COURSES & CERTIFICATES TAB ----
    with tabs[5]:
        st.subheader("Extra Courses & Certificates")

        if "certificates" not in st.session_state:
            st.session_state["certificates"] = []

        # Unique keys
        cert_title_key = "cert_title"
        cert_org_key = "cert_org"
        cert_date_key = "cert_date"

        def add_certificate_callback():
            title = st.session_state.get(cert_title_key, "").strip()
            organization = st.session_state.get(cert_org_key, "").strip()
            date = st.session_state.get(cert_date_key)
            if title:
                st.session_state["certificates"].append({
                    "title": title,
                    "organization": organization,
                    "date": str(date)
                })
                st.session_state[cert_title_key] = ""
                st.session_state[cert_org_key] = ""
                st.session_state["_last_added_certificate"] = title
            else:
                st.session_state["_cert_warning"] = True

        # Input Fields
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Course/Certificate Title", placeholder="e.g. Python Fundamentals", key=cert_title_key)
            st.date_input("Completion Date", key=cert_date_key)
        with col2:
            st.text_input("Organization / Platform", placeholder="e.g. Coursera, Udemy", key=cert_org_key)

        st.button("➕ Add Course / Certificate", on_click=add_certificate_callback)

        if st.session_state.get("_cert_warning"):
            st.warning("Please fill in at least the course title.")
            st.session_state["_cert_warning"] = False
        if st.session_state.get("_last_added_certificate"):
            st.success(f"Added Certificate: {st.session_state['_last_added_certificate']}")
            st.session_state["_last_added_certificate"] = None

        # Display list
        if st.session_state["certificates"]:
            st.write("### Added Courses & Certificates:")
            for i, c in enumerate(st.session_state["certificates"], 1):
                st.markdown(
                    f"**{i}. {c['title']}** ({c['date']})  \n"
                    f"**Organization:** {c['organization']}"
                )
                st.divider()



             # ---- ADDITIONAL INFORMATION TAB ----
    with tabs[6]:
        st.subheader("Additional Information")

        if "languages" not in st.session_state:
            st.session_state["languages"] = []
        if "interests" not in st.session_state:
            st.session_state["interests"] = []

        # --- Languages Section ---
        st.markdown("### Languages")

        lang_input_key = "lang_input"
        def add_language_callback():
            new_lang = st.session_state.get(lang_input_key, "").strip()
            if new_lang:
                st.session_state["languages"].append(new_lang)
                st.session_state[lang_input_key] = ""

        st.text_input("Add a Language", placeholder="e.g. English", key=lang_input_key)
        st.button("Add Language", on_click=add_language_callback)

        if st.session_state["languages"]:
            st.write("**Added Languages:**", ", ".join(st.session_state["languages"]))
        st.divider()

        # --- Interests Section ---
        st.markdown("### Interests")

        interest_input_key = "interest_input"
        def add_interest_callback():
            new_interest = st.session_state.get(interest_input_key, "").strip()
            if new_interest:
                st.session_state["interests"].append(new_interest)
                st.session_state[interest_input_key] = ""

        st.text_input("Add an Interest", placeholder="e.g. Problem-Solving", key=interest_input_key)
        st.button("Add Interest", on_click=add_interest_callback)

        if st.session_state["interests"]:
            st.write("**Added Interests:**", ", ".join(st.session_state["interests"]))


    # ---- EDUCATION TAB ----
    with tabs[2]:
        st.subheader("Add Your Education")

        if "education" not in st.session_state:
            st.session_state["education"] = []

        # Unique keys for each input field
        edu_degree_key = "edu_degree"
        edu_start_key = "edu_start"
        edu_end_key = "edu_end"
        edu_university_key = "edu_university"
        edu_cgpa_key = "edu_cgpa"
        edu_desc_key = "edu_desc"

        # ---- add education ----
        def add_education_callback():
            degree = st.session_state.get(edu_degree_key, "").strip()
            description = st.session_state.get(edu_desc_key, "").strip()
            start_date = st.session_state.get(edu_start_key)
            end_date = st.session_state.get(edu_end_key)
            university = st.session_state.get(edu_university_key, "")
            cgpa = st.session_state.get(edu_cgpa_key, "")

            if degree:
                st.session_state["education"].append({
                    "degree": degree,
                    "description": description,
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "university": university,
                    "cgpa": cgpa
                })
                # Clear fields 
                st.session_state[edu_degree_key] = ""
                st.session_state[edu_desc_key] = ""
                st.session_state[edu_university_key] = ""
                st.session_state[edu_cgpa_key] = ""
                st.session_state["_last_added_education"] = degree
            else:
                st.session_state["_edu_warning"] = True

        # ---- Input Fields ----
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Degree Title", placeholder="e.g. BS Computer Science", key=edu_degree_key)
            st.date_input("Start Date", key=edu_start_key)
            st.text_input("CGPA / Percentage", placeholder="e.g. 3.5 / 4.00", key=edu_cgpa_key)
            
        with col2:
            st.text_input("University / Institute", placeholder="e.g. Sukkur IBA University", key=edu_university_key)
            st.date_input("End Date", key=edu_end_key)

        st.text_area("Description", key=edu_desc_key, placeholder="Mention key coursework, thesis, or achievements...")

        # Add button with callback
        st.button("➕ Add Education", on_click=add_education_callback)

        # Warnings or messages
        if st.session_state.get("_edu_warning"):
            st.warning("Please fill in at least Degree Title")
            st.session_state["_edu_warning"] = False
        if st.session_state.get("_last_added_education"):
            st.success(f"Added Education: {st.session_state['_last_added_education']}")
            st.session_state["_last_added_education"] = None

        # ---- Show Added Education ----
        if st.session_state["education"]:
            st.write("### Added Education:")
            for i, e in enumerate(st.session_state["education"], 1):
                st.markdown(
                    f"**{i}. {e['degree']}** ({e['start_date']} to {e['end_date']})  \n"
                    f"**University:** {e['university']}  \n"
                    f"**CGPA:** {e['cgpa']}  \n"
                    f"**Description:** {e['description']}"
                )
                st.divider()



    # ---- EXPERIENCE TAB ----
    with tabs[1]:
        st.subheader("Add Your Work Experience")

        if "experience" not in st.session_state:
            st.session_state["experience"] = []

        # Unique keys for input fields
        exp_title_key = "exp_title"
        exp_company_key = "exp_company"
        exp_start_key = "exp_start"
        exp_end_key = "exp_end"
        exp_desc_key = "exp_desc"

        # Callback to add experience
        def add_experience_callback():
            title = st.session_state.get(exp_title_key, "").strip()
            company = st.session_state.get(exp_company_key, "").strip()
            start_date = st.session_state.get(exp_start_key)
            end_date = st.session_state.get(exp_end_key)
            description = st.session_state.get(exp_desc_key, "").strip()

            if title and company:
                st.session_state["experience"].append({
                    "title": title,
                    "company": company,
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "description": description
                })
                st.session_state[exp_title_key] = ""
                st.session_state[exp_company_key] = ""
                st.session_state[exp_desc_key] = ""
                st.session_state["_last_added_experience"] = title
            else:
                st.session_state["_exp_warning"] = True

        # Input fields
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Job Title / Position", placeholder="Software Engineer", key=exp_title_key)
            st.date_input("Start Date", key=exp_start_key)
        with col2:
            st.text_input("Company / Organization", placeholder="ABC Pvt Ltd", key=exp_company_key)
            st.date_input("End Date", key=exp_end_key)

        st.text_area("Description (Optional)", placeholder="Describe your role, achievements, or key projects...", key=exp_desc_key)

        st.button("➕ Add Experience", on_click=add_experience_callback)

        # Warnings 
        if st.session_state.get("_exp_warning"):
            st.warning("Please fill in at least Job Title and Company.")
            st.session_state["_exp_warning"] = False
        if st.session_state.get("_last_added_experience"):
            st.success(f"Added Experience: {st.session_state['_last_added_experience']}")
            st.session_state["_last_added_experience"] = None

        # Show added experiences
        if st.session_state["experience"]:
            st.write("### Added Experience:")
            for i, e in enumerate(st.session_state["experience"], 1):
                st.markdown(
                    f"**{i}. {e['title']}** ({e['start_date']} to {e['end_date']})  \n"
                    f"**Company:** {e['company']}  \n"
                    f"**Description:** {e['description'] if e['description'] else 'N/A'}"
                )
                st.divider()

    


        #----------- Generate Resume ---------------

    if st.button("Generate Resume"):
        resume_data = {
            "name": personal.get("name", ""),
            "title": personal.get("title", ""),
            "email": personal.get("email", ""),
            "linkedin": personal.get("linkedin", ""),
            "github": personal.get("github", ""),
            "phone": personal.get("phone", ""),
            "location": personal.get("location", ""),
            "summary": personal.get("summary", ""),
            "skills": st.session_state["skills"],
            "projects": st.session_state["projects"],
            "experience": st.session_state.get("experience", []),

            "education": st.session_state.get("education", []),
            "certificates": st.session_state.get("certificates", []),

            "languages": st.session_state.get("languages", []),
            "interests": st.session_state.get("interests", [])
        }

        # Display Preview
        st.subheader("Resume Preview")
        md = f"### {resume_data['name']}\n**{resume_data['title']}**\n\n"
        contact_line = " | ".join([
            x for x in [
                resume_data['email'],
                resume_data['linkedin'],
                resume_data['github'],
                resume_data['phone'],
                resume_data['location']
            ] if x
        ])
        if contact_line:
            md += contact_line + "\n\n"
        if resume_data['summary']:
            md += resume_data['summary'] + "\n\n"

        if resume_data['education']:
            md += "**Education**\n\n"
            for i, edu in enumerate(resume_data['education'], 1):
                md += f"{i}. **{edu['degree']}**, {edu['university']} ({edu['start_date']} - {edu['end_date']})  \n"
                if edu['cgpa']:
                    md += f"CGPA: {edu['cgpa']}  \n"
                if edu['description']:
                    md += f"{edu['description']}  \n"
                md += "\n"

        if resume_data['skills']:
            md += "**Skills**\n\n" + ", ".join(resume_data['skills']) + "\n\n"

        if resume_data['experience']:
            md += "**Experience**\n\n"
            for exp in resume_data['experience']:
                md += f"**{exp['title']}** at {exp['company']}  \n"
                md += f"*{exp['start_date']} - {exp['end_date']}*  \n"
                if exp['description']:
                    md += f"{exp['description']}\n\n"


        if resume_data['projects']:
            md += "**Projects**\n\n"
            for proj in resume_data['projects']:
                md += f"**{proj['title']}** ({proj['start_date']} - {proj['end_date']})  \n"
                md += f"**Company:** {proj['company']}  \n"
                md += f"**Budget:** {proj['budget']}  \n"
                md += f"**Project description:** {proj['description']}\n\n"

        if resume_data.get('certificates'):
            md += "**Courses & Certificates**\n\n"
            for cert in resume_data['certificates']:
                md += f"**{cert['title']}**  \n"
                md += f"{cert['organization']}  \n"
                md += f"**Completion Date:** {cert['date']}\n\n"


        if resume_data.get("languages") or resume_data.get("interests"):
            md += "**Additional Information**\n\n"
            if resume_data.get("languages"):
                md += f"**Languages:** {', '.join(resume_data['languages'])}\n\n"
            if resume_data.get("interests"):
                md += f"**Interests:** {', '.join(resume_data['interests'])}\n\n"

        st.markdown(md)

        pdf_bytes = create_pdf(resume_data)
        st.download_button("Download Resume as PDF", data=pdf_bytes, file_name="resume.pdf", mime="application/pdf")

# -------------------- CV Upload Section --------------------
elif mode == 'upload':
    st.header("📤 Upload Your Resume (ATS Score)")
    uploaded = st.file_uploader("Upload your CV (PDF or TXT)", type=["pdf", "txt"])
    if uploaded:
        
        resume_text = parse_uploaded_resume(uploaded)

        st.success("✅ Your CV has been uploaded successfully!")

        # ATS analysis button
        if st.button("Analyze ATS Score"):
            with st.spinner("Analyzing CV for ATS optimization..."):
                result = analyze_cv_with_gemini(resume_text)
                if result.get("error"):
                    st.error(result["error"])
                else:
                    # show score
                    score = result.get("score")
                    if isinstance(score, (int, float)):
                        st.metric("ATS Score", f"{score} / 100")

                    # matching jobs
                    matching_jobs = result.get("matching_jobs", [])
                    if matching_jobs:
                        st.markdown("**Matching Jobs (Based on your CV):**")
                        for job in matching_jobs:
                            st.write(f"- {job}")

                    # highlights
                    highlights = result.get("highlights", [])
                    if highlights:
                        st.markdown("**Highlights:**")
                        for h in highlights:
                            st.write("- " + h)

                    # improvements
                    improvements = result.get("improvements", [])
                    if improvements:
                        st.markdown("**Improvements:**")
                        for imp in improvements:
                            st.write("- " + imp)

    else:
        st.info("Please upload your CV (PDF/TXT) to analyze.")

