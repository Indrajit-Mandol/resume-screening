import sys
import streamlit as st
import re
import nltk
import spacy
import PyPDF2
import pytesseract
import pdf2image
import numpy as np
from io import BytesIO
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from spacy import displacy
import plotly.express as px

st.write("Python version:", sys.version)



# Initialize NLP
nlp = spacy.load("en_core_web_lg")
nltk.download('punkt')
nltk.download('stopwords')

# ------------------- Advanced PDF Processing -------------------
def extract_text_from_pdf(file):
    try:
        # First try standard text extraction
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text.strip() != "":
                text += page_text + "\n"
        if text.strip() != "":
            return text
        
        # Fallback to OCR for image-based PDFs
        images = pdf2image.convert_from_bytes(file.read())
        text = ""
        for image in images:
            text += pytesseract.image_to_string(np.array(image)) + "\n"
        return text
    except Exception as e:
        st.error(f"PDF Processing Error: {str(e)}")
        return ""

# ------------------- AI-Powered Resume Parser -------------------
def parse_resume(text):
    doc = nlp(text)
    entities = {}
    
    # Extract entities
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    
    # Enhanced experience extraction
    experience = re.findall(r"(\d+[+]?\s*(?:years?|yrs?))", text, re.I)
    
    # Skill extraction using matcher
    skill_pattern = [{"LOWER": {"IN": ["python", "java", "sql"]}},
                    {"LOWER": "machine", "OP": "?", "LOWER": "learning"}]
    matcher = spacy.matcher.Matcher(nlp.vocab)
    matcher.add("SKILLS", [skill_pattern])
    matches = matcher(doc)
    skills = [doc[start:end].text for _, start, end in matches]
    
    return {
        "name": entities.get("PERSON", ["Not Found"])[0],
        "email": re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text),
        "phone": re.findall(r"\b(?:\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", text),
        "skills": list(set(skills)),
        "experience": list(set(experience)),
        "education": entities.get("ORG", []),
        "entities": entities
    }

# ------------------- Advanced Text Analysis -------------------
def analyze_text(text):
    doc = nlp(text)
    pos_counts = {}
    for token in doc:
        if token.pos_ not in pos_counts:
            pos_counts[token.pos_] = 0
        pos_counts[token.pos_] += 1
    
    return {
        "readability_score": len(text) / (len(text.split()) + 1e-9),
        "pos_distribution": pos_counts,
        "ner_visualization": displacy.render(doc, style="ent")
    }

# ------------------- ATS Scoring System -------------------
def calculate_ats_score(resume, jd):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume, jd])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Add other scoring factors
    score = similarity * 100
    if len(resume.split()) < 300:
        score *= 0.9  # Penalize short resumes
    return min(score, 100)

# ------------------- UI Configuration -------------------
st.set_page_config(page_title="ResumeIQ Pro", layout="wide", page_icon="📊")
st.title("🚀 ResumeIQ Pro - Advanced Resume Analysis System")

# ------------------- Sidebar Controls -------------------
with st.sidebar:
    st.header("Configuration")
    analysis_mode = st.selectbox("Analysis Mode", 
                               ["Standard", "Advanced", "Expert"],
                               help="Choose analysis depth level")
    enable_ocr = st.checkbox("Enable OCR", True,
                            help="Process image-based PDFs (slower)")
    risk_level = st.slider("Risk Threshold", 1, 5, 3,
                          help="Adjust matching strictness")

# ------------------- Main Interface -------------------
upload_col, jd_col = st.columns([2, 3])
with upload_col:
    uploaded_files = st.file_uploader("Upload Resumes (PDF/DOCX)", 
                                    type=["pdf", "docx"],
                                    accept_multiple_files=True,
                                    help="Max 5 files, 5MB each")

with jd_col:
    jd_text = st.text_area("Paste Job Description", height=250,
                          placeholder="Enter job description here...",
                          help="Include key requirements and qualifications")

# ------------------- Analysis Controls -------------------
if st.button("🔍 Start Analysis", use_container_width=True):
    if uploaded_files and jd_text:
        with st.spinner("🚀 Processing Documents..."):
            # ------------------- Process Resumes -------------------
            resume_data = []
            for file in uploaded_files:
                try:
                    if file.type == "application/pdf":
                        text = extract_text_from_pdf(file)
                    else:
                        text = file.getvalue().decode()
                    
                    parsed = parse_resume(text)
                    analysis = analyze_text(text)
                    ats_score = calculate_ats_score(text, jd_text)
                    
                    resume_data.append({
                        "file": file.name,
                        "text": text,
                        "parsed": parsed,
                        "analysis": analysis,
                        "ats_score": ats_score
                    })
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
            
            # Store in session state
            st.session_state.resume_data = resume_data
            st.session_state.jd_analysis = analyze_text(jd_text)

# ------------------- Results Display -------------------
if "resume_data" in st.session_state:
    st.header("📊 Analysis Results")
    
    # ------------------- Summary Dashboard -------------------
    with st.expander("📈 Summary Dashboard", expanded=True):
        cols = st.columns(4)
        cols[0].metric("Total Resumes", len(st.session_state.resume_data))
        cols[1].metric("Top ATS Score", 
                      f"{max(r['ats_score'] for r in st.session_state.resume_data):.1f}%")
        cols[2].metric("Average Experience", 
                       np.mean([len(r['parsed']['experience']) for r in st.session_state.resume_data]))
        cols[3].metric("Keyword Density", 
                       len(st.session_state.jd_analysis['pos_distribution']))
    
    # ------------------- Resume Comparison -------------------
    with st.expander("🔍 Detailed Analysis", expanded=True):
        tabs = st.tabs([f"📄 {r['file']}" for r in st.session_state.resume_data] + ["📋 JD Analysis"])
        
        for i, resume in enumerate(st.session_state.resume_data):
            with tabs[i]:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader("Candidate Profile")
                    st.write(f"**Name:** {resume['parsed'].get('name', 'N/A')}")
                    st.write(f"**Contact:** {', '.join(resume['parsed'].get('email', []))}")
                    st.write(f"**Experience:** {', '.join(resume['parsed'].get('experience', []))}")
                    st.write(f"**Education:** {', '.join(resume['parsed'].get('education', []))}")
                    
                    st.subheader("ATS Score")
                    st.progress(resume['ats_score']/100, 
                               f"{resume['ats_score']:.1f}% - {'Good' if resume['ats_score'] > 70 else 'Needs Improvement'}")
                
                with col2:
                    st.subheader("Text Analysis")
                    st.plotly_chart(px.bar(pd.DataFrame.from_dict(
                        resume['analysis']['pos_distribution'], 
                        orient='index').reset_index(),
                        x='index', y=0,
                        labels={'index': 'POS Tag', 0: 'Count'},
                        title="Part-of-Speech Distribution"))
                    
                    st.subheader("NER Visualization")
                    st.markdown(resume['analysis']['ner_visualization'], 
                              unsafe_allow_html=True)
        
        # JD Analysis Tab
        with tabs[-1]:
            st.subheader("Job Description Insights")
            st.plotly_chart(px.treemap(
                pd.DataFrame.from_dict(
                    st.session_state.jd_analysis['pos_distribution'],
                    orient='index').reset_index(),
                path=['index'], values=0,
                title="POS Distribution in JD"))
            
            st.subheader("Keyword Cloud")
            wordcloud = WordCloud().generate(jd_text)
            plt.figure(figsize=(10,5))
            plt.imshow(wordcloud)
            plt.axis("off")
            st.pyplot(plt)

    # ------------------- AI Recommendations -------------------
    with st.expander("🤖 AI-Powered Optimization Tips", expanded=True):
        for resume in st.session_state.resume_data:
            st.subheader(f"Recommendations for {resume['file']}")
            
            tips = []
            if len(resume['text'].split()) < 300:
                tips.append("⚠️ Resume appears too short - consider adding more details")
            if 'PROG' not in resume['analysis']['pos_distribution']:
                tips.append("🔧 Consider adding more technical skills")
            if resume['ats_score'] < 70:
                tips.append("📈 Improve keyword alignment with job description")
            
            if tips:
                for tip in tips:
                    st.info(tip)
            else:
                st.success("✅ Strong resume structure detected!")

    # ------------------- Data Export -------------------
    st.download_button("💾 Export Full Report", 
                      data=pd.DataFrame(st.session_state.resume_data).to_csv(),
                      file_name="resume_analysis_report.csv",
                      mime="text/csv",
                      use_container_width=True)
