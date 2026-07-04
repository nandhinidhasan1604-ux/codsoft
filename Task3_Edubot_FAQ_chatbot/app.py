import nltk
import pandas as pd
import gradio as gr
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")

faq_data = [
    {"question": "What documents are required for college admission?", "answer": "You need 10th marksheet, 12th marksheet, transfer certificate, conduct certificate, community certificate, and passport size photos for admission."},
    {"question": "How do I apply for a bonafide certificate?", "answer": "Submit an application to the college office with your name, roll number, and purpose. Bonafide certificate will be issued within 2 working days."},
    {"question": "How to get transfer certificate from college?", "answer": "Submit a written application to the principal office with your reason for leaving. TC will be issued after clearing all dues within 7 working days."},
    {"question": "How to get no objection certificate from college?", "answer": "Apply for NOC at the principal office with details of the event or program you are attending. NOC is issued within 2 working days."},
    {"question": "How to get duplicate ID card?", "answer": "Apply for duplicate ID card at the admin office with a written application and Rs.200 fee. Duplicate ID card will be issued within 3 working days."},
    {"question": "How to get course completion certificate?", "answer": "Course completion certificate is issued after all arrears are cleared and dues are paid. Apply at the exam section after your final semester results."},
    {"question": "How to get migration certificate?", "answer": "Apply for migration certificate at the exam section with TC and original marksheets. Migration certificate is issued within 7 working days."},
    {"question": "What is the process for exam fee payment?", "answer": "Exam fees can be paid online through the college portal or at the college office. The last date is usually announced 30 days before exams."},
    {"question": "How to apply for arrear exam?", "answer": "Log in to the university portal, select arrear exam registration, pay the fee, and submit before the deadline. Check the notice board for dates."},
    {"question": "What is the attendance requirement to write exams?", "answer": "You need a minimum of 75% attendance to be eligible to write semester exams. Students below 75% must get special permission from the principal."},
    {"question": "How do I get my semester marksheet?", "answer": "Semester marksheets are available for download on the university portal after results are published. Physical copies can be collected from the exam section."},
    {"question": "What is the process for applying revaluation?", "answer": "Apply for revaluation through the university portal within 2 weeks of result publication. Pay the revaluation fee and submit the application online."},
    {"question": "How to contact the college exam cell?", "answer": "The exam cell can be reached at examcell@college.edu or visit room number 105 in the admin block between 10 AM and 3 PM on working days."},
    {"question": "How to apply for study leave?", "answer": "Submit a study leave application to your class advisor at least 2 days before the exam. Study leave is granted for 3 days before semester exams."},
    {"question": "Where can I get previous year question papers?", "answer": "Previous year question papers are available in the college library and on the university official website under the examination section."},
    {"question": "How are internal marks calculated?", "answer": "Internal marks are calculated based on attendance (5 marks), assignment (5 marks), and internal test performance (15 marks). Total internal marks is 25."},
    {"question": "How many internal tests are conducted?", "answer": "Two internal tests are conducted per semester. Best of two marks are considered for internal assessment in most subjects."},
    {"question": "How to improve internal marks?", "answer": "Attend all classes regularly, submit assignments on time, perform well in internal tests, and participate in seminars to improve internal marks."},
    {"question": "How to pay college tuition fees?", "answer": "Tuition fees can be paid online through the college payment portal or via DD at the accounts section. Pay before the due date to avoid fine."},
    {"question": "What is the fine for late fee payment?", "answer": "A fine of Rs.100 per day is charged for late fee payment after the due date. Contact the accounts section for fee due date details."},
    {"question": "Is there any fee concession available?", "answer": "Fee concession is available for students from economically weaker sections. Apply through the college office with income certificate and community certificate."},
    {"question": "What scholarships are available for students?", "answer": "Government scholarships like BC, MBC, SC/ST scholarships are available. Apply through the Tamil Nadu government scholarship portal before the deadline each year."},
    {"question": "How to apply for sports quota admission?", "answer": "Sports quota admissions require a sports certificate from district or national level. Apply through the sports department with certificates during admission period."},
    {"question": "How do I register for placement drives?", "answer": "Register on the placement portal using your college email. Keep your resume updated and attend the pre-placement orientation conducted by the training and placement officer."},
    {"question": "When does the new semester start?", "answer": "The new semester typically starts in July for odd semesters and January for even semesters. Check the official college calendar for exact dates."},
    {"question": "How to apply for hostel accommodation?", "answer": "Fill the hostel application form available at the admin office. Submit with your parent contact details and fee payment receipt. Rooms are allotted based on availability."},
    {"question": "What is the library timing and borrowing limit?", "answer": "Library is open from 9 AM to 5 PM on weekdays. Students can borrow up to 3 books at a time for a period of 14 days."},
    {"question": "What are the rules for using the college wifi?", "answer": "College wifi is available for registered students only. Login with your college ID and password. Misuse of wifi for illegal activities will result in suspension."},
    {"question": "How to access online learning resources?", "answer": "College provides access to NPTEL, SWAYAM, and digital library resources. Login with your college credentials to access all online courses."},
    {"question": "How to meet the Head of Department?", "answer": "You can meet the HOD during office hours between 10 AM and 12 PM on working days. Take prior appointment from the department office."},
    {"question": "How to apply for industrial visit?", "answer": "Industrial visit applications are submitted through your class advisor. The department coordinator will arrange visits based on syllabus requirements."},
    {"question": "How to participate in college events?", "answer": "Register for college events through the student affairs office or the event registration portal. Watch the notice board for upcoming event announcements."},
    {"question": "How to join college clubs and committees?", "answer": "Club registrations happen at the start of each academic year. Contact the respective club coordinator or student affairs office to register."},
    {"question": "Does college provide bus facility?", "answer": "Yes college provides bus facility covering major routes. Contact the transport office for route details and monthly bus pass charges."},
    {"question": "How to get bus pass from college?", "answer": "Apply for bus pass at the transport office with your ID card and fee receipt. Monthly and semester bus passes are available at subsidized rates."},
    {"question": "Does college have a medical facility?", "answer": "Yes the college has a medical room with a nurse available from 9 AM to 5 PM. For emergencies contact the college office immediately."},
    {"question": "What to do in case of medical emergency in college?", "answer": "Contact the college security or office immediately. The college has a tie up with nearby hospitals for emergency medical care for students."},
    {"question": "How to raise a grievance or complaint?", "answer": "Submit grievance in writing to the grievance committee through your class advisor or directly at the principal office. All complaints are addressed within 7 days."},
    {"question": "Is there an anti ragging committee in college?", "answer": "Yes the college has an anti ragging committee. Report any ragging incidents to the committee or call the anti ragging helpline immediately."},
    {"question": "How to get internet access in college campus?", "answer": "College campus has wifi coverage in all blocks. Register your device at the IT department with your college ID to get wifi access credentials."}
]

df = pd.DataFrame(faq_data)
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

preprocessed_questions = df["question"].apply(preprocess_text)
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(preprocessed_questions)

custom_css = """
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes titleGlow {
    0%, 100% { text-shadow: 0 0 10px #00d4ff, 0 0 20px #00d4ff; }
    50% { text-shadow: 0 0 20px #ff00ff, 0 0 40px #ff00ff; }
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 10px #00d4ff; }
    50% { box-shadow: 0 0 20px #ff00ff, 0 0 40px #ff00ff; }
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0px); }
}
@keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 0.7; }
    33% { transform: translateY(-30px) rotate(120deg); opacity: 1; }
    66% { transform: translateY(-15px) rotate(240deg); opacity: 0.8; }
}
.gradio-container {
    background: linear-gradient(-45deg, #0a0a1a, #0d1b2a, #1a0a2e, #0a1628) !important;
    background-size: 400% 400% !important;
    animation: gradientShift 8s ease infinite !important;
    font-family: 'Segoe UI', sans-serif !important;
}
.gradio-container::before {
    content: '🤖 ⭐ 💡 📚 🎓 ✨ 🔬 💫';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    font-size: 1.5em;
    animation: float 6s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
    opacity: 0.15;
    word-spacing: 100px;
    line-height: 200px;
}
.gradio-container h1 {
    color: #00d4ff !important;
    text-align: center !important;
    font-size: 2.5em !important;
    font-weight: 800 !important;
    animation: titleGlow 3s ease-in-out infinite !important;
    letter-spacing: 2px !important;
    padding: 20px 0 5px 0 !important;
}
.gradio-container .prose p {
    color: #a8c7fa !important;
    text-align: center !important;
}
.gradio-container .block {
    background: rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    border-radius: 20px !important;
}
.message.user {
    background: linear-gradient(135deg,rgba(0,212,255,0.2),rgba(0,100,200,0.3)) !important;
    color: #ffffff !important;
    border-radius: 20px 20px 5px 20px !important;
    padding: 15px 20px !important;
    border: 1px solid rgba(0,212,255,0.5) !important;
    box-shadow: 0 0 15px rgba(0,212,255,0.3) !important;
    animation: slideIn 0.3s ease !important;
}
.message.bot {
    background: linear-gradient(135deg,rgba(255,0,255,0.1),rgba(100,0,200,0.2)) !important;
    color: #e8f4fd !important;
    border-radius: 20px 20px 20px 5px !important;
    padding: 15px 20px !important;
    border: 1px solid rgba(255,0,255,0.3) !important;
    box-shadow: 0 0 15px rgba(255,0,255,0.2) !important;
    animation: slideIn 0.3s ease !important;
}
.gradio-container textarea {
    background: rgba(0,10,30,0.8) !important;
    color: #00d4ff !important;
    border: 2px solid #00d4ff !important;
    border-radius: 15px !important;
    padding: 15px !important;
}
.gradio-container textarea:focus {
    border-color: #ff00ff !important;
    box-shadow: 0 0 20px rgba(255,0,255,0.5) !important;
    color: #ffffff !important;
}
.gradio-container button.primary {
    background: linear-gradient(135deg, #00d4ff, #ff00ff) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 15px !important;
    font-weight: 800 !important;
    padding: 15px 30px !important;
    transition: all 0.3s ease !important;
    animation: pulse 3s ease-in-out infinite !important;
}
.gradio-container button.primary:hover {
    transform: translateY(-3px) scale(1.05) !important;
}
.gradio-container button.secondary {
    background: rgba(0,212,255,0.1) !important;
    color: #00d4ff !important;
    border: 1px solid #00d4ff !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}
.gradio-container button.secondary:hover {
    background: rgba(0,212,255,0.3) !important;
    transform: translateY(-2px) !important;
}
.gradio-container .chatbot {
    background: rgba(0,5,20,0.7) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    border-radius: 20px !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.3); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#00d4ff, #ff00ff);
    border-radius: 3px;
}
.gradio-container .example-set button {
    background: rgba(0,212,255,0.1) !important;
    color: #00d4ff !important;
    border: 1px solid rgba(0,212,255,0.4) !important;
    border-radius: 25px !important;
    padding: 8px 18px !important;
    transition: all 0.3s ease !important;
    margin: 4px !important;
}
.gradio-container .example-set button:hover {
    background: linear-gradient(135deg, #00d4ff, #ff00ff) !important;
    color: #ffffff !important;
    transform: translateY(-2px) !important;
}
"""

CATEGORIES = {
    "📋 Admissions": "What documents are required for college admission?",
    "📝 Exams": "What is the attendance requirement to write exams?",
    "🎓 Certificates": "How do I apply for a bonafide certificate?",
    "💰 Scholarships": "What scholarships are available for students?",
    "💼 Placements": "How do I register for placement drives?",
    "🏠 Hostel": "How to apply for hostel accommodation?"
}

def get_confidence_bar(score):
    percentage = int(score * 100)
    filled = int(percentage / 5)
    empty = 20 - filled
    if percentage >= 70:
        label = "High Match 🟢"
        bar_char = "█"
    elif percentage >= 40:
        label = "Good Match 🟡"
        bar_char = "▓"
    else:
        label = "Low Match 🔴"
        bar_char = "░"
    bar = bar_char * filled + "░" * empty
    return f"\n\n📊 Confidence: [{bar}] {percentage}% — {label}"

def chatbot_response_ui(user_message, history):
    if not user_message.strip():
        return "", history
    time.sleep(0.5)
    cleaned = preprocess_text(user_message)
    user_vector = vectorizer.transform([cleaned])
    scores = cosine_similarity(user_vector, faq_vectors)
    best_idx = scores.argmax()
    best_score = scores[0][best_idx]
    if best_score < 0.25:
        bot_reply = (
            "❌ Sorry! I could not find a relevant answer.\n\n"
            "Please contact:\n"
            "📞 College Office: 044-XXXXXXXX\n"
            "📧 Email: info@college.edu"
            + get_confidence_bar(best_score)
        )
    else:
        answer = df["answer"][best_idx]
        matched_q = df["question"][best_idx]
        bot_reply = (
            f"✅ {answer}\n\n"
            f"📌 Matched: \"{matched_q}\""
            + get_confidence_bar(best_score)
        )
    history = history + [[user_message, bot_reply]]
    return "", history

def clear_chat():
    return []

def ask_category(question, history):
    time.sleep(0.3)
    cleaned = preprocess_text(question)
    user_vector = vectorizer.transform([cleaned])
    scores = cosine_similarity(user_vector, faq_vectors)
    best_idx = scores.argmax()
    best_score = scores[0][best_idx]
    answer = df["answer"][best_idx]
    matched_q = df["question"][best_idx]
    bot_reply = (
        f"✅ {answer}\n\n"
        f"📌 Matched: \"{matched_q}\""
        + get_confidence_bar(best_score)
    )
    history = history + [[question, bot_reply]]
    return history

welcome_msg = "🤖 Hello! I am EduBot!\n\n⚡ Your Smart College FAQ Assistant!\n\nAsk me anything about:\n🎓 Admissions | 📝 Exams | 📜 Certificates\n💰 Scholarships | 💼 Placements | 🏠 Hostel\n\nType your question or click a category below! 👇"

with gr.Blocks(css=custom_css, title="EduBot") as demo:

    gr.Markdown("# 🎓 EduBot — Smart College FAQ Assistant")
    gr.Markdown("#### ⚡ Powered by NLTK • TF-IDF Vectorization • Cosine Similarity")

    with gr.Row():
        gr.Markdown("""<div style='text-align:center;
        background:rgba(0,212,255,0.1);
        border:1px solid rgba(0,212,255,0.4);
        border-radius:12px;padding:10px;color:#a8c7fa'>
        📚 <b style='color:#00d4ff'>40</b><br>FAQ Questions
        </div>""")
        gr.Markdown("""<div style='text-align:center;
        background:rgba(0,212,255,0.1);
        border:1px solid rgba(0,212,255,0.4);
        border-radius:12px;padding:10px;color:#a8c7fa'>
        🧠 <b style='color:#00d4ff'>TF-IDF</b><br>NLP Engine
        </div>""")
        gr.Markdown("""<div style='text-align:center;
        background:rgba(0,212,255,0.1);
        border:1px solid rgba(0,212,255,0.4);
        border-radius:12px;padding:10px;color:#a8c7fa'>
        🎯 <b style='color:#00d4ff'>Cosine</b><br>Similarity
        </div>""")
        gr.Markdown("""<div style='text-align:center;
        background:rgba(0,212,255,0.1);
        border:1px solid rgba(0,212,255,0.4);
        border-radius:12px;padding:10px;color:#a8c7fa'>
        ⚡ <b style='color:#00d4ff'>Instant</b><br>Response
        </div>""")

    gr.Markdown("---")

    chatbot_ui = gr.Chatbot(
        value=[[None, welcome_msg]],
        height=430,
        show_label=False
    )

    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="💬 Ask EduBot anything about your college...",
            show_label=False,
            scale=5,
            container=False
        )
        send_btn = gr.Button("Send ✨", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

    gr.Markdown("---")
    gr.Markdown(
        "<div style='color:#00d4ff;font-weight:700;"
        "margin-bottom:8px'>⚡ Quick Categories:</div>"
    )

    with gr.Row():
        cat_btns = []
        for label, question in CATEGORIES.items():
            btn = gr.Button(label, variant="secondary")
            cat_btns.append((btn, question))

    gr.Examples(
        examples=[
            "How to apply for bonafide certificate?",
            "What is minimum attendance for exams?",
            "How to register for campus placement?",
            "What scholarships are available?",
            "How to get transfer certificate?",
            "How to pay college fees?"
        ],
        inputs=msg_input
    )

    gr.Markdown("""---
<div style='text-align:center;color:#4a6fa5;
font-size:0.8em;padding:8px'>
🎓 EduBot | NLTK • TF-IDF • Cosine Similarity • Gradio | 40 FAQs
</div>""")

    send_btn.click(
        chatbot_response_ui,
        inputs=[msg_input, chatbot_ui],
        outputs=[msg_input, chatbot_ui]
    )
    msg_input.submit(
        chatbot_response_ui,
        inputs=[msg_input, chatbot_ui],
        outputs=[msg_input, chatbot_ui]
    )
    clear_btn.click(clear_chat, outputs=[chatbot_ui])

    for btn, question in cat_btns:
        btn.click(
            fn=lambda history, q=question: ask_category(q, history),
            inputs=[chatbot_ui],
            outputs=[chatbot_ui]
        )

demo.launch()