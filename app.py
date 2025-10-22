import json
from flask import Flask, request, jsonify, render_template_string
from rag_pipeline import answer_question # Import your core logic

app = Flask(__name__)

# --- HTML Template for the Web Interface (Endpoint /) ---
HTML_TEMPLATE = """
<!doctype html>
<title>Policy Q&A Bot - Snug-Project</title>
<style>
    body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f9; }
    .container { max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); }
    h1 { color: #333; text-align: center; }
    form { display: flex; margin-bottom: 20px; }
    input[type="text"] { flex-grow: 1; padding: 10px; border: 1px solid #ccc; border-radius: 4px 0 0 4px; }
    input[type="submit"] { background-color: #007bff; color: white; padding: 10px 15px; border: none; border-radius: 0 4px 4px 0; cursor: pointer; }
    .answer-box { border: 1px solid #ddd; padding: 15px; border-radius: 4px; background-color: #e9ecef; }
    .sources { margin-top: 10px; font-size: 0.9em; color: #555; }
</style>
<div class="container">
    <h1>Snug-Project Policy Q&A Bot</h1>

    <form method="post" action="/">
        <input type="text" name="question" placeholder="Ask a question about company policies..." required>
        <input type="submit" value="Ask">
    </form>

    {% if answer %}
        <h2>Answer:</h2>
        <div class="answer-box">
            {{ answer | safe }}
        </div>
    {% endif %}

    <p style="text-align: center; margin-top: 20px;">Use the /chat endpoint for API access, or the /health endpoint for status.</p>
</div>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    """Endpoint / - Web chat interface."""
    answer = None
    if request.method == 'POST':
        question = request.form['question']
        raw_answer = answer_question(question)
        
        # Simple formatting for web display
        answer = raw_answer.replace("\n\n**Sources:**", '<div class="sources">Sources:') + '</div>'
        
    return render_template_string(HTML_TEMPLATE, answer=answer)

@app.route('/chat', methods=['POST'])
def chat_api():
    """Endpoint /chat - API endpoint that receives user questions."""
    
    # Project requirement: API endpoint that receives user questions (POST)
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    # Get the raw answer
    raw_answer = answer_question(question)
    
    # Simple extraction of answer and sources for JSON output
    if "**Sources:**" in raw_answer:
        answer_part, sources_part = raw_answer.split("**Sources:**", 1)
        sources = [s.strip() for s in sources_part.split(';') if s.strip()]
    else:
        # Case for guardrail/refusal messages
        answer_part = raw_answer
        sources = []

    # Project requirement: returns model-generated answers with citations
    return jsonify({
        "question": question,
        "answer": answer_part.strip(),
        "citations": sources
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint /health - returns simple status via JSON."""
    
    # Project requirement: returns simple status via JSON
    return jsonify({"status": "ok", "service": "snug-project-rag-bot"})

if __name__ == '__main__':
    # Flask runs on port 5000 by default
    app.run(debug=True)
