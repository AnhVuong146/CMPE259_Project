import sqlite3, json, time, re, os, warnings
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings("ignore")

app = Flask(__name__)

#Database 
DB_PATH = os.path.join(os.path.dirname(__file__), "sjsu_msai.db")

APPROVED_DOMAINS = ["sjsu.edu", "catalog.sjsu.edu", "www.sjsu.edu"]

def query_database(query: str) -> dict:
    ql = query.lower()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    result = {"status": "success", "data": [], "message": ""}
    try:
        if any(kw in ql for kw in ["prerequisite","prereq","need before","before taking","required before"]):
            m = re.search(r'(cmpe|ise|engr)\s*(\d+[a-z]?)', ql)
            if m:
                cid = (m.group(1).upper() + " " + m.group(2)).strip()
                c.execute("""
                    SELECT p.course_id, p.prereq_id, p.prereq_description,
                           c.title AS prereq_title
                    FROM prerequisites p
                    LEFT JOIN courses c ON p.prereq_id = c.course_id
                    WHERE UPPER(p.course_id) = ?
                """, (cid,))
                rows = c.fetchall()
                if rows:
                    result["data"] = [dict(r) for r in rows]
                    result["message"] = f"Prerequisites for {cid}"
                else:
                    result["message"] = f"No listed prerequisites for {cid} (or none required)."
            else:
                result["message"] = "Specify a course ID, e.g. CMPE 258."
        elif any(kw in ql for kw in ["core course","required course","mandatory"]):
            c.execute("SELECT * FROM courses WHERE category='Core' ORDER BY course_id")
            result["data"] = [dict(r) for r in c.fetchall()]
            result["message"] = "Core courses (9 units required)"
        elif any(kw in ql for kw in ["elective","area a","area b"]):
            if "area a" in ql:
                c.execute("SELECT * FROM courses WHERE category='Elective' AND area='Area A'")
            elif "area b" in ql:
                c.execute("SELECT * FROM courses WHERE category='Elective' AND area='Area B'")
            else:
                c.execute("SELECT * FROM courses WHERE category='Elective' ORDER BY area, course_id")
            result["data"] = [dict(r) for r in c.fetchall()]
            result["message"] = "Elective courses"
        elif any(kw in ql for kw in ["specialization","data science","autonomous"]):
            if "data science" in ql:
                c.execute("""
                    SELECT DISTINCT c.* FROM courses c
                    JOIN specializations s ON c.course_id = s.course_id
                    WHERE s.specialization='Data Science'
                """)
            elif "autonomous" in ql:
                c.execute("""
                    SELECT DISTINCT c.* FROM courses c
                    JOIN specializations s ON c.course_id = s.course_id
                    WHERE s.specialization='Autonomous Systems'
                """)
            else:
                c.execute("SELECT * FROM specializations")
            result["data"] = [dict(r) for r in c.fetchall()]
            result["message"] = "Specialization courses"
        elif any(kw in ql for kw in ["graduation","degree requirement","total unit","how many unit","33 unit","graduate","complete the"]):
            c.execute("SELECT * FROM degree_requirements")
            result["data"] = [dict(r) for r in c.fetchall()]
            result["message"] = "MSAI degree requirements (33 units total)"
        elif any(kw in ql for kw in ["gwar","writing requirement","294","200w"]):
            c.execute("SELECT * FROM courses WHERE category LIKE '%Writing%'")
            result["data"] = [dict(r) for r in c.fetchall()]
            result["message"] = "Graduate Writing Requirement (GWAR) options"
        elif any(kw in ql for kw in ["culminating","thesis","project","plan a","plan b","299","295"]):
            c.execute("SELECT * FROM courses WHERE category LIKE '%Culminating%' ORDER BY course_id")
            result["data"] = [dict(r) for r in c.fetchall()]
            result["message"] = "Culminating experience options (Plan A & Plan B)"
        elif any(kw in ql for kw in ["all course","list course","every course","available course"]):
            c.execute("SELECT course_id,title,units,category,area,semester_offered FROM courses ORDER BY category,course_id")
            result["data"] = [dict(r) for r in c.fetchall()]
            result["message"] = "All MSAI courses"
        else:
            m = re.search(r'(cmpe|ise|engr)\s*(\d+[a-z]?)', ql)
            if m:
                cid = (m.group(1).upper() + " " + m.group(2)).strip()
                c.execute("SELECT * FROM courses WHERE UPPER(course_id)=?", (cid,))
                row = c.fetchone()
                if row:
                    result["data"] = [dict(row)]
                    result["message"] = f"Course info for {cid}"
                else:
                    result["message"] = f"{cid} not found in database."
            else:
                result["message"] = "Could not interpret query. Please specify a course ID or topic."
    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)
    finally:
        conn.close()
    return result


def web_search(query: str) -> dict:
    from urllib.parse import urlparse
    ql = query.lower()
    url_map = {
        "deadline"    : "https://www.sjsu.edu/classes/calendar/index.php",
        "add drop"    : "https://www.sjsu.edu/classes/calendar/index.php",
        "calendar"    : "https://www.sjsu.edu/classes/calendar/index.php",
        "advising"    : "https://www.sjsu.edu/cmpe/about/contact.php",
        "office hour" : "https://www.sjsu.edu/cmpe/about/contact.php",
        "tuition"     : "https://www.sjsu.edu/bursar/tuition/",
        "fee"         : "https://www.sjsu.edu/bursar/tuition/",
        "admission"   : "https://www.sjsu.edu/admissions/",
    }
    target = next((url for kw, url in url_map.items() if kw in ql),
                  "https://www.sjsu.edu/cmpe/academic/ms-ai/")
    domain = urlparse(target).netloc
    if not any(d in domain for d in APPROVED_DOMAINS):
        return {"status":"error","data":"","message":f"Domain '{domain}' not approved.","source":""}
    FALLBACK = {
        "deadline" : "Add/Drop: typically end of Week 1. Check MySJSU or sjsu.edu/classes/calendar for exact dates.",
        "advising" : "CMPE Advising: cmpeadvising@sjsu.edu | Engineering Student Services, ENG 285.",
        "tuition"  : "Graduate tuition: ~$3,300/semester base + $270/unit for 7+ units. Verify at sjsu.edu/bursar.",
    }
    try:
        r = requests.get(target, headers={"User-Agent": "SJSU-MSAI-VA/1.0"}, timeout=8)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","nav","header","footer","aside"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())[:600]
        return {"status":"success","data":text,"message":f"Retrieved from {target}","source":target}
    except Exception as e:
        for kw, info in FALLBACK.items():
            if kw in ql:
                return {"status":"success","data":info,"message":"Using cached fallback","source":"cached"}
        return {"status":"error","data":"","message":str(e),"source":target}


TOOLS = {
    "query_database": {"fn": query_database},
    "web_search":     {"fn": web_search},
}

# ── Model loading ─────────────────────────────────────────────────────────────
print("Loading TinyLlama-1.1B-Chat …")
tinyllama_tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tinyllama_mdl = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32, device_map="cpu")
tinyllama_mdl.eval()
print("  TinyLlama ready.")

print("Loading Qwen2-0.5B-Instruct …")
qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
qwen_mdl = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct", torch_dtype=torch.float32, device_map="cpu")
qwen_mdl.eval()
print("  Qwen2 ready.")

# ── Prompts ───────────────────────────────────────────────────────────────────
TOOL_SCHEMA = """
You have access to these tools:
  query_database(query) – Look up courses, prerequisites, and degree requirements.
  web_search(query)     – Retrieve info from approved SJSU web pages (deadlines, advising).

To call a tool output EXACTLY this JSON on its own line:
  TOOL_CALL: {"tool": "<tool_name>", "query": "<your query>"}

NEVER fabricate course data – always use the database tool for academic information.
"""

PROGRAM_SUMMARY = """
SJSU MSAI Program (33 units total):
  Core (9 units)          : CMPE 252, CMPE 257, ISE 201
  Specialization (6 units): Data Science OR Autonomous Systems (pick ONE)
  Electives (9 units)     : ≥3 from Area A; rest from Area A or Area B
  GWAR (3 units)          : CMPE 294 (recommended) or ENGR 200W
  Culminating (6 units)   : Plan A – Thesis (299A+B) OR Plan B – Project (295A+B)
"""

META_SYS = f"""You are an expert academic advisor for the MSAI program at San José State University.
You help graduate students with course prerequisites, degree requirements, semester planning, and deadlines.
STRICT RULES:
  1. Always call query_database before answering about courses or requirements.
  2. Call web_search for questions about deadlines, policies, or advising.
  3. Never fabricate academic information.
  4. Be concise, accurate, and professionally warm.
{TOOL_SCHEMA}{PROGRAM_SUMMARY}"""

CHAIN_CLASSIFY_SYS = """Classify the student query into EXACTLY ONE category (output only the label):
  PREREQUISITE_CHECK | COURSE_INFO | DEGREE_REQUIREMENTS | SPECIALIZATION |
  ELECTIVE_PLANNING | CULMINATING_EXPERIENCE | DEADLINE_POLICY | GENERAL_ADVISING"""

CHAIN_RESPOND_SYS = f"""You are an SJSU MSAI academic advisor.
Query classified as: {{intent}}
Tool results: {{tool_data}}
Answer the student's question accurately using the tool data above.
{PROGRAM_SUMMARY}"""

REFLECT_SYS = f"""You are an SJSU MSAI academic advisor.
{TOOL_SCHEMA}{PROGRAM_SUMMARY}"""

REFLECT_CRITIQUE = """Review your response for: {query}
Initial response: {initial}
Tool data: {tool_data}
Write your FINAL, improved response (concise and accurate):"""

# ── Generation helpers ────────────────────────────────────────────────────────
def extract_tool_call(text):
    m = re.search(r'TOOL_CALL:\s*(\{[^}]+\})', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None

def run_tool(tc):
    name = tc.get("tool","")
    query = tc.get("query","")
    if name not in TOOLS:
        return json.dumps({"status":"error","message":f"Unknown tool '{name}'"})
    return json.dumps(TOOLS[name]["fn"](query), indent=2)

def _generate(model, tok, prompt, max_new_tokens=512):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.1, do_sample=True,
            pad_token_id=tok.eos_token_id, repetition_penalty=1.1)
    lat = time.time() - t0
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True).strip(), lat

def _fmt_tinyllama(messages, tok):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def _fmt_qwen(messages, tok):
    merged, sys_text = [], ""
    for m in messages:
        if m["role"] == "system":
            sys_text = m["content"]
        elif m["role"] == "user" and sys_text:
            merged.append({"role":"user","content": sys_text + "\n\n" + m["content"]})
            sys_text = ""
        else:
            merged.append(m)
    return tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)

def _fmt(model_name, messages, tok):
    return _fmt_tinyllama(messages, tok) if model_name == "tinyllama" else _fmt_qwen(messages, tok)

def _model_tok(model_name):
    return (tinyllama_mdl, tinyllama_tok) if model_name == "tinyllama" else (qwen_mdl, qwen_tok)

# ── Strategies ────────────────────────────────────────────────────────────────
def run_meta(query, model_name):
    mdl, tok = _model_tok(model_name)
    tool_calls, tool_results, total_lat = [], [], 0.0

    msgs = [{"role":"system","content":META_SYS}, {"role":"user","content":query}]
    r1, lat = _generate(mdl, tok, _fmt(model_name, msgs, tok))
    total_lat += lat

    tc = extract_tool_call(r1)
    if tc:
        tool_calls.append(tc)
        tool_str = run_tool(tc)
        tool_results.append(tool_str)
        msgs2 = [
            {"role":"system","content":META_SYS},
            {"role":"user","content":query},
            {"role":"assistant","content":r1},
            {"role":"user","content":f"Tool result:\n{tool_str}\n\nNow give your final answer."},
        ]
        r2, lat2 = _generate(mdl, tok, _fmt(model_name, msgs2, tok))
        total_lat += lat2
        response = r2
    else:
        tc_forced = {"tool":"query_database","query":query}
        tool_calls.append(tc_forced)
        tool_str = run_tool(tc_forced)
        tool_results.append(tool_str)
        msgs2 = [
            {"role":"system","content":META_SYS},
            {"role":"user","content":query},
            {"role":"user","content":f"Database result:\n{tool_str}\n\nAnswer the question."},
        ]
        r2, lat2 = _generate(mdl, tok, _fmt(model_name, msgs2, tok))
        total_lat += lat2
        response = r2

    return {"response": response, "tool_calls": tool_calls, "tool_results": tool_results, "latency": round(total_lat, 2)}

def run_chain(query, model_name):
    mdl, tok = _model_tok(model_name)
    tool_calls, tool_results, total_lat = [], [], 0.0

    cls_msgs = [{"role":"system","content":CHAIN_CLASSIFY_SYS}, {"role":"user","content":query}]
    intent_raw, lat = _generate(mdl, tok, _fmt(model_name, cls_msgs, tok), max_new_tokens=20)
    total_lat += lat
    intent = intent_raw.strip().split("\n")[0].strip().upper()

    tool_name = "web_search" if intent == "DEADLINE_POLICY" else "query_database"
    tc = {"tool": tool_name, "query": query}
    tool_calls.append(tc)
    tool_str = run_tool(tc)
    tool_results.append(tool_str)

    respond_sys = CHAIN_RESPOND_SYS.format(intent=intent, tool_data=tool_str)
    rsp_msgs = [{"role":"system","content":respond_sys}, {"role":"user","content":query}]
    r3, lat3 = _generate(mdl, tok, _fmt(model_name, rsp_msgs, tok))
    total_lat += lat3

    return {"response": r3, "tool_calls": tool_calls, "tool_results": tool_results,
            "intent": intent, "latency": round(total_lat, 2)}

def run_reflect(query, model_name):
    mdl, tok = _model_tok(model_name)
    tool_calls, tool_results, total_lat = [], [], 0.0

    msgs1 = [{"role":"system","content":REFLECT_SYS}, {"role":"user","content":query}]
    init, lat1 = _generate(mdl, tok, _fmt(model_name, msgs1, tok))
    total_lat += lat1

    tc = extract_tool_call(init) or {"tool":"query_database","query":query}
    tool_calls.append(tc)
    tool_str = run_tool(tc)
    tool_results.append(tool_str)

    critique_text = REFLECT_CRITIQUE.format(query=query, initial=init, tool_data=tool_str)
    msgs2 = [{"role":"system","content":REFLECT_SYS}, {"role":"user","content":critique_text}]
    final, lat2 = _generate(mdl, tok, _fmt(model_name, msgs2, tok))
    total_lat += lat2

    return {"response": final, "tool_calls": tool_calls, "tool_results": tool_results,
            "initial_response": init, "latency": round(total_lat, 2)}

STRATEGY_FNS = {"meta": run_meta, "chain": run_chain, "reflect": run_reflect}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_query  = data.get("query", "").strip()
    model_name  = data.get("model", "tinyllama")
    strategy    = data.get("strategy", "meta")

    if not user_query:
        return jsonify({"error": "Query is empty."}), 400
    if model_name not in ("tinyllama", "qwen"):
        return jsonify({"error": "Invalid model."}), 400
    if strategy not in STRATEGY_FNS:
        return jsonify({"error": "Invalid strategy."}), 400

    result = STRATEGY_FNS[strategy](user_query, model_name)
    result["model"] = model_name
    result["strategy"] = strategy
    return jsonify(result)

if __name__ == "__main__":
    print("\n SJSU MSAI Virtual Advisor running at http://localhost:8080\n")
    app.run(debug=False, port=8080)
    # app.run(debug=False, port=5000)

