import pandas as pd
import google.generativeai as genai

# ---------- Load Questions ----------
def load_questions(file_path):
    try:
        df = pd.read_csv(file_path, quotechar='"')
        # Validate required columns
        if 'QuestionID' not in df.columns or 'Question' not in df.columns:
            print("❌ CSV file must contain 'QuestionID' and 'Question' columns.")
            exit(1)
        # Filter rows with valid QuestionID (e.g., Q1, S1)
        df = df[df['QuestionID'].str.match(r'^[QS]\d+$', na=False)]
        if df.empty:
            print("❌ No valid QuestionIDs found in CSV (must be like Q1, S1, etc.).")
            exit(1)
        # Ensure NextIfYes and NextIfNo are strings or empty
        df['NextIfYes'] = df['NextIfYes'].fillna('').astype(str)
        df['NextIfNo'] = df['NextIfNo'].fillna('').astype(str)
        return df
    except Exception as e:
        print("❌ Error parsing CSV. Make sure every row has the correct number of columns and questions with commas are quoted.")
        print(e)
        exit(1)

# ---------- Get Next Question from Gemini AI ----------
def get_next_question_from_gemini(history, remaining_questions, api_key, is_career=True):
    """
    Uses Gemini AI to select the most relevant next question from remaining CSV questions based on history.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    type_str = "career" if is_career else "skill"
    remaining_q_list = "\n".join([f"{i+1}. {q}" for i, q in enumerate(remaining_questions)])
    prompt = (
        f"You are a {type_str} advisor. Based on previous answers:\n" +
        "\n".join(f"- {q}: {a.upper()}" for q, a in history) +
        f"\nSelect the most relevant yes/no question from this list to narrow down {type_str} recommendations:\n" +
        remaining_q_list +
        "\nOutput only the selected question text."
    )
    
    try:
        response = model.generate_content(prompt)
        selected_question = response.text.strip()
        return selected_question if selected_question in remaining_questions else remaining_questions[0]
    except Exception as e:
        print(f"⚠️ Error selecting next question from Gemini: {e}")
        return remaining_questions[0] if remaining_questions else "Do you have any other interests?"

# ---------- Get Suggestion from Gemini AI ----------
def get_suggestion(history, api_key, is_career=True):
    """
    Uses Gemini AI to suggest a career or skill based on user answers.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    type_str = "career" if is_career else "skill"
    prompt = (
        f"Based on these yes/no answers:\n" +
        "\n".join(f"- {q}: {a.upper()}" for q, a in history) +
        f"\nSuggest a suitable {type_str}."
    )
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ Error fetching suggestion: {e}")
        return "N/A"

# ---------- Fetch Market Trend from Gemini AI ----------
def fetch_market_trend(suggestion, api_key):
    """
    Uses Gemini AI to fetch market-aware career/skill insights.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = (
        f"Analyze market trends for the {suggestion}. Provide:\n"
        "- Market Demand: high/medium/low with brief explanation\n"
        "- Top Skills: 3-5 skills\n"
        "- Expected Salary Range: entry-level to experienced, in USD\n"
        "- Aligns with Market Trends: yes/no with brief explanation"
    )
    
    try:
        response = model.generate_content(prompt)
        text = response.text
        
        data = {"Raw Output": text}
        keys = ["Market Demand", "Top Skills", "Expected Salary Range", "Aligns with Market Trends"]
        
        current_pos = 0
        for key in keys:
            start = text.find(key + ":", current_pos)
            if start != -1:
                end = text.find("\n\n", start) if "\n\n" in text[start:] else len(text)
                next_key_pos = len(text)
                for next_key in keys[keys.index(key) + 1:]:
                    next_pos = text.find(next_key + ":", start + 1)
                    if next_pos != -1:
                        next_key_pos = min(next_pos, next_key_pos)
                end = min(end, next_key_pos) if next_key_pos != len(text) else end
                value = text[start + len(key) + 1:end].strip()
                data[key] = value
                current_pos = end
            else:
                data[key] = "N/A"
        
        return data
    
    except Exception as e:
        return {
            "Market Demand": "N/A",
            "Top Skills": "N/A",
            "Expected Salary Range": "N/A",
            "Aligns with Market Trends": "N/A",
            "Raw Output": str(e)
        }

# ---------- Ask Questions ----------
def ask_questions(questions, api_key, start_id="Q1", min_questions=10, is_career=True):
    history = []  # List of (question, answer) tuples
    asked_ids = set()  # Track asked QuestionIDs
    remaining_questions = list(questions['Question'].unique())  # Available questions
    
    current_id = start_id
    asked_count = 0
    use_csv_branching = True  # Start with CSV branching
    
    while asked_count < min_questions:
        # Switch to Gemini after 5 questions if branching fails
        if asked_count >= 5:
            use_csv_branching = False
        
        if use_csv_branching:
            row_df = questions[questions["QuestionID"] == current_id]
            if row_df.empty:
                use_csv_branching = False
                question = get_next_question_from_gemini(history, remaining_questions, api_key, is_career)
                current_id = None  # Gemini-generated question
            else:
                row = row_df.iloc[0]
                question = row["Question"]
                next_yes = row.get("NextIfYes", "")
                next_no = row.get("NextIfNo", "")
                asked_ids.add(current_id)
        else:
            # Use Gemini to select next question
            if remaining_questions:
                question = get_next_question_from_gemini(history, remaining_questions, api_key, is_career)
                # Find corresponding ID if available
                matching_row = questions[questions["Question"] == question]
                current_id = matching_row.iloc[0]["QuestionID"] if not matching_row.empty else None
                if current_id:
                    asked_ids.add(current_id)
            else:
                print("\n⚠️ No more questions available.")
                break
        
        # Ask question
        answer = input(f"{question} (yes/no): ").strip().lower()
        if answer not in ['yes', 'no']:
            print("⚠️ Please answer 'yes' or 'no'. Using 'no' as default.")
            answer = 'no'
        
        # Append to history and remove from remaining
        history.append((question, answer))
        if question in remaining_questions:
            remaining_questions.remove(question)
        asked_count += 1
        
        # Determine next question for CSV branching
        if use_csv_branching and current_id:
            next_id = next_yes if answer == 'yes' else next_no
            # Validate next_id
            if next_id and not pd.isna(next_id) and next_id in questions["QuestionID"].values and next_id not in asked_ids:
                current_id = next_id
            else:
                use_csv_branching = False
                if remaining_questions:
                    question = get_next_question_from_gemini(history, remaining_questions, api_key, is_career)
                    matching_row = questions[questions["Question"] == question]
                    current_id = matching_row.iloc[0]["QuestionID"] if not matching_row.empty else None
                    if current_id:
                        asked_ids.add(current_id)
                else:
                    print("\n⚠️ No more questions available.")
                    break
        else:
            # Continue with Gemini if no branching or branching failed
            if remaining_questions:
                question = get_next_question_from_gemini(history, remaining_questions, api_key, is_career)
                matching_row = questions[questions["Question"] == question]
                current_id = matching_row.iloc[0]["QuestionID"] if not matching_row.empty else None
                if current_id:
                    asked_ids.add(current_id)
            else:
                print("\n⚠️ No more questions available.")
                break
    
    # Fetch suggestion from Gemini
    suggestion = get_suggestion(history, api_key, is_career)
    
    # Fetch market trend
    market_data = fetch_market_trend(suggestion, api_key)
    
    # Print user-friendly output
    type_str = "Career" if is_career else "Skill"
    print(f"\n✅ Recommended {type_str}: {suggestion}")
    print(f"\nMarket Insights:")
    print(f"  Demand: {market_data['Market Demand']}")
    print(f"  Top Skills: {market_data['Top Skills']}")
    print(f"  Salary Range (USD): {market_data['Expected Salary Range']}")
    print(f"  Market Alignment: {market_data['Aligns with Market Trends']}")
    print(f"\nDetails:\n{market_data['Raw Output']}")

# ---------- Main ----------
def main():
    print("Choose mode:\n1. Career Advisor\n2. Skills Advisor")
    choice = input("Enter 1 or 2: ").strip()
    
    api_key = input("Enter your Gemini API Key: ").strip()
    
    if choice == "1":
        questions_file = "data/career_questions_cleaned_fixed_updated.csv"
        start_id = "Q1"
        is_career = True
    elif choice == "2":
        questions_file = "data/skills_questions_fixed_updated_fixed_updated.csv"
        start_id = "S1"
        is_career = False
    else:
        print("❌ Invalid choice.")
        return
    
    questions = load_questions(questions_file)
    ask_questions(questions, api_key=api_key, start_id=start_id, min_questions=10, is_career=is_career)

if __name__ == "__main__":
    main()