# ResearchPilot Agent

ResearchPilot is an advanced, self-correcting AI research agent built using **LangGraph**. It combines a Plan-and-Execute architecture with dynamic web searching, local knowledge base retrieval, and a Python sandbox for robust analytical reasoning.

## Architecture

The agent runs as a structured graph with three main phases:

1. **Planner**: Evaluates complex research questions and breaks them down into 2-4 concrete sub-tasks.
2. **Executor**: Solves the plan sequentially by reasoning through tool calls. Uses tools like Tavily for web search, Qdrant for vector retrieval, and Python for data analysis.
3. **Critic**: Reviews the Executor's draft for completeness and accurate citations. If it detects unsupported claims or missing steps, it provides feedback and routes the graph back to the Executor for a revision (limited to 1 retry).

## Setup Instructions

1. **Python Virtual Environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment Variables (`.env`)**
   You need the following API keys in your `.env` file:
   ```env
   GROQ_API_KEY=your_key
   TAVILY_API_KEY=your_key
   GEMINI_API_KEY=your_key
   ```

3. **Start Qdrant Knowledge Base**
   Ensure your local Qdrant vector database is running on `localhost:6333`.
   ```bash
   docker start backend-qdrant-1
   ```

## Running the Application

ResearchPilot comes with a beautiful Streamlit UI. To start it, run:
```bash
streamlit run app.py
```
This will launch a web interface where you can enter questions and watch the agent's thought process unfold through collapsible traces.

## Known Limitations & Resilience

- **Free-Tier Limits**: This project is optimized for Groq's 6000 Tokens-Per-Minute free tier. The Executor minimizes overhead by resolving the entire plan continuously rather than spawning fresh agents per task.
- **LLM Formatting Hallucinations**: Because Groq models (like `llama-3.3-70b-versatile`) convert native XML tool tags into JSON on the backend, large text payloads occasionally cause the model to drop a bracket, resulting in a `400 BadRequestError`. 
  - **The Fix**: The application wraps the LLM calls in a resilient `try/except` block. When a 400 error occurs, it intercepts the crash, appends a simplified formatting note, and retries the generation. If it still fails, the Critic gracefully self-corrects!

See `demo_traces/plan_execute_self_correction_example.txt` for a complete example of the agent hitting an error, catching it, and successfully self-correcting.
