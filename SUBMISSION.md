# HW5 — My LangGraph helpdesk

## Part 1 — How the graph works

### My graph and state

My graph has five nodes:

- `read_request` checks the user's message and detects the problem category.
- `wifi_help` provides Wi-Fi instructions from `helpdesk.json`.
- `password_help` provides password instructions from `helpdesk.json`.
- `other_help` asks whether the problem is related to Wi-Fi or a password.
- `clarify_problem` asks the user to choose when both topics are mentioned.

The state contains three fields:

- `message` — the original user request.
- `category` — the detected problem category.
- `reply` — the final response.

The graph starts at `read_request`. The function `choose_next_node` reads the category and selects the next node. Each help node updates the `reply` field, and then the graph reaches `END`.

**Initial graph:**

```text
START
  |
read_request
  |
  +-- wifi ------> wifi_help --------+
  |                                  |
  +-- password --> password_help ----+--> END
  |                                  |
  +-- other -----> other_help -------+
  |                                  |
  +-- both ------> clarify_problem --+
```

### Predictions before running

| Message | Predicted route | Actual route | What I learned |
|---|---|---|---|
| I cannot connect to Wi-Fi. | wifi_help | wifi_help | Wi-Fi keywords are detected. |
| I forgot my PASSWORD. | password_help | password_help | Detection is case-insensitive. |
| My laptop screen is broken. | other_help | other_help | Unsupported requests go to the other node. |
| My Wi-Fi password does not work. | clarify_problem | clarify_problem | Both topics are detected, so the graph asks for clarification. |

### My change to the graph

I added the `clarify_problem` node to handle requests that mention both Wi-Fi and password problems.

Before this change, the graph did not have a separate clarification node. Now, `read_request` assigns the category `both`, and `choose_next_node` routes the request to `clarify_problem`.

**New graph:**

```text
START
  |
read_request
  |
  +-- wifi ------> wifi_help --------+
  |                                  |
  +-- password --> password_help ----+--> END
  |                                  |
  +-- other -----> other_help -------+
  |                                  |
  +-- both ------> clarify_problem --+
```

For example, the message "My Wi-Fi password does not work" now produces the category `both` and asks the user which problem to solve first.

LangGraph controls the execution order and transitions between nodes. My Python functions decide how to classify the message and what response to return.









## Part 2 — Model and tools

### Model and tools

I used `ChatOpenAI` with the model name configured through `MODEL_NAME` in the `.env` file. The model provider is OpenAI.

The assistant has three tools:

- `find_instructions(topic)` — retrieves Wi-Fi or password recovery instructions from `helpdesk.json`.
- `check_service_status(service)` — checks Wi-Fi or account recovery status using `service_status.json`.
- `prepare_ticket(problem_description, steps_tried)` — creates a local JSON ticket draft in the `outputs` folder.

### Model/tool loop

```text
START
  |
assistant (LLM)
  |
  +-- No tool calls ------> END
  |
  +-- Tool calls
        |
   Check tool limit
        |
   count_tools
        |
      tools
        |
    assistant
        |
   Repeat or END
```

The model receives the system instructions and the user's message. It decides whether tools are needed. LangGraph routes tool calls to `ToolNode`, which executes the selected tools. Their results are added to the conversation, and the assistant receives them to prepare its answer.

### Direct tool calls and actual run

The tools can be called directly through the model:

- `find_instructions` accepts `wifi` or `password`.
- `check_service_status` accepts `wifi` or `account_recovery`.
- `prepare_ticket` requires a problem description and the troubleshooting steps already tried.

In my actual runs, the assistant used multiple tools in one request. For example, it checked Wi-Fi status and retrieved Wi-Fi instructions. In another run, it checked the service status and prepared a local ticket draft.

The execution records are saved in `outputs/part2_runs.jsonl`.

### Tool-call limit

I set `MAX_TOOL_STEPS = 5` to prevent the assistant from calling tools indefinitely. After five tool rounds, the graph routes to `limit_reached` and returns a message asking the user to provide more details.

The assistant also follows safety rules: it does not invent troubleshooting steps, does not claim that a local ticket was sent, and does not request passwords or recovery codes.







## Part 3 — Memory

I implemented two types of memory using SQLite.

### Conversation memory

I used LangGraph's `SqliteSaver` to save conversation history in `outputs/helpdesk_memory.sqlite`.

- `thread_id` identifies a conversation.
- The same `thread_id` allows the assistant to continue an earlier conversation.
- A new `thread_id` starts a separate conversation.

The `add_messages` reducer keeps new messages together with the existing conversation history.

### User preferences

I created a separate SQLite database: `outputs/user_memory.sqlite`.

It contains a `user_preferences` table with two fields:

- `user_id` — identifies the user.
- `language` — stores the user's preferred language.

The functions `get_user_language()` and `save_user_language()` retrieve and update this preference. The assistant checks the saved language before generating a response.

### Helpdesk data

The assistant retrieves troubleshooting instructions from `data/helpdesk.json` using `find_instructions`. Service information is loaded from `data/service_status.json` using `check_service_status`.

### Memory scenarios

| Situation | Command used | Log file | What happened |
|---|---|---|---|
| Refer to an earlier message | Continue using the same thread | `outputs/part3_runs.jsonl` | The assistant can use earlier messages from the current conversation. |
| Close, restart and resume | Restart `python part3.py` with the same IDs | `outputs/part3_runs.jsonl`, `outputs/helpdesk_memory.sqlite` | Conversation history is stored in SQLite and can be restored after restarting. |
| Same user, new conversation | Keep the same user ID and change the thread ID | `outputs/part3_runs.jsonl`, `outputs/user_memory.sqlite` | The language preference can remain available, while the new conversation has separate history. |
| Different user | Change the user ID | `outputs/part3_runs.jsonl`, `outputs/user_memory.sqlite` | A different user has a separate preference record. |

The language preference was initially added using a direct Python command. It was not automatically learned from a normal conversation.









## Part 4 — Running and checking my work

### Setup

The installation and running instructions are provided in `README.md`.

The project uses Python 3.13.15 and pinned package versions in `requirements.txt`:

- `langgraph==1.2.20`
- `langchain-core==1.7.7`
- `langchain-openai==1.7.4`
- `langgraph-checkpoint-sqlite==3.1.3`
- `python-dotenv==1.2.4`
- `pytest==9.2.0`

### Resetting memory

To reset stored memory, stop the program and delete:

- `outputs/helpdesk_memory.sqlite` — conversation history.
- `outputs/user_memory.sqlite` — saved user preferences.

The databases will be created again when the program runs.

### Tests

Run all tests with:

```powershell
python -m pytest -v
```

There are 9 tests:

| Test | Mistake it catches |
|---|---|
| User preferences are separate | One user's preference is assigned to another user. |
| Conversation survives restart | Conversation history is lost after restarting. |
| Wi-Fi routing | Wi-Fi requests go to the wrong node. |
| Both-problem routing | Messages mentioning Wi-Fi and password are not clarified. |
| Other-problem routing | Unknown requests are routed incorrectly. |
| Find instructions | The instruction tool returns incorrect information. |
| Check service status | The status tool does not return the expected result. |
| Prepare ticket | The ticket tool does not prepare the draft correctly. |
| Separate conversation histories | Different conversations share the same history. |

### Fake model

The Part 3 memory tests use a fake model. This allows testing memory and conversation behavior without making real OpenAI API calls.

The routing and direct tool tests also run without requiring a real model response.

### GitHub Actions

The workflow is located at `.github/workflows/tests.yml`. It runs automatically on every push and pull request. It installs the requirements and executes `python -m pytest -v`.
















## Choices and limitations

### My choices

- **LangGraph:** I chose it to control the workflow and manage transitions between the model and tools.
- **OpenAI:** I used an OpenAI model to understand user requests and decide which tools to call.
- **SQLite:** I used SQLite to save conversation history and user language preferences. It is lightweight and does not require a separate database server.
- **JSON files:** I used JSON files to store helpdesk instructions and service information.
- **Pytest:** I used it to check routing, tools, and memory behavior automatically.

### Limitations

- The model can sometimes misunderstand a request or choose an unnecessary tool.
- The service status depends on the available data and may not always be current.
- SQLite stores data locally, so memory is not automatically shared between different computers.
- The language preference is currently stored separately rather than always learned naturally during conversation.
- The ticket is prepared as a draft; it is not automatically submitted to a real support system.
- Tests with a fake model do not guarantee that the real OpenAI model will always behave correctly.

















### Supplied scenarios — my actual results

| Scenario | Command or manual steps | Log file | What happened / what I changed |
|---|---|---|---|
| S01 — Simple Wi-Fi route | `python part1.py` | No saved Part 1 log | The request was classified as `wifi` and routed to `wifi_help`. |
| S02 — Case-insensitive password route | `python part1.py` | No saved Part 1 log | The uppercase word PASSWORD was detected and routed to `password_help`. |
| S03 — Unsupported request | `python part1.py` | No saved Part 1 log | The laptop screen problem was classified as `other` and routed to `other_help`. |
| S04 — Two topics before the graph change | Manual test before the change | No saved Part 1 log | The original graph did not have a separate clarification node for requests containing both topics. |
| S05 — Two topics after adding clarification | `python part1.py` | No saved Part 1 log | The Wi-Fi and password request was classified as `both` and routed to `clarify_problem`. |
| S06 — Instructions then local ticket | `python part2.py` | `outputs/part2_runs.jsonl` | The service status was checked and a local ticket draft was created. However, the log for this run does not show a call to `find_instructions`, so the scenario was only partially completed. |
| S07 — The same request during a Wi-Fi outage | `python part2.py` | `outputs/part2_runs.jsonl` | The system detected a Wi-Fi outage and informed the user that no repair time was available. It did not prepare a ticket. |
| S08 — Status source unavailable | `python part2.py` | `outputs/part2_runs.jsonl` | The system retrieved Wi-Fi instructions but could not confirm the service status because the status source was unavailable. It did not claim that an outage was confirmed. |
| S09 — Missing ticket details | `python part2.py` | `outputs/part2_runs.jsonl` | A ticket draft was created. The system stated that the user had not specified the troubleshooting steps, so this information was left incomplete. |
| S10 — Use the current conversation | Manual test with the same conversation thread | `outputs/part3_runs.jsonl` | The system used the current conversation context and could refer to the earlier Wi-Fi discussion. |
| S11 — Persist and recall after restart | Restart the program and continue with the same thread | `outputs/part3_runs.jsonl`, `outputs/helpdesk_memory.sqlite` | The previous conversation was recalled after restarting the program. |
| S12 — Preference across threads without replay | Use the same user ID with a new thread ID | `outputs/part3_runs.jsonl`, `outputs/user_memory.sqlite` | The user's language preference remained available in a new thread, while the previous conversation history was not replayed. |
| S13 — Do not share memory between users | Use a different user ID and a new thread | `outputs/part3_runs.jsonl`, `outputs/user_memory.sqlite` | A different user did not receive the first user's saved language preference. |
| S14 — Password recovery outage | `python part2.py` | `outputs/part2_runs.jsonl` | The system detected that account recovery was unavailable and reported that no repair time was provided. However, the password instructions call used an unsupported topic and returned an error. |
