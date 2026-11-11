
import json
import os
from pathlib import Path
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages


load_dotenv()


# Load fictional helpdesk data
with open("data/helpdesk.json", "r", encoding="utf-8") as file:
    helpdesk = json.load(file)

with open("data/service_status.json", "r", encoding="utf-8") as file:
    service_data = json.load(file)


# Select service-status snapshot
STATUS_PROFILE = os.getenv("STATUS_PROFILE", "normal")

# Maximum number of tool rounds
MAX_TOOL_STEPS = 5


# --------------------------------------------------
# Tools
# --------------------------------------------------

@tool
def find_instructions(topic: str) -> str:
    """Find instructions for wifi or password recovery."""

    if topic not in helpdesk["instructions"]:
        return "Unknown topic. Use wifi or password."

    instructions = helpdesk["instructions"][topic]

    return (
        instructions["title"]
        + "\n"
        + "\n".join(instructions["steps"])
    )


@tool
def check_service_status(service: str) -> str:
    """Check wifi or account_recovery status."""

    profile = service_data["profiles"].get(STATUS_PROFILE)

    if profile is None:
        return "Unknown status profile."

    result = profile.get(service)

    if result is None:
        return "Unknown service."

    return json.dumps(result, ensure_ascii=False)


@tool
def prepare_ticket(
    problem_description: str,
    steps_tried: str
) -> str:
    """Create a local support ticket draft."""

    if not problem_description.strip() or not steps_tried.strip():
        return (
            "Missing information. Ask the user for the missing details."
        )

    ticket = {
        "problem_description": problem_description,
        "steps_tried": steps_tried,
        "delivery": "local draft only"
    }

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    ticket_path = output_dir / "ticket_draft.json"

    with open(ticket_path, "w", encoding="utf-8") as file:
        json.dump(ticket, file, ensure_ascii=False, indent=4)

    return f"Ticket draft saved locally: {ticket_path}"


tools = [
    find_instructions,
    check_service_status,
    prepare_ticket
]


# --------------------------------------------------
# Language model
# --------------------------------------------------

model = ChatOpenAI(
    model=os.getenv("MODEL_NAME")
)

model_with_tools = model.bind_tools(tools)


# --------------------------------------------------
# Graph state
# --------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_steps: int


# --------------------------------------------------
# Assistant node
# --------------------------------------------------

def assistant(state: AgentState):

    system_message = SystemMessage(
        content=(
            "You are a fictional university IT helpdesk assistant. "

            "Use the supplied tools to answer user requests. "

            "For Wi-Fi problems, use service wifi. "
            "For password recovery problems, use service account_recovery. "

            "Use find_instructions to retrieve the relevant instructions. "
            "Use check_service_status to check the requested service. "

            "If the user mentions both Wi-Fi and password recovery, "
            "handle both problems separately and check both services. "

            "If status is outage or down, inform the user about the outage. "
            "Do not prepare a ticket automatically during an outage. "

            "If status is unknown, clearly explain that the service "
            "status cannot be confirmed. Never invent a repair time. "

            "If status is operational, provide the relevant instructions. "
            "If the problem remains unresolved and the user requests "
            "a ticket, use prepare_ticket. "

            "Include only steps that the user actually tried. "
            "If the steps are unclear, do not invent them. "

            "A ticket is only a local draft. Never claim that it was "
            "sent to the university helpdesk. "

            "Never request or store passwords or recovery codes. "

            "Use only the supplied fictional helpdesk information. "

            "Respond to the user in clear, natural language. "
            "Do not show internal tool calls or technical reasoning."
        )
    )

    response = model_with_tools.invoke(
        [system_message] + state["messages"]
    )

    return {"messages": [response]}


# --------------------------------------------------
# Count tool rounds
# --------------------------------------------------

def run_tools(state: AgentState):

    return {
        "tool_steps": state.get("tool_steps", 0) + 1
    }


# --------------------------------------------------
# Handle the tool limit
# --------------------------------------------------

def limit_reached(state: AgentState):

    return {
        "messages": [
            (
                "assistant",
                "I could not complete the request within the tool-call "
                "limit. Please try again with more details."
            )
        ]
    }


# --------------------------------------------------
# Conditional routing
# --------------------------------------------------

def route_after_assistant(state: AgentState):

    last_message = state["messages"][-1]

    if last_message.tool_calls:

        if state.get("tool_steps", 0) < MAX_TOOL_STEPS:
            return "tools"

        return "limit"

    return END


# --------------------------------------------------
# Build LangGraph
# --------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("count_tools", run_tools)
builder.add_node("limit_reached", limit_reached)

builder.add_edge(START, "assistant")

builder.add_conditional_edges(
    "assistant",
    route_after_assistant,
    {
        "tools": "count_tools",
        "limit": "limit_reached",
        END: END
    }
)

builder.add_edge("count_tools", "tools")
builder.add_edge("tools", "assistant")
builder.add_edge("limit_reached", END)

graph = builder.compile()


# --------------------------------------------------
# Extract only normal text from model response
# --------------------------------------------------

def extract_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item["text"])

        return "\n".join(text_parts)

    return str(content)


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":

    print("University IT Helpdesk")
    print("Status profile:", STATUS_PROFILE)
    print("Type 'exit' to stop.\n")

    while True:

        try:
            user_message = input("You: ")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if user_message.strip().lower() == "exit":
            print("Goodbye!")
            break

        if not user_message.strip():
            continue

        result = graph.invoke(
            {
                "messages": [("user", user_message)],
                "tool_steps": 0
            },
            config={"recursion_limit": 20}
        )

        # Extract final answer
        final_message = result["messages"][-1]
        answer = extract_text(final_message.content)

        # Save execution log
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        log_path = output_dir / "part2_runs.jsonl"

        with open(log_path, "a", encoding="utf-8") as file:

            for message in result["messages"]:

                if hasattr(message, "tool_calls") and message.tool_calls:

                    for call in message.tool_calls:

                        record = {
                            "type": "tool_call",
                            "tool": call["name"],
                            "arguments": call["args"]
                        }

                        file.write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )

                elif getattr(message, "type", "") == "tool":

                    record = {
                        "type": "tool_result",
                        "tool": message.name,
                        "result": message.content
                    }

                    file.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

            final_record = {
                "type": "final_answer",
                "answer": answer
            }

            file.write(
                json.dumps(final_record, ensure_ascii=False) + "\n"
            )

        print("\nAssistant:", answer)
        print("Log saved:", log_path)
        print()
