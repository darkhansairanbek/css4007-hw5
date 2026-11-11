
import json
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

with open("data/helpdesk.json", "r", encoding="utf-8") as file:
    helpdesk = json.load(file)


class HelpdeskState(TypedDict):
    message: str
    category: str
    reply: str


def read_request(state: HelpdeskState):
    message = state["message"].lower()

    has_wifi = "wifi" in message or "wi-fi" in message
    has_password = "password" in message

    if has_wifi and has_password:
        category = "both"
    elif has_wifi:
        category = "wifi"
    elif has_password:
        category = "password"
    else:
        category = "other"

    return {"category": category}


def wifi_help(state: HelpdeskState):
    instructions = helpdesk["instructions"]["wifi"]
    reply = instructions["title"] + "\n"
    reply += "\n".join(instructions["steps"])
    return {"reply": reply}


def password_help(state: HelpdeskState):
    instructions = helpdesk["instructions"]["password"]
    reply = instructions["title"] + "\n"
    reply += "\n".join(instructions["steps"])
    return {"reply": reply}


def other_help(state: HelpdeskState):
    return {"reply": "Is your problem related to Wi-Fi or your password?"}


def clarify_problem(state: HelpdeskState):
    return {
        "reply": "You mentioned both Wi-Fi and password problems. Which one would you like to solve first?"
    }


def choose_next_node(state: HelpdeskState):
    return state["category"]


builder = StateGraph(HelpdeskState)

builder.add_node("read_request", read_request)
builder.add_node("wifi_help", wifi_help)
builder.add_node("password_help", password_help)
builder.add_node("other_help", other_help)
builder.add_node("clarify_problem", clarify_problem)

builder.add_edge(START, "read_request")

builder.add_conditional_edges(
    "read_request",
    choose_next_node,
    {
        "wifi": "wifi_help",
        "password": "password_help",
        "other": "other_help",
        "both": "clarify_problem"
    }
)

builder.add_edge("wifi_help", END)
builder.add_edge("password_help", END)
builder.add_edge("other_help", END)
builder.add_edge("clarify_problem", END)

graph = builder.compile()


if __name__ == "__main__":
    message = input("Describe your IT problem: ")

    initial_state = {
        "message": message,
        "category": "",
        "reply": ""
    }

    current_state = initial_state.copy()

    print("\nInitial state:")
    print(current_state)

    print("\nExecution trace:")

    for update in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_update in update.items():
            current_state.update(node_update)

            print("\nExecuted node:", node_name)
            print("Node update:", node_update)
            print("Full state after node:")
            print(current_state)

    print("\nFinal reply:")
    print(current_state["reply"])
