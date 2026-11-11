
import part3


def test_user_preferences_are_separate(tmp_path, monkeypatch):
    # Используем временную базу данных, чтобы не менять основную.
    test_db = tmp_path / "test_user_memory.sqlite"
    monkeypatch.setattr(part3, "USER_MEMORY_DB", test_db)

    part3.initialize_user_memory()

    # Сохраняем язык для первого пользователя.
    part3.save_user_language("demo-user-a", "Kazakh")

    # Проверяем, что первый пользователь сохранил предпочтение.
    assert part3.get_user_language("demo-user-a") == "Kazakh"

    # У второго пользователя предпочтение не должно появиться.
    assert part3.get_user_language("demo-user-b") is None





from langchain_core.messages import AIMessage, HumanMessage


def test_conversation_memory_survives_restart(tmp_path, monkeypatch):
    # Подменяем модель, чтобы не обращаться к OpenAI.
    class FakeModel:
        def invoke(self, messages):
            user_messages = [
                message.content
                for message in messages
                if isinstance(message, HumanMessage)
            ]

            return AIMessage(
                content="History: " + " | ".join(user_messages)
            )

    monkeypatch.setattr(part3, "model_with_tools", FakeModel())

    # Используем отдельную временную базу данных.
    user_db = tmp_path / "test_user_memory.sqlite"
    conversation_db = tmp_path / "test_conversation.sqlite"

    monkeypatch.setattr(part3, "USER_MEMORY_DB", user_db)
    part3.initialize_user_memory()

    config = {
        "configurable": {
            "thread_id": "test-conversation"
        }
    }

    # Первый запуск программы.
    with part3.SqliteSaver.from_conn_string(
        str(conversation_db)
    ) as checkpointer:

        graph = part3.builder.compile(
            checkpointer=checkpointer
        )

        graph.invoke(
            {
                "messages": [
                    ("user", "My Wi-Fi does not work.")
                ],
                "tool_steps": 0,
                "user_id": "test-user"
            },
            config
        )

    # Здесь база закрыта. Имитируем перезапуск программы.

    # Второй запуск с той же базой и тем же thread_id.
    with part3.SqliteSaver.from_conn_string(
        str(conversation_db)
    ) as checkpointer:

        graph = part3.builder.compile(
            checkpointer=checkpointer
        )

        result = graph.invoke(
            {
                "messages": [
                    ("user", "What did I say before?")
                ]
            },
            config
        )

    # Проверяем, что модель получила оба сообщения.
    final_answer = result["messages"][-1].content

    assert "My Wi-Fi does not work." in final_answer
    assert "What did I say before?" in final_answer







import part1


def test_part1_detects_wifi():
    result = part1.read_request({
        "message": "My Wi-Fi does not work."
    })

    assert result["category"] == "wifi"


def test_part1_detects_both_problems():
    result = part1.read_request({
        "message": "My Wi-Fi and password do not work."
    })

    assert result["category"] == "both"


def test_part1_detects_other_problem():
    result = part1.read_request({
        "message": "My computer screen is broken."
    })

    assert result["category"] == "other"










import json
import part2


def test_find_instructions():
    result = part2.find_instructions.invoke({
        "topic": "wifi"
    })

    assert isinstance(result, str)
    assert len(result) > 0


def test_check_service_status(monkeypatch):
    monkeypatch.setattr(part2, "STATUS_PROFILE", "normal")

    result = part2.check_service_status.invoke({
        "service": "wifi"
    })

    data = json.loads(result)

    assert isinstance(data, dict)
    assert len(data) > 0


def test_prepare_ticket(tmp_path, monkeypatch):
    # Направляем черновик в отдельную временную папку.
    monkeypatch.chdir(tmp_path)

    result = part2.prepare_ticket.invoke({
        "problem_description": "Wi-Fi does not work.",
        "steps_tried": "Restarted the router."
    })

    ticket_path = tmp_path / "outputs" / "ticket_draft.json"

    assert ticket_path.exists()
    assert "local" in result.lower()

    with open(ticket_path, "r", encoding="utf-8") as file:
        ticket = json.load(file)

    assert ticket["problem_description"] == "Wi-Fi does not work."
    assert ticket["steps_tried"] == "Restarted the router."
    assert ticket["delivery"] == "local draft only"










def test_different_conversations_have_separate_history(tmp_path, monkeypatch):
    class FakeModel:
        def invoke(self, messages):
            user_messages = [
                message.content
                for message in messages
                if isinstance(message, HumanMessage)
            ]

            return AIMessage(
                content="History: " + " | ".join(user_messages)
            )

    monkeypatch.setattr(part3, "model_with_tools", FakeModel())

    user_db = tmp_path / "test_user_memory.sqlite"
    conversation_db = tmp_path / "test_conversations.sqlite"

    monkeypatch.setattr(part3, "USER_MEMORY_DB", user_db)
    part3.initialize_user_memory()

    with part3.SqliteSaver.from_conn_string(
        str(conversation_db)
    ) as checkpointer:

        graph = part3.builder.compile(
            checkpointer=checkpointer
        )

        # Первый разговор.
        graph.invoke(
            {
                "messages": [("user", "My Wi-Fi is broken.")],
                "tool_steps": 0,
                "user_id": "same-user"
            },
            {"configurable": {"thread_id": "conversation-A"}}
        )

        # Второй разговор того же пользователя.
        result = graph.invoke(
            {
                "messages": [("user", "I need help with my password.")],
                "tool_steps": 0,
                "user_id": "same-user"
            },
            {"configurable": {"thread_id": "conversation-B"}}
        )

    final_answer = result["messages"][-1].content

    assert "I need help with my password." in final_answer
    assert "My Wi-Fi is broken." not in final_answer




