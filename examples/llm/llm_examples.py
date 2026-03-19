"""LLM usage examples

Contains example code for various LLM usage scenarios.
"""

from pathlib import Path

import yaml

from evomaster.agent import (
    Dialog,
    FunctionSpec,
    SystemMessage,
    ToolSpec,
    UserMessage,
)
from evomaster.utils.llm import LLMConfig, create_llm


def load_config():
    """Load configuration."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def example_simple_chat():
    """Example 1: Simple conversation"""
    print("\n" + "=" * 60)
    print("Example 1: Simple Conversation")
    print("=" * 60)

    # Load configuration
    config = load_config()
    llm_name = config["llm"]["default"]
    llm_config = LLMConfig(**config["llm"][llm_name])

    # Create LLM
    llm = create_llm(llm_config)
    print(f"Using LLM: {llm_config.model}")
    print(f"Base URL: {llm_config.base_url}")

    # Create conversation
    dialog = Dialog(
        messages=[
            SystemMessage(content="你是一个友好的助手。"),
            UserMessage(content="你好，请用一句话介绍一下你自己。"),
        ]
    )

    # Call LLM
    print("\nSending request...")
    response = llm.query(dialog)

    print(f"\nAssistant reply:\n{response.content}")

    # View metadata
    if response.meta:
        print(f"\nMetadata:")
        print(f"  Model: {response.meta.get('model', 'N/A')}")
        if "usage" in response.meta:
            usage = response.meta["usage"]
            print(f"  Input tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"  Output tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")


def example_multi_turn_chat():
    """Example 2: Multi-turn conversation"""
    print("\n" + "=" * 60)
    print("Example 2: Multi-turn Conversation")
    print("=" * 60)

    # Load configuration
    config = load_config()
    llm_name = config["llm"]["default"]
    llm_config = LLMConfig(**config["llm"][llm_name])

    # Create LLM
    llm = create_llm(llm_config)
    print(f"Using LLM: {llm_config.model}")

    # Initialize conversation
    dialog = Dialog(
        messages=[
            SystemMessage(content="你是一个数学助手，擅长解答数学问题。"),
        ]
    )

    # First turn
    print("\nFirst turn:")
    user_msg_1 = "请帮我计算 15 的平方根，保留两位小数。"
    print(f"  User: {user_msg_1}")

    dialog.add_message(UserMessage(content=user_msg_1))
    response_1 = llm.query(dialog)
    dialog.add_message(response_1)

    print(f"  Assistant: {response_1.content}")

    # Second turn (depends on previous context)
    print("\nSecond turn:")
    user_msg_2 = "那这个数字的 3 次方是多少？"
    print(f"  User: {user_msg_2}")

    dialog.add_message(UserMessage(content=user_msg_2))
    response_2 = llm.query(dialog)
    dialog.add_message(response_2)

    print(f"  Assistant: {response_2.content}")

    # Third turn
    print("\nThird turn:")
    user_msg_3 = "总结一下我们刚才的计算过程。"
    print(f"  User: {user_msg_3}")

    dialog.add_message(UserMessage(content=user_msg_3))
    response_3 = llm.query(dialog)
    dialog.add_message(response_3)

    print(f"  Assistant: {response_3.content}")


def example_tool_calling():
    """Example 3: Tool calling"""
    print("\n" + "=" * 60)
    print("Example 3: Tool Calling")
    print("=" * 60)

    # Load configuration
    config = load_config()
    llm_name = config["llm"]["default"]
    llm_config = LLMConfig(**config["llm"][llm_name])

    # Create LLM
    llm = create_llm(llm_config)
    print(f"Using LLM: {llm_config.model}")

    # Define tools
    tools = [
        ToolSpec(
            type="function",
            function=FunctionSpec(
                name="get_weather",
                description="获取指定城市的天气信息",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "城市名称，例如：北京、上海",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "温度单位",
                        },
                    },
                    "required": ["location"],
                },
            ),
        ),
        ToolSpec(
            type="function",
            function=FunctionSpec(
                name="calculate",
                description="执行数学计算",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "数学表达式，例如：2+2, sqrt(16)",
                        }
                    },
                    "required": ["expression"],
                },
            ),
        ),
    ]

    # Create conversation
    dialog = Dialog(
        messages=[
            SystemMessage(content="你是一个助手，可以使用工具来帮助用户。"),
            UserMessage(content="北京今天天气怎么样？温度用摄氏度表示。"),
        ],
        tools=tools,
    )

    # Call LLM
    print("\nSending request...")
    response = llm.query(dialog)

    print(f"\nAssistant reply: {response.content if response.content else '(no text reply)'}")

    # Check tool calls
    if response.tool_calls:
        print(f"\nTool calls:")
        for tool_call in response.tool_calls:
            print(f"  Tool: {tool_call.function.name}")
            print(f"  Arguments: {tool_call.function.arguments}")

        # Simulate tool execution result
        print("\nSimulated tool execution result:")
        tool_result = {
            "location": "北京",
            "temperature": 15,
            "unit": "celsius",
            "condition": "晴天",
        }
        print(f"  {tool_result}")


def example_different_models():
    """Example 4: Using different models"""
    print("\n" + "=" * 60)
    print("Example 4: Using Different Models")
    print("=" * 60)

    # Load configuration
    config = load_config()

    # Test question
    question = "用一句话解释什么是机器学习。"

    # Iterate over all configured models
    for model_name, model_config in config["llm"].items():
        if model_name == "default":
            continue

        print(f"\nUsing model: {model_name}")
        print("-" * 40)

        try:
            # Create LLM configuration
            llm_config = LLMConfig(**model_config)
            llm = create_llm(llm_config)

            # Create conversation
            dialog = Dialog(
                messages=[
                    SystemMessage(content="你是一个简洁的助手。"),
                    UserMessage(content=question),
                ]
            )

            # Query
            response = llm.query(dialog)
            print(f"Reply: {response.content}")

        except Exception as e:
            print(f"Error: {e}")


def example_streaming():
    """Example 5: Streaming output (if supported)"""
    print("\n" + "=" * 60)
    print("Example 5: Streaming Output")
    print("=" * 60)

    # Load configuration
    config = load_config()
    llm_name = config["llm"]["default"]
    llm_config = LLMConfig(**config["llm"][llm_name])

    # Create LLM
    llm = create_llm(llm_config)
    print(f"Using LLM: {llm_config.model}")

    # Create conversation
    dialog = Dialog(
        messages=[
            SystemMessage(content="你是一个助手。"),
            UserMessage(content="请写一首关于春天的短诗，四行即可。"),
        ]
    )

    print("\nGenerating...")
    print("-" * 40)

    # Check if streaming output is supported
    if hasattr(llm, "stream"):
        # Streaming output
        for chunk in llm.stream(dialog):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print()
    else:
        # Normal output
        response = llm.query(dialog)
        print(response.content)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LLM Usage Examples")
    print("=" * 60)

    # Run each example
    example_simple_chat()
    example_multi_turn_chat()
    example_tool_calling()
    example_different_models()
    example_streaming()

    print("\n" + "=" * 60)
    print("All examples completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
