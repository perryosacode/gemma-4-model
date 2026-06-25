"""Gemma default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# Gemma Instruction
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="gemma_instruction",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "<start_of_turn>user", "assistant": "<start_of_turn>model"},
        seps=["<end_of_turn>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<end_of_turn>"],
        stop_token_ids=[1, 107],
        system_prefix_token_ids=[2],
    )
)

# Gemma 3 Instruction. Same as gemma_instruction but with different stop token id
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="gemma3_instruction",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "<start_of_turn>user", "assistant": "<start_of_turn>model"},
        seps=["<end_of_turn>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<end_of_turn>"],
        stop_token_ids=[1, 106],
        system_prefix_token_ids=[2],
    )
)

# Gemma 4 Instruction.
# Gemma 4 uses a DIFFERENT chat format from Gemma 1/2/3: turns are delimited by the
# single tokens <|turn> (105) and <turn|> (106) — NOT <start_of_turn>/<end_of_turn>
# (which are not special tokens in the Gemma 4 tokenizer). With thinking disabled
# (the default), the model generation prompt emits an empty thought channel
# "<|channel>thought\n<channel|>" before the answer. The assembled generation prompt is:
#   <bos><|turn>user\n{msg}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
# (verified token-for-token against the checkpoint's apply_chat_template).
# eos = [1 <eos>, 106 <turn|>, 50 <|tool_response>] per generation_config.json.
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="gemma4_instruction",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="",
        roles={"user": "<|turn>user", "assistant": "<|turn>model"},
        seps=["<turn|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n<|channel>thought\n<channel|>",
        stop_str=["<turn|>"],
        stop_token_ids=[1, 106, 50],
        system_prefix_token_ids=[2],
    )
)
