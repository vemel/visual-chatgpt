import os
import sys
from typing import BinaryIO, Tuple

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import os
import re

import gradio as gr
from dotenv import load_dotenv
from langchain.agents.initialize import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from PIL import Image

from visual_chatgpt.tool_manager import ToolManager
from visual_chatgpt.tools.base import BaseTool
from visual_chatgpt.types import State

load_dotenv()

VISUAL_CHATGPT_PREFIX = """Visual ChatGPT is designed to be able to assist with a wide range of text and visual related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. Visual ChatGPT is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Visual ChatGPT is able to process and understand large amounts of text and images. As a language model, Visual ChatGPT can not directly read images, but it has a list of tools to finish different visual tasks. Each image will have a file name formed as "image/xxx.png", and Visual ChatGPT can invoke different tools to indirectly understand pictures. When talking about images, Visual ChatGPT is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new image files, Visual ChatGPT is also known that the image may not be the same as the user's demand, and will use other visual question answering tools or description tools to observe the real image. Visual ChatGPT is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the image content and image file name. It will remember to provide the file name from the last tool observation, if a new image is generated.

Human may provide new figures to Visual ChatGPT with a description. The description helps Visual ChatGPT to understand this image, but Visual ChatGPT should use tools to finish following tasks, rather than directly imagine from the description.

Overall, Visual ChatGPT is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 


TOOLS:
------

Visual ChatGPT  has access to the following tools:"""

VISUAL_CHATGPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

VISUAL_CHATGPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since Visual ChatGPT is a text language model, Visual ChatGPT must use tools to observe images rather than imagination.
The thoughts and observations are only visible for Visual ChatGPT, Visual ChatGPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad}"""


def cut_dialogue_history(history_memory: str, keep_last_n_words: int = 500) -> str:
    if len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"hitory_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    else:
        paragraphs = history_memory.split("\n")
        last_n_tokens = n_tokens
        while last_n_tokens >= keep_last_n_words:
            last_n_tokens = last_n_tokens - len(paragraphs[0].split(" "))
            paragraphs = paragraphs[1:]
        return "\n" + "\n".join(paragraphs)


class ConversationBot:
    def __init__(self) -> None:
        print("Initializing VisualChatGPT")
        self.llm = OpenAI(temperature=0)
        self.tool_manager = ToolManager()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", output_key="output"
        )
        self.tools = self.tool_manager.get_tools()
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={
                "prefix": VISUAL_CHATGPT_PREFIX,
                "format_instructions": VISUAL_CHATGPT_FORMAT_INSTRUCTIONS,
                "suffix": VISUAL_CHATGPT_SUFFIX,
            },
        )

    def run_text(self, text: str, state: State) -> Tuple[State, State]:
        print("===============Running run_text =============")
        print("Inputs:", text, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        assert self.agent.memory is not None
        self.agent.memory.buffer = cut_dialogue_history(
            self.agent.memory.buffer or "", keep_last_n_words=500
        )
        res = self.agent({"input": text})
        print("======>Current memory:\n %s" % self.agent.memory)
        response = re.sub(
            r"(image/\S*png)",
            lambda m: f"![](/file={m.group(0)})*{m.group(0)}*",
            res["output"],
        )
        state = state + [(text, response)]
        print("Outputs:", state)
        return state, state

    def run_image(
        self, image: BinaryIO, state: State, txt: str
    ) -> Tuple[State, State, str]:
        print("===============Running run_image =============")
        print("Inputs:", image, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        image_filename = BaseTool.get_image_filename()
        print("======>Auto Resize Image...")
        img = Image.open(image.name)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        img = img.resize((width_new, height_new))
        img = img.convert("RGB")
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        assert self.tool_manager.i2t is not None
        description = self.tool_manager.i2t.inference(image_filename)
        Human_prompt = (
            "\nHuman: provide a figure named {}. The description is: {}. This information helps you to understand this image, but you should use tools to finish following tasks, "
            'rather than directly imagine from my description. If you understand, say "Received". \n'.format(
                image_filename, description
            )
        )
        AI_prompt = "Received.  "
        self.agent.memory.buffer = (
            self.agent.memory.buffer + Human_prompt + "AI: " + AI_prompt
        )
        print("======>Current memory:\n %s" % self.agent.memory)
        state = state + [(f"![](/file={image_filename})*{image_filename}*", AI_prompt)]
        print("Outputs:", state)
        return state, state, txt + " " + image_filename + " "


def main() -> None:
    bot = ConversationBot()
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot", label="Visual ChatGPT")
        state = gr.State([])  # type: ignore
        with gr.Row():
            with gr.Column(scale=70):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter, or upload an image",
                ).style(container=False)
            with gr.Column(scale=15, min_width=0):
                clear = gr.Button("Clear️")
            with gr.Column(scale=15, min_width=0):
                btn = gr.UploadButton("Upload", file_types=["image"])

        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_image, [btn, state, txt], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == "__main__":
    main()
