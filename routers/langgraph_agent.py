import operator
from typing import TypedDict, Annotated

from fastapi.routing import APIRouter
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
import stripe
from starlette.responses import RedirectResponse
from fastapi import Request


from config import BaseConfig

agent_router = APIRouter()
settings = BaseConfig()
stripe.api_key = 'sk_test_51QdJ6YPsHVuECJNRZqdnGKziIrQ6TP00h2D0knFskqzgHfIyoCPOEG3SMQUMey2zNCxTEdm2DsSSvO1Kk2CN3CpZ00n5MUMEDj'

'''
langgraph agent tutorial:
'''
class State(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

'''
Tools 
'''

@tool("Web Search")
def web_search(user_query: str):
    """
    Useful to search for results from the web. Use when the user needs
    results not in the original training context
    """
    search = DuckDuckGoSearchResults()
    search_results = search.invoke(user_query, max_results=1,output_format=list)

    return search_results

@tool("RAG")
def rag(user_query: str):
    """
    Useful for finding key facts and information from the user contracts
    """
    # TODO: Check if tools need to return the o/p in a specific format - don't think so tbh
    return "RAG Results"

def process_stripe_payment():
    product_id = create_stripe_payment_product("test shrey loan")
    price_id = create_stripe_payment_price(product_id, 250)
    return generate_stripe_payment_link(price_id)

def create_stripe_payment_product(name: str, description: str = "loan info"):
    product = stripe.Product.create(name=name, description=description)
    return product.id

def create_stripe_payment_price(product_id: str, amount: int):
    price = stripe.Price.create(  product= product_id, unit_amount= amount, currency="usd")
    return price.id

def generate_stripe_payment_link(price_id: str):
    """
    Useful for processing payments
    """
    try:
        checkout_session = stripe.checkout.Session.create(
            line_items=[
                {
                    # Provide the exact Price ID (for example, pr_1234) of the product you want to sell
                    'price': price_id,
                    'quantity': 1,
                },
            ],
            mode='payment',
            success_url="http://localhost:5173/" + '?success=true',
            cancel_url= "http://localhost:5173/" + '?canceled=true',
        )
        return checkout_session.url
    except Exception as e:
        return str(e)



@agent_router.get("/langgraph/tools/test/{user_query}")
def langraph_tools_test(user_query: str):
    return RedirectResponse(url = process_stripe_payment())


@agent_router.get("/langgraph/chat")
def langgraph_agent_chat():
    # Add state, each node receives the state as the i/p and the o/p can update the state
    # When defining the State, we define the vars that we need to track in the state
    graph_builder = StateGraph(State)
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        huggingfacehub_api_token=settings.HF_ACCESS_TOKEN,
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03
    )

    memory = MemorySaver()

    def format_user_output(llm_output: str):
        """
        Always use this when returning the result to the user
        """
        return "The output from the LLM: \n\n ---> " + llm_output + " Your welcome !"

    # TODO: See how multiple I/P can be passed
    tools = [rag, web_search]
    #tool.invoke("What's a 'node' in LangGraph?")

    llm = ChatHuggingFace(llm=llm_endpoint)
    llm_with_tools = llm.bind_tools(tools=tools) # TODO: figure out tool_choice='auto'/'any' equivalent for HF Chat Models

    def chatbot(state: State):
        return {"chat_history": [llm_with_tools.invoke(state["chat_history"])]}

    # The first argument is the unique node name
    # The second argument is the function or object that will be called whenever
    # the node is used.
    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    graph = graph_builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "1"}}

    user_input = "Hi there! My name is Will."

    # # The config is the **second positional argument** to stream() or invoke()!
    # events = graph.stream(
    #     {"chat_history": [("user", user_input)]}, config, stream_mode="values"
    # )
    # for event in events:
    #     event["chat_history"][-1].pretty_print()

    # Chat functionality
    # TODO: To check how the model can be stopped for relevant Human Inputs
    def stream_graph_updates(user_input: str):
        for event in graph.stream({"chat_history": [("user", user_input)]}, config):
            for value in event.values():
                print("Assistant:", value["chat_history"][-1].content)


    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break

# TODO: Use the Oracle Set up
# TODO: Use Fixed Egde to force the model to format the results before the END state
# class AgentState(TypedDict):
#     input: str
#     chat_history: list[BaseMessage]
#     intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
#
# import requests
# # our regex
# abstract_pattern = re.compile(
#     r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
#     re.DOTALL
# )
# # @tool("fetch_arxiv")
# # def fetch_arxiv(arxiv_id: str):
# #     """Gets the abstract from an ArXiv paper given the arxiv ID. Useful for
# #     finding high-level context about a specific paper."""
# #     # get paper page in html
# #     res = requests.get(
# #         f"https://export.arxiv.org/abs/{arxiv_id}"
# #     )
# #     # search html for abstract
# #     re_match = abstract_pattern.search(res.text)
# #     # return abstract text
# #     return re_match.group(1)
#
# @tool("format_string")
# def format_string(input_string: str):
#     """
#     Given a User Input String, it formats it and returns it.
#     Useful for returning response to the user
#     """
#     return "Your cute lil string is : " + input_string + "\n You're welcome !"
#
# @tool("final_answer")
# def final_answer(
#     introduction: str,
#     research_steps: str,
#     main_body: str,
#     conclusion: str,
#     sources: str
# ):
#     """Returns a natural language response to the user in the form of a research
#     report. There are several sections to this report, those are:
#     - `introduction`: a short paragraph introducing the user's question and the
#     topic we are researching.
#     - `research_steps`: a few bullet points explaining the steps that were taken
#     to research your report.
#     - `main_body`: this is where the bulk of high quality and concise
#     information that answers the user's question belongs. It is 3-4 paragraphs
#     long in length.
#     - `conclusion`: this is a short single paragraph conclusion providing a
#     concise but sophisticated view on what was found.
#     - `sources`: a bulletpoint list provided detailed sources for all information
#     referenced during the research process
#     """
#     if type(research_steps) is list:
#         research_steps = "\n".join([f"- {r}" for r in research_steps])
#     if type(sources) is list:
#         sources = "\n".join([f"- {s}" for s in sources])
#     return ""
#
# # define a function to transform intermediate_steps from list
# # of AgentAction to scratchpad string
# def create_scratchpad(intermediate_steps: list[AgentAction]):
#     research_steps = []
#     for i, action in enumerate(intermediate_steps):
#         if action.log != "TBD":
#             # this was the ToolExecution
#             research_steps.append(
#                 f"Tool: {action.tool}, input: {action.tool_input}\n"
#                 f"Output: {action.log}"
#             )
#     return "\n---\n".join(research_steps)
#
# @app.get("/agent")
# async def use_langgraph_agent():
#     system_prompt = """You are the oracle, the great AI decision maker.
#     Given the user's query you must decide what to do with it based on the
#     list of tools provided to you.
#
#     If you see that a tool has been used (in the scratchpad) with a particular
#     query, do NOT use that same tool with the same query again. Also, do NOT use
#     any tool more than twice (ie, if the tool appears in the scratchpad twice, do
#     not use it again).
#
#     You should aim to collect information from a diverse range of sources before
#     providing the answer to the user. Once you have collected plenty of information
#     to answer the user's question (stored in the scratchpad) use the final_answer
#     tool."""
#
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("user", "{input}"),
#         ("assistant", "scratchpad: {scratchpad}"),
#     ])
#
#     tools = [
#         format_string,
#         final_answer
#     ]
#
#     llm = HuggingFaceEndpoint(
#         repo_id="HuggingFaceH4/zephyr-7b-beta",
#         task="text-generation",
#         max_new_tokens=512,
#         do_sample=False,
#         repetition_penalty=1.03,
#         huggingfacehub_api_token=settings.HF_ACCESS_TOKEN
#     )
#
#     # # Authenticate to Hugging Face and access the model
#     # llm = InferenceClient(
#     #     "mistralai/Mistral-7B-Instruct-v0.3",
#     #     token= settings.HF_ACCESS_TOKEN)
#     # Reference: https://python.langchain.com/docs/integrations/chat/huggingface/
#     chat_model = ChatHuggingFace(llm=llm)
#
#
#     oracle = (
#             {
#                 "input": lambda x: x["input"],
#                 "chat_history": lambda x: x["chat_history"],
#                 "scratchpad": lambda x: create_scratchpad(
#                     intermediate_steps=x["intermediate_steps"]
#                 ),
#             }
#             | prompt
#             | chat_model.bind_tools(tools=tools, tool_choice="any") # TODO: Fix using the tool_choice
#     )
#     inputs = {
#         "input": "tell me something interesting about dogs",
#         "chat_history": [],
#         "intermediate_steps": [],
#     }
#     out = oracle.invoke(inputs)
#
#     return {"Response: ": out}
