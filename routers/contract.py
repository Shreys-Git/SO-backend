from fastapi import APIRouter
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from config import BaseConfig
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from operator import add

import asyncio
import operator
from typing_extensions import TypedDict
from typing import  Annotated, List, Optional, Literal
from pydantic import BaseModel, Field

from tavily import TavilyClient, AsyncTavilyClient

from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage, SystemMessage
from difflib import unified_diff

from langchain_core.runnables import RunnableConfig
from IPython.display import Markdown

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langsmith import traceable
from difflib import Differ
import re


contract_router = APIRouter()
settings = BaseConfig()

class UserPrompt(BaseModel):
    prompt: str

@contract_router.post("/langgraph/llm")
def langgraph_contract_agent(user_prompt: UserPrompt):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        max_retries=1,
        api_key=settings.OPENAI_SECRET_KEY
    )

    prompt = "You're a Contract Drafting agent. Goal is to draft a very good contract in great depth based on instruction : " + user_prompt.prompt
    response = llm.invoke(prompt)

    return {"llmContent" : response }

class EditInput(BaseModel):
    prompt: str
    agreement: str

class AIEdit(BaseModel):
  original_agreement_text: str = Field(
        description="Contains the exact original legal agreememt provided by the user",
    )
  updated_agreement_text: str = Field(
        description="Contains the updated response",
    )
  update_summary: str = Field(
        description="Summary of the changes made described using bullet points",
    )

class SectionState(TypedDict):
    section: AIEdit
    prompt: str
    agreement_text: str
    updated_agreement_text: str

# Magic Edit instructions
magic_edit_instructions="""You are an expert law analyst editing a legal agreement based on the instructions given to you.

Change Instructions:
{prompt}

Legal Agreement:
{agreement}

Guidelines for writing:

1. Technical Accuracy:
- Use technical terminology precisely

2. Length and Style:
- Keep the suggested changes word length similar to the original document
- No marketing language
- Technical focus
- Write in simple, clear language

3. Writing Approach:
- Use concrete details over general statements
- Make every word count
- No preamble prior to creating the section content
- Focus on your single most important point

4. Quality Checks:
- No preamble prior to creating the section content"""

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    max_retries=1,
    api_key=settings.OPENAI_SECRET_KEY,
)

def update_agreement(state: SectionState):
    """ Update a section of the report """

    # Get state
    agreement = state["agreement_text"]
    prompt = state["prompt"]

    # Format system instructions
    system_instructions = magic_edit_instructions.format(prompt=prompt, agreement=agreement)

    # Update section
    section_content = llm.with_structured_output(AIEdit).invoke([SystemMessage(content=system_instructions)]+[HumanMessage(content="Update the given documents based on the provided input")])
    return {"response": section_content}

def find_differences(original_text, updated_text):
    lines1 = original_text.splitlines()
    lines2 = updated_text.splitlines()

    # Create a Differ object and compare the lines
    differ = Differ()
    diff = differ.compare(lines1, lines2)

    formatted_diffs = []
    for line in diff:
        if not line.startswith("?"):
            formatted_diffs.append(line)
        if line.strip() == "":
            # Replace spaces after the first one with '\n'
            formatted_diffs.append(line[0] + re.sub(r' +', '\n', line[1:]))

    return formatted_diffs

@contract_router.post("/langgraph/magicEdit")
def langgraph_contract_agent(edit_input: EditInput):
    print("Agreement: \n\n", edit_input.agreement)

    updated_response = update_agreement({
    "agreement_text": edit_input.agreement,
    "prompt": edit_input.prompt
    })

    ai_response = updated_response["response"]
    original_agreement = ai_response.original_agreement_text
    updated_agreement = ai_response.updated_agreement_text
    update_summary = ai_response.update_summary

    differences = find_differences(original_agreement, updated_agreement)

    formatted_response = {
        "updated_agreement": updated_agreement,
        "update_summary": update_summary,
        "differences": differences,
    }

    return formatted_response


# Insights
class Insight(BaseModel):
    insight_type: Literal["clause", "obligation"] = Field(
        description="Describes whether a clause or an obligation is being extracted",
    )
    explanation: str = Field(
        description="Brief explanation of the field being extracted",
    )
    extraction: str = Field(
        description="Exact words extracted from the given document",
    )
    document_lookup: bool = Field(
        description="Whether more information is needed from the document to help research this section",
    )
    deviation: bool = Field(
        description="Whether this kind of text is expected in the document",
    )
    insight_generated: str = Field(
        description="Insight generated using a combination of web search result and optionally, more information from the document"
    )

class Insights(BaseModel):
    insights: List[Insight] = Field(
        description="Insights generated from the document",
    )

class Extraction(BaseModel):
    extraction: str = Field(description="Obligation or Clause wording extracted directly from the document")

class Extractions(BaseModel):
    extractions: List[Extraction] = Field(description="List of all the extractions from the document")

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )

class ReportState(TypedDict):
    topic: str  # Report topic
    insights: List[Insight]  # List of report insights
    completed_insights: List[Insight]  # Completed insights for the report
    report_insights_from_research: str  # String of any completed insights from research to write final insights
    final_report: str  # Final report
    number_of_queries: int
    tavily_topic: str
    tavily_days: str


# Prompt to extract the clauses/ obligation from the report
insight_extraction_instructions = """You are an expert legal researcher. You goal is, given a legal document,
extract all the {insight_type} from it.

While extracting the {insight_type} make sure:
1. Extract the exact words from the legal document, without changing them in any way.
2. Only present the extracted {insight_type} without any additional information or conversation.

The agreement to extact {insight_type} is:

{agreement}

"""

# Prompt to generate a search query to help with planning the report outline
report_planner_query_writer_instructions="""You are an expert lawyer, helping to extract intelligent
insights about the given {insight_type}.

The {insight_type} is:

{insight_extraction}

Your goal is to generate {number_of_queries} search queries that will help gather additional information for drawing insights about the above {insight_type}.

Make the query specific enough to find high-quality, relevant sources while covering the breadth needed for the report structure."""

# Prompt generating the report outline
report_planner_instructions="""You are an expert lawyer, helping to research {insight_type} in a legal document.

Your goal is, given all the {insight_type} in the document, create a high-quality report outline
which helps draw insights for each of the given {insight_type}.

The report should have a section dedicated to each of the extracted {insight_type} from the document.

Here are all the {insight_type}s you need to create an outline for:

{insight_extraction}

You should reflect on this information to plan the sections of the report:

{context}

Now, generate the sections of the report. Each section should have the following fields:

- insight_type - Type of {insight_type}.
- explanation - Brief overview of the main topics and concepts to be covered in this section.
- extraction - Exact original {insight_type} being researched.
- Document Lookup - Whether to look up for more information in the legal document.
- deviation - Whether this type of {insight_type} is expected
- insight_generated - The insights generated for the given {insight_type}, which you will leave blank for now.

Consider which sections require web research and which will require more data from the document."""

# ------------------------------------------------------------
# Utility functions

def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=True):
    """
    Takes either a single search response or list of responses from Tavily API and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.

    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results

    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response['results']
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")

    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source['url'] not in unique_sources:
            unique_sources[source['url']] = source

    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"

    return formatted_text.strip()

def format_insights(insights: list[Insight]) -> str:
    """ Format a list of insights into a string """
    formatted_str = ""
    for idx, insight in enumerate(insights, 1):
        formatted_str += f"""
        {'='*60}
        Insight {idx}: {insight.name}
        {'='*60}
        Description:
        {insight.description}
        Requires Research:
        {insight.research}

        Content:
        {insight.content if insight.content else '[Not yet written]'}"""

    return formatted_str

def tavily_search(query):
    """ Search the web using the Tavily API.

    Args:
        query (str): The search query to execute

    Returns:
        dict: Tavily search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available"""
    tavily_client = TavilyClient(api_key="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")
    return tavily_client.search(query,
                         max_results=5,
                         include_raw_content=True)


async def tavily_search_async(search_queries, tavily_topic, tavily_days):
    """
    Performs concurrent web searches using the Tavily API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process
        tavily_topic (str): Type of search to perform ('news' or 'general')
        tavily_days (int): Number of days to look back for news articles (only used when tavily_topic='news')

    Returns:
        List[dict]: List of search results from Tavily API, one per query

    Note:
        For news searches, each result will include articles from the last `tavily_days` days.
        For general searches, the time range is unrestricted.
    """
    tavily_async_client = AsyncTavilyClient(api_key="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")
    search_tasks = []
    for query in search_queries:
        if tavily_topic == "news":
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=5,
                    include_raw_content=True,
                    topic="news",
                    days=tavily_days
                )
            )
        else:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=5,
                    include_raw_content=True,
                    topic="general"
                )
            )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)

    return search_docs

async def generate_report_plan(state: ReportState):
    """ Generate the report plan """
    # Inputs
    number_of_queries = state["number_of_queries"]
    tavily_topic = state["tavily_topic"]
    tavily_days = state.get("tavily_days", None)
    insight_type = state["insight_type"]
    agreement = state["agreement"]

    # Extract the given insight type
    extraction_instruction_query = insight_extraction_instructions.format(insight_type = insight_type, agreement = agreement)
    structured_extraction_llm = llm.with_structured_output(Extractions)
    extractions = structured_extraction_llm.invoke(extraction_instruction_query)

    # Generate web search queries
    structured_query_llm = llm.with_structured_output(Queries)
    system_instructions_query = report_planner_query_writer_instructions.format(insight_type = insight_type, insight_extraction = extractions, number_of_queries=number_of_queries)
    queries = structured_query_llm.invoke([SystemMessage(content=system_instructions_query)]+[HumanMessage(content="Generate search queries that will help with planning the sections of the report.")])
    query_list = [query.search_query for query in queries.queries]
    search_docs = await tavily_search_async(query_list, tavily_topic, tavily_days)
    source_str = deduplicate_and_format_sources(search_docs, max_tokens_per_source=1000, include_raw_content=False)

    # Generate the report structure
    system_instructions_sections = report_planner_instructions.format(insight_type = insight_type, insight_extraction = extractions, context=source_str)
    report_structured_llm = llm.with_structured_output(Insights)
    report_sections = report_structured_llm.invoke([SystemMessage(content=system_instructions_sections)]+[HumanMessage(content="Generate the sections of the report. Your response must include a 'insights' field containing a list of sections. Each section must have: type, explanation, extraction, document_lookup, deviation and insights_generated fields.")])

    # Update the state with insight outline
    return {"insights": report_sections.insights}

# Prompt to extract the clauses/ obligation from the report
insight_extraction_instructions = """You are an expert legal researcher. You goal is, given a legal document,
extract all the {insight_type} from it.

While extracting the {insight_type} make sure:
1. Extract the exact words from the legal document, without changing them in any way.
2. Only present the extracted {insight_type} without any additional information or conversation.

The agreement to extact {insight_type} is:

{agreement}

"""
class InsightState(TypedDict):
    tavily_topic: Literal["general", "news"]
    tavily_days: Optional[int]
    number_of_queries: int
    insight: Insight
    extractions: Extractions
    agreement: str
    search_queries: list[SearchQuery] # List of search queries
    source_str: str # String of formatted source content from web search
    report_insights_from_research: str # String of any completed sections from research to write final sections
    completed_insights: Annotated[list[Insight], add]

class InsightAgreement(BaseModel):
    agreement: str
    insight_type: str

def get_extractions(state: InsightState):
    """ Generate the report plan """
    # Inputs
    number_of_queries = state["number_of_queries"]
    insight = state["insight"]
    agreement = state["agreement"]

    # Extract the given insight type
    extraction_instruction_query = insight_extraction_instructions.format(insight_type = insight.insight_type, agreement = agreement)
    structured_extraction_llm = llm.with_structured_output(Extractions)
    extractions = structured_extraction_llm.invoke(extraction_instruction_query)

    return {"extractions" : extractions}

# Query writer instructions
query_writer_instructions="""Your goal is to generate targeted web search queries that will gather comprehensive information for deriving insights for the given {insight_type}.

{insight_type}:
{extraction}

When generating {number_of_queries} search queries, ensure they:
1. Cover different aspects of the topic (e.g., core features, real-world applications, technical architecture)
2. Include specific technical terms related to the topic
3. Target recent information by including year markers where relevant (e.g., "2024")
4. Look for comparisons or differentiators from similar technologies/approaches
5. Search for both official documentation and practical implementation examples

Your queries should be:
- Specific enough to avoid generic results
- Technical enough to capture detailed implementation information
- Diverse enough to cover all aspects of the section plan
- Focused on authoritative sources (documentation, technical blogs, academic papers)"""

# Section writer instructions
insight_writer_instructions = """You are an expert laywer, drawing your insights for one {insight_type}.

You need to generate insights for:
{extraction}

Guidelines for writing:

1. Technical Accuracy:
- Include specific version numbers
- Reference concrete metrics/benchmarks
- Cite official documentation
- Use technical terminology precisely

2. Length and Style:
- Strict 150-200 word limit
- No marketing language
- Technical focus
- Write in simple, clear language
- Start with your most important insight in **bold**
- Use short paragraphs (2-3 sentences max)

3. Structure:
- Use ## for section title (Markdown format)
- Only use ONE structural element IF it helps clarify your point:
  * Either a focused table comparing 2-3 key items (using Markdown table syntax)
  * Or a short list (3-5 items) using proper Markdown list syntax:
    - Use `*` or `-` for unordered lists
    - Use `1.` for ordered lists
    - Ensure proper indentation and spacing
- End with ### Sources that references the below source material formatted as:
  * List each source with title, date, and URL
  * Format: `- Title : URL`

3. Writing Approach:
- Include at least one specific example or case study
- Use concrete details over general statements
- Make every word count
- No preamble prior to creating the section content
- Focus on your single most important point

4. Use this source material to help write the section:
{context}

5. Quality Checks:
- Exactly 150-200 words (excluding title and sources)
- Careful use of only ONE structural element (table or list) and only if it helps clarify your point
- One specific example / case study
- Starts with bold insight
- No preamble prior to creating the section content
- Sources cited at end"""

def generate_queries(state: InsightState):
    """ Generate search queries for a insight section """

    # Get state
    insight = state["insight"]
    number_of_queries = state["number_of_queries"]

    # Generate queries
    structured_llm = llm.with_structured_output(Queries)

    # Format system instructions
    system_instructions = query_writer_instructions.format(insight_type=insight.insight_type, extraction = insight.extraction, number_of_queries=number_of_queries)

    # Generate queries
    queries = structured_llm.invoke([SystemMessage(content=system_instructions)]+[HumanMessage(content="Generate search queries on the provided topic.")])

    return {"search_queries": queries.queries}

async def search_web(state: InsightState):
    """ Search the web for each query, then return a list of raw sources and a formatted string of sources."""

    # Get state
    search_queries = state["search_queries"]

    # Get configuration
    tavily_topic = state["tavily_topic"]
    tavily_days = state["tavily_days"]

    # Web search
    query_list = [query.search_query for query in search_queries]
    search_docs = await tavily_search_async(query_list, tavily_topic, tavily_days)

    # Deduplicate and format sources
    source_str = deduplicate_and_format_sources(search_docs, max_tokens_per_source=5000, include_raw_content=True)

    return {"source_str": source_str}

def generate_insight(state: InsightState):
    """ Write a insights for a given clause or obligation"""

    # Get state
    insight = state["insight"]
    source_str = state["source_str"]


    # Format system instructions
    system_instructions = insight_writer_instructions.format(insight_type=insight.insight_type, extraction = insight.extraction, context=source_str)

    # Generate section
    section_content = llm.invoke([SystemMessage(content=system_instructions)]+[HumanMessage(content="Generate your insights based on the provided sources.")])

    # Write content to the section object
    insight_generated = section_content.content

    # Write the updated section to completed sections
    insight.insight_generated = insight_generated
    return {"completed_insights": [insight]}


@contract_router.post("/langgraph/insights")
async def langgraph_contract_agent(insight_input: InsightAgreement):
    report_state = {
        "number_of_queries" : 2,
        "tavily_topic" : "general",
        "tavily_days" : None,
        "insight_type" : insight_input.insight_type,
        "agreement" : insight_input.agreement
    }
    report_plan = await generate_report_plan(report_state)

    # Add nodes and edges
    insight_builder = StateGraph(InsightState)
    insight_builder.add_node("generate_queries", generate_queries)
    insight_builder.add_node("search_web", search_web)
    insight_builder.add_node("generate_insight", generate_insight)

    insight_builder.add_edge(START, "generate_queries")
    insight_builder.add_edge("generate_queries", "search_web")
    insight_builder.add_edge("search_web", "generate_insight")
    insight_builder.add_edge("generate_insight", END)
    memory = MemorySaver()

    insight_builder_graph = insight_builder.compile(checkpointer=memory)

    final_state = None
    for index in range(len(report_plan["insights"])):
        insight = report_plan["insights"][index]
        report_state["insight"] = insight
        print("Initial State is: ", report_state)
        final_state = await insight_builder_graph.ainvoke(report_state,
                                                  config={"configurable": {"thread_id": "1"}})

    return final_state["completed_insights"]






