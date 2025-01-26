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