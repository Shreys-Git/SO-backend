from core.llm.openai import llm
from core.utility.prompts.calender import obligation_extraction_instructions
from schemas.calender import ExtractionState, Obligations


async def extract_obligations(state: ExtractionState):
    # Get the vector search results
    vector_search = state["vector_search"]

    extraction_instruction_query = obligation_extraction_instructions.format(vector_search=vector_search)
    structured_extraction_llm = llm.with_structured_output(Obligations)
    extractions = structured_extraction_llm.invoke(extraction_instruction_query)

    return {"extractions": extractions}