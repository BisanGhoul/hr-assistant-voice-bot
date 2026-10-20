"""Tools Shahd can call during a conversation.

The LLM reads each tool's docstring to decide when to call it,
so the docstring must say clearly what the tool covers.
"""

import asyncio

from pipecat.frames.frames import EndWorkerFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallParams

from retrieval import search


async def search_hr_procedures(params: FunctionCallParams, query: str):
    """Search ASAL's HR and Admin work procedures, such as vacations, sick leave, attendance and check-in, and office keys. Use it for any question about company rules or how to do something at ASAL.

    Args:
        query: The employee's question rewritten as a short, clear search query in English.
    """
    try:
        # search() is slow, blocking work (it embeds the query), so run it in a
        # background thread: otherwise the audio would freeze while it runs.
        results = await asyncio.to_thread(search, query)
        await params.result_callback({
            "results": [
                {
                    "procedure": r["metadata"].get("procedure"),
                    "procedure_number": r["metadata"].get("procedure_number"),
                    "text": r["text"],
                }
                for r in results
            ]
        })
    except Exception as e:
        # Tell the LLM it failed, so Shahd says so instead of guessing
        await params.result_callback({"error": f"Search failed: {e}"})


def warm_up():
    """Load the embedding model and open the index now, so the first question isn't slow."""
    search("warm up")

async def end_conversation(params: FunctionCallParams):
    """End the call. Use it when the employee says goodbye or says they have no more questions."""
    await params.llm.push_frame(TTSSpeakFrame("Goodbye, have a nice day!"))
    # Resolve the call first so the LLM isn't left waiting. None = no extra LLM reply,
    # so Shahd doesn't say goodbye twice.
    await params.result_callback(None)
    # Ends the session gracefully: the goodbye above is spoken before shutdown
    await params.llm.push_frame(EndWorkerFrame(), FrameDirection.DOWNSTREAM)