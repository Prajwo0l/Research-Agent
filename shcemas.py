from __future__ import annotations
import operator
from typing import Annotated,List,Literal,Optional 
from pydantic import BaseModel,Field
from typing_extensions import TypedDict

# Phase 1 - Topic Decomposition

class SearchQuery(BaseModel):
    '''A single targeted search query with domain and rationale'''
    query:str=Field(
        ...,
        description = 'A precise search query string(5-12 words)'
    )
    domain : Literal[
        'academic_papers',
        'preprints',
        'patents_applied',
        'grey_litreature',
        'news_commentary',
        'conference_proceedings',
    ]= Field(...,description = 'Target source domain for this query.')
    rationale:str=Field(
        ...,
        description ='One sentence explaining why this query is important for the topic.', 
    )
class TopicDecomposition(BaseModel):
    '''Structured output opf Phase 1 topic analysis'''
    topic :str 
    scope_statement:str = Field(
        ...,
        description = "1-2 sentence defintion of the research scope",

    )
    core_subtopics  :List[str]=Field(
        ...,
        min_length=4,
        max_length=10,
        description='Primary sub-fields and adjacent research areas.'
    )
    disciplines:List[str]=Field(
        ...,
        description='Academic disciplines involved in this topic.',
    )
    search_queries:List[SearchQuery]=Field(
        ...,
        min_length=10,
        max_length=20,
        description="10-20 precise queries covering all source domains.",
    )

    