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


# Phase 2 -Evidence Collection 

class EvidenceItem(BaseModel):
    '''A single retrieved evidence item from a search result'''
    title:str
    url:str
    snippet:Optional[str]=None
    source:Optional[str]=None
    domain:str='unknown'
    query_used:str=""

# Phase 3 -Critical Evaluation
class EvidenceEvaluation(BaseModel):
    '''Critical evaluation of the collected evidence corpus.'''
    consensus_areas:List[str]=Field(
        ...,
        description = 'Topics where strong empirical consensus ecists across sources.',   
    )
    contested_claims:List[str]=Field(
        ...,
        description='findings that are actively debated or challenged in the litreature.'
    )
    key_debates:List[str]=Field(
        ...,
        description='Fundamental unresolved disagreements between researcher or schools of thought.',
    )
    replication_concerns:List[str]=Field(
        ...,
        description='Results with known replication issues or contradicting null findings.'
    )
    methodological_landscapes:str=Field(
        ...,
        description='Summary of dominant methods,their strengths/limitations,and emerging approaches',
    )
    quality_notes:str=Field(
        ...,
        description='Overall quality assessment of the evidence corpus and any notable biases.'
    )

# Phase 4 -Synthesis Worker State(internal)
class SynthesisWorkerState(TypedDict):
    '''State passed to each parallel synthesis worker.'''
    output_type:str
    topic:str
    decomposition_json:str
    evidence_json:str
    evaluation_json:str
    synthesis_outputs:Annotated[List[str],operator.add]

class SearchWorkerState(TypedDict):
    '''State passed to each parallel search worker'''
    topic:str
    query:str
    domain:str
    rationale:str
    evidence_items:Annotated[List[EvidenceItem],operator.add]


# Main Langgraph State
class ResearchState(TypedDict):
    '''Top-level state flowing through the entire research graph.'''
    # input
    topic:str
    scope:Optional[str]
    focus:Optional[str]

    # phase 1
    decomposition:TopicDecomposition
    
    # phase 2
    evidence_items:Annotated[List[EvidenceItem],operator.add]

    # phase 3
    evaluation:EvidenceEvaluation

    # phase 4
    synthesis_output:Annotated[List[str],operator.add]

    # Final
    final_report:str
    output_path:str