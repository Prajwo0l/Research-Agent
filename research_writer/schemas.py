from __future__ import annotations
import operator
from typing import Annotated, Any ,Dict,List,Optional
from pydantic import BaseModel,Field
from typing_extensions import TypedDict


#############Input#############################
class SourceItem(BaseModel):
    title : str
    url : str
    snippet : Optional[str]=None
    domain: str="unknown"
    query_used : str =""

class WriterInput(BaseModel):
    topic : str
    scope :Optional[str] = None
    focus : Optional[str]=None
    sources: List[SourceItem]=Field (default_factory=list)
    literature_summary :Optional[str]=None
    knowledge_map : Optional[str]=None
    annotated_bib:Optional[str]=None
    pre_fetched_content : Dict[str,str]=Field(default_factory=dict)
    max_debate_rounds : int=Field(default=3,ge=1,le=6)
    max_clusters: int=Field(default=8,ge=2,le=15)
    target_section_words: int=Field(default=600,ge=200,le=2000)

######################################Stage 1 -Url content Fetching
class FetchedSource(BaseModel):
    title : str
    url :str
    domain : str
    query_used :str
    content:str
    fetch_status:str
    word_count:int=0

class SourceFetchWorkerState(TypedDict):
    source :SourceItem
    pre_fetched_content:Dict[str,str]
    fetched_sources:Annotated[List[FetchedSource],operator.add]



####################Stage 2 Themantic Clustering

class Cluster(BaseModel):
    cluster_id :str
    theme:str
    section_title:str
    rationale :str
    source_urls:List[str] =Field(default_factory=list)
    writing_goal :str

class ClusterPlan(BaseModel):
    document_title :str
    document_abstract:str
    clusters:List[Cluster]

###############stage 3 Miro fish debate Loop

class DebateTurn(BaseModel):
    role : str
    content :str
    round: int

class DebateResult(BaseModel):
    cluster_id :int
    cluster_theme:str
    section_title:str
    debate_turns:List[DebateTurn]
    final_section:str
    word_count:int
    rounds_taken:int


class DebateWorkerState(TypedDict):
    cluster : Cluster
    fetched_sources :List[FetchedSource]
    topic : str
    literature_context :str
    max_rounds :int
    target_words :int
    debate_results: Annotated[List[DebateResult], operator.add]


###### Stage 4 - Output

class WriterOutput(BaseModel):
    topic :str
    document_title :str
    abstract :str
    sections :List[DebateResult]
    full_document :str
    output_path :str
    total_words:int
    total_rounds:int
    sources_used :int

class WriterState(TypedDict):
    writer_input: WriterInput
    thread_id: str
    # stage 1
    fetched_sources :Annotated[List[FetchedSource],operator.add]
    #stage 2
    cluster_plan :ClusterPlan
    #stage 3
    debate_results : Annotated[List[DebateResult],operator.add]
    #stage 4
    writer_output: WriterOutput
    revision_cycle: int

