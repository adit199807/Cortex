from langgraph.graph import START, StateGraph,  END
from src.state.rag_state import RAGState
from src.node.nodes import RAGNodes

class GraphBuilder:
    def __init__(self, retriever, llm) -> None:
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None

    def build(self):
        builder  = StateGraph(RAGState)
        builder.add_node('retriever', self.nodes.retrieve_docs)
        builder.add_node('responder', self.nodes.generate_answer)

        # builder.set_entry_point('retriever')
        builder.add_edge(START,'retriever')
        builder.add_edge('retriever', 'responder')
        builder.add_edge('responder', END)


        self.graph = builder.compile()
        return self.graph

    def run(self, question: str)->dict:
        if not self.graph:
            self.build()

        initial_state = RAGState(question = question)
        return self.graph.invoke(initial_state)