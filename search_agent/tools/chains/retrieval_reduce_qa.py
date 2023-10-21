# Set Reduce chain
from typing import Any, List
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chains import LLMChain, ReduceDocumentsChain, StuffDocumentsChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.schema.document import Document
from langchain.schema import BasePromptTemplate
from langchain.chat_models import ChatOpenAI


class CustomStuffDocumentsChain(StuffDocumentsChain):
    def _get_inputs(self, docs: List[Document], **kwargs: Any) -> dict:
        # Format each document according to the prompt
        doc_strings = [format_document(
            doc, self.document_prompt) for doc in docs]
        # Join the documents together to put them in the prompt.
        inputs = {
            k: v
            for k, v in kwargs.items()
            if k in self.llm_chain.prompt.input_variables
        }
        inputs[self.document_variable_name] = self.document_separator.join(
            doc_strings)
        return inputs


def create_retrieval_reduce_qa_tool(llm: ChatOpenAI, retriever: VectorStoreRetriever, reduce_template: str, document_template: str, input_variables):
    """Create a retrieval qa tool with a reduce chain."""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    document_prompt = PromptTemplate(
        input_variables=input_variables,
        template=document_template
    )

    combine_documents_chain = CustomStuffDocumentsChain(
        llm_chain=reduce_chain,
        document_prompt=document_prompt,
        document_variable_name="content",
        verbose=True
    )

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000,
        verbose=True
    )

    return RetrievalQA(
        retriever=retriever,
        combine_documents_chain=reduce_documents_chain,
        verbose=True
    )


def format_document(doc: Document, prompt: BasePromptTemplate) -> str:
    """Format a document into a string based on a prompt template."""
    base_info = {"page_content": doc.page_content, **doc.metadata}
    missing_metadata = set(prompt.input_variables).difference(base_info)

    document_info = {
        k: base_info[k] if k not in missing_metadata else None for k in prompt.input_variables}
    prompt_format = prompt.format(**document_info)
    return prompt_format
