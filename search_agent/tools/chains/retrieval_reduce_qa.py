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
    """
    A custom document stuffing chain that formats each document individually.

    This class overrides the default behavior to ensure each document is formatted
    using the provided `document_prompt` before being joined together. This is
    useful for when documents have metadata that needs to be included in the final
    context string.
    """
    def _get_inputs(self, docs: List[Document], **kwargs: Any) -> dict:
        """
        Formats and combines documents into a single input for the LLM chain.

        Args:
            docs (List[Document]): The list of documents to be processed.
            **kwargs: Additional keyword arguments to be passed to the LLM chain.

        Returns:
            dict: A dictionary of inputs for the LLM chain, with the formatted
                  documents under the `document_variable_name` key.
        """
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
    """
    Creates a RetrievalQA tool that uses a map-reduce style approach.

    This function constructs a complex chain that first retrieves relevant documents,
    then combines them (stuffing) into a single context, and finally "reduces"
    them by passing them to an LLM to generate a final answer based on a template.

    Args:
        llm (ChatOpenAI): The language model to use for the final reduction step.
        retriever (VectorStoreRetriever): The retriever for fetching relevant documents.
        reduce_template (str): The prompt template for the final summarization/reduction.
        document_template (str): The prompt template for formatting each individual document.
        input_variables (list[str]): The list of variable names expected by the document_template.

    Returns:
        RetrievalQA: A configured RetrievalQA chain instance.
    """
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
    """
    Safely formats a document into a string using a prompt template.

    This function combines the document's page_content and metadata and uses them
    to format the prompt. It handles cases where metadata keys required by the
    prompt are missing from the document by substituting `None`.

    Args:
        doc (Document): The document to format.
        prompt (BasePromptTemplate): The template to use for formatting.

    Returns:
        str: The formatted document as a string.
    """
    base_info = {"page_content": doc.page_content, **doc.metadata}
    missing_metadata = set(prompt.input_variables).difference(base_info)

    document_info = {
        k: base_info[k] if k not in missing_metadata else None for k in prompt.input_variables}
    prompt_format = prompt.format(**document_info)
    return prompt_format
