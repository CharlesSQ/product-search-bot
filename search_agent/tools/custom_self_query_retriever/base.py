from typing import Any, Dict, List, cast

from langchain.prompts import ChatPromptTemplate,  HumanMessagePromptTemplate
from langchain.retrievers.self_query.base import _get_builtin_translator
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.ir import StructuredQuery
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema import BasePromptTemplate
from langchain.vectorstores import VectorStore
from langchain.schema import SystemMessage
from langchain.chains.llm import LLMChain
from langchain.schema import Document

from .structure_output_parser import StructuredQueryOutputParser
from .prompt import SCHEMA_INSTRUCTIONS, SUFFIX


def load_query_constructor_chain(
    llm: BaseLanguageModel,
    document_contents: str,
    attribute_info: Dict[str, Any],
) -> LLMChain:

    prompt = get_prompt(
        document_contents,
        attribute_info
    )
    return LLMChain(llm=llm, prompt=prompt)


def get_prompt(
    document_contents: str,
    attribute_info: Dict[str, Any],
) -> BasePromptTemplate:

    template = SCHEMA_INSTRUCTIONS.format(
        content=document_contents, attributes=attribute_info)

    prompt_messages = [
        SystemMessage(content=template),
        HumanMessagePromptTemplate.from_template(SUFFIX)

    ]

    return ChatPromptTemplate(messages=prompt_messages)


class CustomSelfQueryRetriever(SelfQueryRetriever):
    output_parser: StructuredQueryOutputParser
    max_retry: int
    retry_message: str

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        retry_count = 0
        while retry_count < self.max_retry:
            try:
                inputs = self.llm_chain.prep_inputs({"query": query})
                output = self.llm_chain(inputs)

                parsed = self.output_parser.parse(output)

                structured_query = cast(
                    StructuredQuery,
                    parsed
                )

                # if self.verbose:
                #     print(structured_query)

                new_query, new_kwargs = self.structured_query_translator.visit_structured_query(
                    structured_query
                )

                # if structured_query.limit is not None:
                #     new_kwargs["k"] = structured_query.limit

                # if self.use_original_query:
                #     new_query = query

                search_kwargs = {**self.search_kwargs, **new_kwargs}
                print('\n\nsearch_kwargs:', search_kwargs)
                docs = self.vectorstore.search(
                    new_query, self.search_type, **search_kwargs)
                return docs

            except Exception as e:
                retry_count += 1
                print('\nRetry count:', retry_count)

        if retry_count == self.max_retry:
            return self.retry_message

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        vectorstore: VectorStore,
        document_contents: str,
        metadata_field_info: Dict[str, Any],
        max_retry: int = 1,
        retry_message: str = "",
    ) -> SelfQueryRetriever:
        print('\n\ncustom structured_query_translator:')
        structured_query_translator = _get_builtin_translator(vectorstore)

        output_parser = StructuredQueryOutputParser.from_components(
            allowed_comparators=structured_query_translator.allowed_comparators, allowed_operators=structured_query_translator.allowed_operators
        )

        llm_chain = load_query_constructor_chain(
            llm,
            document_contents,
            attribute_info=metadata_field_info,
        )
        return cls(
            llm_chain=llm_chain,
            vectorstore=vectorstore,
            output_parser=output_parser,
            structured_query_translator=structured_query_translator,
            max_retry=max_retry,
            retry_message=retry_message
        )
