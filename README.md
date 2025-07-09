# Product Search Bot

A conversational bot that integrates LangChain, Pinecone, and OpenAI models to provide a seamless product search experience. The bot is exposed via a FastAPI endpoint.

## ✨ Features

-   **Conversational Interface**: Interact with the bot through a simple chat interface.
-   **Semantic Product Search**: Find products using natural language queries.
-   **Advanced Filtering**: Search for products based on metadata such as brand, price, category, and ingredients.
-   **Detailed Information**: Retrieve specific product details and reviews using product IDs.
-   **Context-Aware Conversations**: The bot maintains a memory of the conversation to provide contextual answers.
-   **Asynchronous API**: Built with FastAPI for a high-performance, non-blocking API.

## 🏗️ Architecture

The project is structured into two main agents:

1.  **`ConversationalAgent`**: This is the high-level agent that manages the overall flow of the conversation. It interprets the user's intent and decides whether to respond directly or delegate the task to a specialized tool.
2.  **`SearchAgent`**: This is the core worker agent, equipped with a suite of tools to interact with the product database (Pinecone). Its tools include:
    -   `product_search_tool`: Performs general semantic searches.
    -   `metadata_filter_tool`: Filters products based on specific attributes.
    -   `product_info_tool`: Fetches detailed information for a given product ID.
    -   `review_search_tool`: Retrieves reviews for a specific product.

The application uses **Poetry** for dependency management and loads configuration from a `.env` file.

## 🚀 Getting Started

### Prerequisites

-   Python 3.9+ and Pip
-   [Poetry](https://python-poetry.org/docs/#installation) for dependency management.
-   Access keys for OpenAI and Pinecone.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd product-search-bot
    ```

2.  **Set up environment variables:**
    Copy the example environment file and fill in your API keys.
    ```sh
    cp .env.example .env
    ```
    Then, edit `.env` with your credentials:
    ```
    OPENAI_API_KEY="your_openai_api_key"
    PINECONE_API_KEY="your_pinecone_api_key"
    PINECONE_ENVIROMENT="your_pinecone_environment"
    PINECONE_INDEX_NAME="your_pinecone_index_name"
    ```

3.  **Install dependencies:**
    Use Poetry to install the required packages.
    ```sh
    poetry install
    ```

### Running the Application

To start the FastAPI server, run the following command from the root directory:

```sh
poetry run uvicorn app:app --reload
```

The server will be available at `http://127.0.0.1:8000`.

## ⚙️ API Usage

The main endpoint for interacting with the bot is `/chat`.

-   **Endpoint**: `POST /chat`
-   **Request Body**: A JSON object with a `message` key.
    ```json
    {
      "message": "your_question_here"
    }
    ```

**Example using cURL:**

```sh
cURL -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Do you have any face creams under $50?"}'
```
