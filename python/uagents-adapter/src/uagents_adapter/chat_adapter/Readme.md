# Chat Adapter

A universal plugin that connects any AI agent framework (LlamaIndex, LangChain, LangGraph, CrewAI, or custom) with the uAgents ecosystem, enabling seamless communication within the Fetch.ai agent network.

## 🌟 Features

- **Framework Agnostic**: Works with LlamaIndex, LangChain, LangGraph, CrewAI, or any custom agent implementation
- **Simple Integration**: Just write a function - that's it!
- **Session Management**: Built-in support for session and user context tracking
- **Agentverse Compatible**: Optional integration with Fetch.ai's Agentverse platform
- **Flexible Configuration**: Use with or without authentication tokens and context management
- **Async-First**: Built on modern async/await patterns for optimal performance

## 📦 Installation

```bash
pip install uagents-adapter
```

Install your preferred framework(s):
```bash
# For LlamaIndex
pip install llama-index

# For LangChain
pip install langchain

# For LangGraph
pip install langgraph

# For CrewAI
pip install crewai
```

## 🚀 Quick Start

The plugin works with **any** framework. Here's a minimal example:

```python
from dataclasses import dataclass  # Import dataclass
from uagents import Agent
from uagents_adapter.chat_adapter import ChatAdapter
from my_workflow import my_workflow  # <-- Import your custom workflow here

@dataclass
class AdapterContext:   
    """A simple data container for session context."""
    session_id: str
    user_id: str

async def My_Agent(query: str, session_id: str, user_id: str) -> str:
            
    # Create the context object, like in your snippet
    ctx = AdapterContext(session_id=session_id, user_id=user_id)
    response = await my_workflow(query, tx=ctx)
    return str(response)

plugin = ChatAdapter(My_Agent)

agent = Agent(
    name="Chat_Agent",
    seed="your_secure_seed_phrase_production",
    port=8000,
    mailbox=True
)

agent.include(plugin.protocol, publish_manifest=True)

print(f"Agent address: {agent.address}")

plugin.run(agent)
```

## 🎯 Framework Examples

### LlamaIndex Example

```python
from uagents import Agent
from uagents_adapter.chat_adapter import ChatAdapter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# Initialize your LlamaIndex agent
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(llm=OpenAI(model="gpt-4"))

async def llamaindex_handler(query: str, session_id: str, user_id: str) -> str:
    response = query_engine.query(query)
    return str(response)

plugin = ChatAdapter(
    llamaindex_handler,
    description="LlamaIndex RAG agent"
)

agent = Agent(name="llamaindex_agent", seed="seed_phrase", port=8000, mailbox=True)
agent.include(plugin.protocol, publish_manifest=True)
plugin.run(agent)
```

### LangChain Example

```python
from uagents import Agent
from uagents_adapter.chat_adapter import ChatAdapter
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize your LangChain agent
llm = ChatOpenAI(model="gpt-4")
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

async def langchain_handler(query: str, session_id: str, user_id: str) -> str:
    response = conversation.invoke(input=query)
    return response

plugin = ChatAdapter(
    langchain_handler,
    description="LangChain conversational agent"
)

agent = Agent(name="langchain_agent", seed="seed_phrase", port=8000, mailbox=True)
agent.include(plugin.protocol, publish_manifest=True)
plugin.run(agent)
```

### CrewAI Example

```python
from uagents import Agent
from uagents_adapter.chat_adapter import ChatAdapter
from crewai import Agent as CrewAgent, Task, Crew

# Initialize your CrewAI setup
researcher = CrewAgent(
    role='Researcher',
    goal='Research and provide accurate information',
    backstory='Expert researcher',
    verbose=True
)

async def crewai_handler(query: str, session_id: str, user_id: str) -> str:
    task = Task(description=query, agent=researcher)
    crew = Crew(agents=[researcher], tasks=[task])
    result = crew.kickoff()
    return str(result)

plugin = ChatAdapter(
    crewai_handler,
    description="CrewAI research agent"
)

agent = Agent(name="crewai_agent", seed="seed_phrase", port=8000, mailbox=True)
agent.include(plugin.protocol, publish_manifest=True)
plugin.run(agent)
```

### Custom Agent Example

```python
from dataclasses import dataclass  # Import dataclass
from uagents import Agent
from uagents_adapter.chat_adapter import ChatAdapter
from my_workflow import my_workflow  # <-- Import your custom workflow here

@dataclass
class AdapterContext:   
    """A simple data container for session context."""
    session_id: str
    user_id: str

async def Chat_agent_function(query: str, session_id: str, user_id: str) -> str:
            
    # Create the context object, like in your snippet
    ctx = AdapterContext(session_id=session_id, user_id=user_id)
    response = await my_workflow(query, tx=ctx)
    return str(response)

plugin = ChatAdapter(Chat_agent_function)

agent = Agent(
    name="Chat_Agent",
    seed="your_secure_seed_phrase_production",
    port=8000,
    mailbox=True
)

agent.include(plugin.protocol, publish_manifest=True)

print(f"Agent address: {agent.address}")

plugin.run(agent)
```

## 📖 Usage Scenarios

The plugin supports four main configuration patterns:

### 1. With Agentverse Token + Session Context

**Use Case**: Production deployment with full tracking and Agentverse integration

```python
from dataclasses import dataclass
from uagents import Agent
from uagents_adapter.chat_adapter import ChatAdapter
from my_workflow import my_workflow  # <-- Import your custom workflow here
AGENTVERSE_API_TOKEN = "av_xxxxxxxxxxxxxxxxxxxxx"

@dataclass
class SessionContext:
    session_id: str
    user_id: str

async def handle_query(query: str, session_id: str, user_id: str) -> str:

     # Create the context object, like in your snippet
    ctx = AdapterContext(session_id=session_id, user_id=user_id)
    response = await my_workflow(query, tx=ctx)
    return str(response)

plugin = ChatAdapter(
    handle_query,
    agentverse_api_token=AGENTVERSE_API_TOKEN,
    description="Production agent with session tracking"
)

agent = Agent(
    name="production_agent",
    seed="your_secure_seed_phrase_production",
    port=8000,
    mailbox=True
)

agent.include(plugin.protocol, publish_manifest=True)
plugin.run(agent)
```

### 2. Session Context Only (No Agentverse)

**Use Case**: Local development or private deployments with session tracking

```python
from dataclasses import dataclass
from uagents import Agent
from uagents_adapter.chat_adapter import ChatAdapter
from my_workflow import my_workflow  # <-- Import your custom workflow here

@dataclass
class SessionContext:
    session_id: str
    user_id: str

async def handle_query(query: str, session_id: str, user_id: str) -> str:
      # Create the context object, like in your snippet
    ctx = AdapterContext(session_id=session_id, user_id=user_id)
    response = await my_workflow(query, tx=ctx)
    return str(response)

plugin = ChatAdapter(handle_query)

agent = Agent(
    name="dev_agent",
    seed="your_secure_seed_phrase_dev",
    port=8000,
    mailbox=True
)

agent.include(plugin.protocol, publish_manifest=True)
plugin.run(agent)
```

### 3. With Agentverse Token (Stateless)

**Use Case**: Simple production deployment without session management

```python
from uagents import Agent
from uagents_adapter.chat_adapter import ChatAdapter
from my_workflow import my_workflow  # <-- Import your custom workflow here

AGENTVERSE_API_TOKEN = "av_xxxxxxxxxxxxxxxxxxxxx"

async def handle_query(query: str, session_id: str, user_id: str) -> str:
     # Create the context object, like in your snippet
    ctx = AdapterContext(session_id=session_id, user_id=user_id)
    response = await my_workflow(query, tx=ctx)
    return str(response)

plugin = ChatAdapter(
    handle_query,
    agentverse_api_token=AGENTVERSE_API_TOKEN,
    description="Stateless agent for simple queries"
)

agent = Agent(
    name="stateless_agent",
    seed="your_secure_seed_phrase_stateless",
    port=8000,
    mailbox=True
)

agent.include(plugin.protocol, publish_manifest=True)
plugin.run(agent)
```

### 4. Minimal Configuration

**Use Case**: Quick prototyping and testing

```python
from uagents import Agent
from uagents_adapter.chat_adapter import ChatAdapter
from my_workflow import my_workflow  # <-- Import your custom workflow here

async def handle_query(query: str, session_id: str, user_id: str) -> str:
     # Create the context object, like in your snippet
    ctx = AdapterContext(session_id=session_id, user_id=user_id)
    response = await my_workflow(query, tx=ctx)
    return str(response)

plugin = ChatAdapter(handle_query)

agent = Agent(
    name="test_agent",
    seed="test_seed_phrase",
    port=8000,
    mailbox=True
)

agent.include(plugin.protocol, publish_manifest=True)
plugin.run(agent)
```

## 🔧 Configuration Options

### ChatAdapter Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query_handler` | `Callable` | Yes | Async function that processes queries |
| `agentverse_api_token` | `str` | No | Token for Agentverse platform integration |
| `description` | `str` | No | Human-readable description of your agent |

### Query Handler Signature

Your query handler must match this signature:

```python
async def query_handler(
    query: str,        # The user's query
    session_id: str,   # Unique session identifier
    user_id: str       # Unique user identifier
) -> str:              # Return response as string
    pass
```

## 💡 Key Concept

The beauty of `ChatAdapter` is its simplicity: **you just write a function**. 

The function signature is always the same regardless of which framework you use:
```python
async def my_function(query: str, session_id: str, user_id: str) -> str:
    # Your logic with any framework
    return response
```

Inside this function, you can use:
- ✅ LlamaIndex query engines
- ✅ LangChain chains
- ✅ LangGraph workflows
- ✅ CrewAI crews
- ✅ Custom implementations
- ✅ API calls
- ✅ Database queries
- ✅ Anything else!

## 🔐 Security Best Practices

- **Never commit tokens**: Store `AGENTVERSE_API_TOKEN` in environment variables
- **Use strong seeds**: Generate secure seed phrases for production agents
- **Validate inputs**: Always sanitize user queries before processing
- **Rate limiting**: Implement rate limiting for production deployments

```python
import os

AGENTVERSE_API_TOKEN = os.getenv("AGENTVERSE_API_TOKEN")
AGENT_SEED = os.getenv("AGENT_SEED")
```

## 🏗️ Advanced Example with Context Management

Here's a complete example showing context-aware agent implementation:

```python
from dataclasses import dataclass
from typing import Optional
from uagents import Agent
from uagents_adapter.chat_adapter import ChatAdapter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

@dataclass
class SessionContext:
    session_id: str
    user_id: str

class ContextAwareAgent:
    def __init__(self, context: Optional[SessionContext] = None):
        self.context = context
        self.llm = OpenAI(model="gpt-4")
        
        # Load your data
        documents = SimpleDirectoryReader("data").load_data()
        self.index = VectorStoreIndex.from_documents(documents)
        self.query_engine = self.index.as_query_engine(llm=self.llm)
    
    async def query(self, question: str) -> str:
        """Process a query with optional context"""
        if self.context:
            # Personalized query with user context
            enhanced_query = f"[User: {self.context.user_id}] {question}"
            response = self.query_engine.query(enhanced_query)
        else:
            response = self.query_engine.query(question)
        
        return str(response)

# Handler function
async def handle_query(query: str, session_id: str, user_id: str) -> str:
    context = SessionContext(session_id=session_id, user_id=user_id)
    agent = ContextAwareAgent(context=context)
    response = await agent.query(query)
    return response

# Setup plugin
plugin = ChatAdapter(
    handle_query,
    agentverse_api_token=os.getenv("AGENTVERSE_API_TOKEN"),
    description="Context-aware LlamaIndex agent"
)

agent = Agent(
    name="context_aware_agent",
    seed=os.getenv("AGENT_SEED"),
    port=8000,
    mailbox=True
)

agent.include(plugin.protocol, publish_manifest=True)
plugin.run(agent)
```



## 💡 Support

For questions and support:
- Open an issue on GitHub
- Join the Fetch.ai Discord community
- Check the documentation

---

**Made with ❤️ for the Fetch.ai agent ecosystem**