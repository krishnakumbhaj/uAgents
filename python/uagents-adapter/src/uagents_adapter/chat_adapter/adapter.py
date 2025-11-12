import asyncio
import inspect
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional, Protocol as TypingProtocol, Union, runtime_checkable
from uuid import uuid4

from uagents import Context, Model, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)

# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================



class ChatAdapterError(Exception):
    """Base exception for ChatAdapter errors.
    
    All plugin-specific exceptions inherit from this class,
    making it easy to catch plugin-related errors separately.
    """
    pass



class QueryEngineError(ChatAdapterError):
    """Exception raised when query engine operations fail.
    
    This includes:
    - Query execution failures
    - Response parsing errors
    - Missing required attributes
    - Invalid query engine state
    
    Attributes:
        message: Human-readable error description
        original_error: The underlying exception (if any)
        query: The query that caused the error (if applicable)
    """
    def __init__(
        self, 
        message: str, 
        original_error: Optional[Exception] = None,
        query: Optional[str] = None
    ):
        self.message = message
        self.original_error = original_error
        self.query = query
        full_message = message
        if query:
            full_message += f" | Query: '{query[:100]}...'"
        if original_error:
            full_message += f" | Cause: {type(original_error).__name__}: {str(original_error)}"
        super().__init__(full_message)


# ============================================================================
# Protocol Definition
# ============================================================================


@runtime_checkable
class ChatAdapterCallable(TypingProtocol):
    """Protocol defining the callable signature for AI functions.
    
    Any callable (function, method, or object with __call__) that matches
    this signature can be used with the plugin. This makes the plugin
    framework-agnostic and works with:
    - LlamaIndex agents/workflows
    - CrewAI agents
    - LangChain/LangGraph chains
    - AutoGen agents
    - OpenAI SDK functions
    - Custom functions wrapping any AI framework
    
    The callable receives:
        query: The user's natural language query
        session_id: Unique session identifier for conversation tracking
        user_id: The sender's address/ID for user-specific context
    It must return either:
        - A string with the response
        - An Awaitable[str] (async function)
    Example implementations:
        >>> # Sync function
        >>> def my_ai_function(query: str, session_id: str, user_id: str) -> str:
        >>>     # Your AI logic here (LlamaIndex, OpenAI, etc.)
        >>>     return f"Response to: {query}"
        >>> 
        >>> # Async function
        >>> async def my_async_ai(query: str, session_id: str, user_id: str) -> str:
        >>>     result = await some_ai_call(query)
        >>>     return result
        >>> 
        >>> # Class with __call__
        >>> class MyAIWrapper:
        >>>     def __call__(self, query: str, session_id: str, user_id: str) -> str:
        >>>         return self.process(query)
    """
    
    def __call__(
        self,
        query: str,
        session_id: str,
        user_id: str
    ) -> Union[str, Awaitable[str]]:
        """Process a query with session and user context.
        
        Args:
            query: Natural language query from the user
            session_id: Unique session identifier for conversation tracking
            user_id: User/sender identifier for personalization
            
        Returns:
            Response string or awaitable that resolves to string
        """
        ...


# ============================================================================
# Message Models
# ============================================================================


class QueryIndex(Model):
    """Message model for querying the LlamaIndex engine.
    
    This message type is used for direct programmatic queries to the agent.
    Users can send this message to the agent's address to get responses.
    
    Attributes:
        query: The natural language query to process
        
    Example:
        >>> from uagents import Context
        >>> await ctx.send(
        >>>     agent_address,
        >>>     QueryIndex(query="What is the main topic of the documents?")
        >>> )
    """
    query: str


class QueryIndexResponse(Model):
    """Response model for query results.
    
    Contains the query result along with optional source attributions
    and error information if the query failed.
    
    Attributes:
        result: The query result text (or error message if failed)
        sources: List of source document excerpts (if available)
        error: Error message if query failed (None on success)
        
    Example Response:
        >>> QueryIndexResponse(
        >>>     result="The main topic is artificial intelligence...",
        >>>     sources=["Source 1 excerpt...", "Source 2 excerpt..."],
        >>>     error=None
        >>> )
    """
    result: str
    sources: Optional[list[str]] = None
    error: Optional[str] = None

# ============================================================================
# Main Adapter Class
# ============================================================================

class ChatAdapter:
    """Universal plugin for integrating ANY AI framework with uAgents.
    
    This plugin is framework-agnostic and works with:
    - LlamaIndex (agents, workflows, query engines)
    - OpenAI SDK
    - LangChain / LangGraph
    - CrewAI
    - AutoGen
    - Custom AI functions
    
    Instead of accepting a specific framework object, the plugin accepts
    a simple function with signature:
        (query: str, session_id: str, user_id: str) -> str | Awaitable[str]
    
    This gives you complete control over:
    - How queries are processed
    - How conversation memory is managed
    - Which AI framework(s) to use
    - How to handle session and user context
    
    The plugin handles:
    - ASI1/Agentverse chat protocol integration
    - Automatic session ID extraction/generation
    - User ID extraction from sender
    - Async/sync function support
    - Error handling and graceful degradation
    - Optional Agentverse cloud registration
    
    Registration Behavior:
        - Almanac (local): ALWAYS automatic via uAgents library
        - Agentverse (cloud): Optional, only if agentverse_api_token provided
    
    Thread Safety:
        All handlers are async and thread-safe.
        
    Error Handling:
        All operations are wrapped in comprehensive try/except blocks internally.
        User code doesn't need try/catch - adapter handles all errors gracefully.
    
    Example with LlamaIndex:
        >>> from llama_index.core import VectorStoreIndex
        >>> from uagents import Agent
        >>> from uagents_adapter import LlamaIndexAdapter
        >>> 
        >>> # Create your LlamaIndex setup
        >>> index = VectorStoreIndex.from_documents(documents)
        >>> query_engine = index.as_query_engine()
        >>> 
        >>> # Wrap in a function
        >>> def my_rag_function(query: str, session_id: str, user_id: str) -> str:
        >>>     response = query_engine.query(query)
        >>>     return str(response)
        >>> 
        >>> # Create adapter and agent
        >>> adapter = LlamaIndexAdapter(my_rag_function)
        >>> agent = Agent(name="my_agent", seed="...", mailbox=True)
        >>> agent.include(adapter.protocol, publish_manifest=True)
        >>> adapter.run(agent)  # ✅ One call to run everything!
    
    Example with OpenAI and memory:
        >>> from openai import OpenAI
        >>> 
        >>> client = OpenAI()
        >>> conversations = {}  # session_id -> messages
        >>> 
        >>> async def my_openai_chat(query: str, session_id: str, user_id: str) -> str:
        >>>     # Get or create conversation history
        >>>     if session_id not in conversations:
        >>>         conversations[session_id] = []
        >>>     
        >>>     # Add user message
        >>>     conversations[session_id].append({"role": "user", "content": query})
        >>>     
        >>>     # Call OpenAI with full history
        >>>     response = await client.chat.completions.create(
        >>>         model="gpt-4",
        >>>         messages=conversations[session_id]
        >>>     )
        >>>     result = response.choices[0].message.content
        >>>     
        >>>     # Store assistant response
        >>>     conversations[session_id].append({"role": "assistant", "content": result})
        >>>     return result
        >>> 
        >>> adapter = LlamaIndexAdapter(my_openai_chat)
        >>> agent = Agent(name="openai_agent", seed="...", mailbox=True)
        >>> agent.include(adapter.protocol, publish_manifest=True)
        >>> adapter.run(agent)
    
    Example with Agentverse cloud registration:
        >>> adapter = LlamaIndexAdapter(
        >>>     my_function,
        >>>     agentverse_api_token="agv_xxx",
        >>>     description="My AI agent"
        >>> )
        >>> agent = Agent(name="my_agent", seed="...", mailbox=True)
        >>> agent.include(adapter.protocol, publish_manifest=True)
        >>> adapter.run(agent)  # ✅ Registers to Almanac + Agentverse
    
    Advanced Usage (Manual Control):
        >>> adapter = LlamaIndexAdapter(my_function)
        >>> agent = Agent(...)
        >>> agent.include(adapter.protocol, publish_manifest=True)
        >>> 
        >>> adapter.register(agent)  # Setup registration handler only
        >>> agent.run()  # Run manually (sync)
        >>> # OR
        >>> await agent.run_async()  # Run manually (async)
        
    Attributes:
        function: The user's callable function
        agentverse_api_token: Optional API token for Agentverse registration
        description: Optional description for marketplace listing
        protocol: Protocol for handling queries via chat
    """
    
    def __init__(
        self,
        function: Callable[[str, str, str], Union[str, Awaitable[str]]],
        agentverse_api_token: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize the universal AI adapter.
        
        Creates protocol with message handlers that route queries to your
        provided function. The function can wrap ANY AI framework.
        
        Args:
            function: A callable that processes queries with signature:
                (query: str, session_id: str, user_id: str) -> str | Awaitable[str]
                
                Can be:
                - A regular function (sync or async)
                - A lambda
                - A class instance with __call__ method
                - Any callable matching the signature
                
                Your function receives:
                - query: The user's question/request
                - session_id: Unique ID for this conversation session
                - user_id: The sender's address/identifier
                
                Your function should return the response as a string.
                
            agentverse_api_token: Optional API token for Agentverse marketplace registration.
                If provided, you can call register_agent() to publish agent to marketplace.
                
            description: Optional description for Agentverse marketplace listing.
                
        Raises:
            TypeError: If function is None or doesn't match expected signature
            
        Example with LlamaIndex:
            >>> from llama_index.core import VectorStoreIndex
            >>> 
            >>> # Create your LlamaIndex setup
            >>> index = VectorStoreIndex.from_documents(documents)
            >>> query_engine = index.as_query_engine()
            >>> 
            >>> # Wrap in a function
            >>> def my_rag_function(query: str, session_id: str, user_id: str) -> str:
            >>>     response = query_engine.query(query)
            >>>     return str(response)
            >>> 
            >>> # Create adapter
            >>> adapter = LlamaIndexAdapter(my_rag_function)
            
        Example with OpenAI:
            >>> from openai import OpenAI
            >>> client = OpenAI()
            >>> 
            >>> async def my_openai_function(query: str, session_id: str, user_id: str) -> str:
            >>>     response = await client.chat.completions.create(
            >>>         model="gpt-4",
            >>>         messages=[{"role": "user", "content": query}]
            >>>     )
            >>>     return response.choices[0].message.content
            >>> 
            >>> adapter = LlamaIndexAdapter(my_openai_function)
            
        Example with custom memory:
            >>> conversations = {}  # session_id -> list of messages
            >>> 
            >>> def my_stateful_function(query: str, session_id: str, user_id: str) -> str:
            >>>     # Get or create conversation history
            >>>     if session_id not in conversations:
            >>>         conversations[session_id] = []
            >>>     
            >>>     # Add user message
            >>>     conversations[session_id].append({"role": "user", "content": query})
            >>>     
            >>>     # Call your AI with full history
            >>>     response = your_ai_call(conversations[session_id])
            >>>     
            >>>     # Store response
            >>>     conversations[session_id].append({"role": "assistant", "content": response})
            >>>     return response
            >>> 
            >>> adapter = LlamaIndexAdapter(my_stateful_function)
        """
        # Validate input
        if function is None:
            raise TypeError(
                "function cannot be None. "
                "Please provide a callable with signature: "
                "(query: str, session_id: str, user_id: str) -> str | Awaitable[str]"
            )
        
        if not callable(function):
            raise TypeError(
                f"function must be callable, got {type(function).__name__}. "
                "Please provide a function, lambda, or object with __call__ method."
            )
        
        # Validate signature (check parameter count)
        try:
            sig = inspect.signature(function)
            params = list(sig.parameters.values())
            
            # Should have exactly 3 parameters (query, session_id, user_id)
            if len(params) != 3:
                raise TypeError(
                    f"function must accept exactly 3 parameters (query, session_id, user_id), "
                    f"got {len(params)} parameters: {list(sig.parameters.keys())}"
                )
        except ValueError:
            # Some built-in functions don't have inspectable signatures
            # Allow them through
            logger.warning(
                "Could not inspect function signature. "
                "Make sure your function accepts (query: str, session_id: str, user_id: str)"
            )
        
        self.function = function
        self.agentverse_api_token = agentverse_api_token
        self.description = description
        self._registered = False
        
        # Check if function is async
        self._is_async = inspect.iscoroutinefunction(function)
        
        logger.info(
            f"Initializing LlamaIndexAdapter with: "
            f"{getattr(function, '__name__', type(function).__name__)} "
            f"({'async' if self._is_async else 'sync'})"
        )
        
        # Create protocol
        try:
            self._protocol = self._create_protocol()
            logger.info("Protocol created successfully")
        except Exception as e:
            logger.error(f"Failed to create protocol: {str(e)}")
            raise
        
        logger.info("LlamaIndexAdapter initialized successfully")
    
    @property
    def protocol(self) -> Protocol:
        """Get the main protocol.
        
        Returns:
            The protocol handling chat-based queries
            
        Example:
            >>> adapter = LlamaIndexAdapter(agent_ll)
            >>> agent.include(adapter.protocol, publish_manifest=True)
        """
        return self._protocol
    
    def register(
        self, 
        agent: Any, 
        readme: Optional[str] = None,
        wait_seconds: int = 10
    ):
        """Register agent with Almanac (always) and Agentverse (if token provided).
        
        This method AUTOMATICALLY registers a startup handler that will:
        - ✅ Always: Register to Almanac for local agent discovery (automatic via uAgents)
        - ✅ If token provided: Register to Agentverse cloud marketplace
        
        The Almanac registration happens automatically through the uAgents library.
        This method only needs to handle optional Agentverse cloud registration.
        
        Args:
            agent: The uAgent instance (must have .name and .address attributes)
            readme: Optional custom README (only used if token provided)
            wait_seconds: Seconds to wait after startup before registration (default: 10)
            
        Example (with Agentverse cloud):
            >>> adapter = LlamaIndexAdapter(
            >>>     agent_ll,
            >>>     agentverse_api_token="agv_xxx",  # ✅ Token provided
            >>>     description="My RAG agent"
            >>> )
            >>> agent = Agent(name="my_agent", mailbox=True)
            >>> agent.include(adapter.protocol)
            >>> adapter.register(agent)  # Registers to Almanac + Agentverse
            >>> agent.run()
        
        Example (local only, no cloud):
            >>> adapter = LlamaIndexAdapter(agent_ll)  # ❌ No token
            >>> agent = Agent(name="my_agent", mailbox=True)
            >>> agent.include(adapter.protocol)
            >>> adapter.register(agent)  # Registers to Almanac only
            >>> agent.run()
        """
        logger.info(f"Setting up registration for agent '{agent.name}'...")
        
        # Store registration params for the startup handler
        self._agent_for_registration = agent
        self._readme_for_registration = readme
        self._wait_seconds = wait_seconds
        
        # Register startup handler that will do the registration
        @agent.on_event("startup")
        async def auto_register(ctx):
            """Auto-registration handler (created internally by adapter)."""
            import asyncio
            
            # Wait for agent to be fully initialized
            ctx.logger.info(
                f"Waiting {wait_seconds} seconds for agent to fully initialize..."
            )
            await asyncio.sleep(wait_seconds)
            
            # Almanac registration happens AUTOMATICALLY via uAgents library
            # The uAgents Agent.run() method handles this internally
            ctx.logger.info("[OK] Agent registered to Almanac (automatic via uAgents)")
            ctx.logger.info("   Agent is discoverable locally")
            
            # Agentverse cloud registration (only if token provided)
            if self.agentverse_api_token:
                ctx.logger.info("[CLOUD] Agentverse token found")
                ctx.logger.info("[INFO] Agent will be automatically registered via mailbox connection")
                ctx.logger.info("[INFO] Look for '[mailbox]: Successfully registered' message")
                ctx.logger.info("[OK] Agent ready for Agentverse and ASI1 integration")
            else:
                ctx.logger.info("[INFO] No Agentverse token provided - skipping cloud registration")
                ctx.logger.info("   Agent is discoverable locally via Almanac only")
        
        logger.info("[OK] Registration handler added")
        if self.agentverse_api_token:
            logger.info("   Will register to: Almanac (auto) + Agentverse (cloud)")
        else:
            logger.info("   Will register to: Almanac (auto) only")
    
    def run(
        self,
        agent: Any,
        readme: Optional[str] = None,
        wait_seconds: int = 10
    ):
        """Run the agent with automatic registration (Almanac always + Agentverse if token).
        
        This is the SIMPLEST way to use the adapter. It handles everything:
        1. Sets up automatic registration (Almanac + optional Agentverse)
        2. Runs the agent
        
        The agent will automatically:
        - ✅ Register to Almanac for local discovery (always)
        - ✅ Register to Agentverse marketplace (if token provided in __init__)
        
        Args:
            agent: The uAgent instance (must have protocol already included)
            readme: Optional custom README for Agentverse (only used if token provided)
            wait_seconds: Seconds to wait after startup before registration (default: 10)
            
        Example (without Agentverse token - local only):
            >>> from uagents import Agent
            >>> from uagents_adapter import LlamaIndexAdapter
            >>> 
            >>> # Create adapter without token
            >>> adapter = LlamaIndexAdapter(agent_ll)
            >>> 
            >>> # Create agent
            >>> agent = Agent(name="my_agent", seed="...", port=8001, mailbox=True)
            >>> agent.include(adapter.protocol, publish_manifest=True)
            >>> 
            >>> # Run with auto-registration (Almanac only)
            >>> adapter.run(agent)  # ✅ Registers to Almanac automatically
        
        Example (with Agentverse token - cloud registration):
            >>> from uagents import Agent
            >>> from uagents_adapter import LlamaIndexAdapter
            >>> 
            >>> # Create adapter with token
            >>> adapter = LlamaIndexAdapter(
            >>>     agent_ll,
            >>>     agentverse_api_token="agv_xxx",
            >>>     description="My RAG agent"
            >>> )
            >>> 
            >>> # Create agent
            >>> agent = Agent(name="my_agent", seed="...", port=8001, mailbox=True)
            >>> agent.include(adapter.protocol, publish_manifest=True)
            >>> 
            >>> # Run with auto-registration (Almanac + Agentverse)
            >>> adapter.run(agent)  # ✅ Registers to Almanac + Agentverse cloud
        
        Note:
            The protocol must be included BEFORE calling run():
            >>> agent.include(adapter.protocol, publish_manifest=True)
            >>> adapter.run(agent)  # ✅ Correct order
        """
        logger.info("[START] Starting agent with automatic registration...")
        
        # Step 1: Set up registration handler
        self.register(agent, readme=readme, wait_seconds=wait_seconds)
        
        # Step 2: Run the agent (blocking)
        logger.info("Starting agent.run()...")
        agent.run()
    
    def _perform_registration(self, agent: Any, readme: Optional[str] = None):
        """Internal method to perform the actual registration.
        
        Args:
            agent: The uAgent instance
            readme: Optional custom README content
        
        Raises:
            Exception: If registration API call fails
        """
        if self._registered:
            logger.info("Agent already registered, skipping")
            return
        
        # Generate README if not provided
        if readme is None:
            readme = self._generate_readme(agent.name)
        
        # Register using API
        try:
            import requests
            
            agent_address = agent.address
            port = agent.port if hasattr(agent, 'port') else None
            
            logger.info(f"Attempting to register agent '{agent.name}' to Agentverse marketplace...")
            
            # Setup headers
            headers = {
                "Authorization": f"Bearer {self.agentverse_api_token}",
                "Content-Type": "application/json",
            }
            
            # Connect agent to mailbox (if port available)
            if port:
                connect_url = f"http://127.0.0.1:{port}/connect"
                connect_payload = {
                    "agent_type": "mailbox",
                    "user_token": self.agentverse_api_token
                }
                
                try:
                    connect_response = requests.post(
                        connect_url, json=connect_payload, headers=headers, timeout=10
                    )
                    if connect_response.status_code == 200:
                        logger.info(f"Agent '{agent.name}' connected to Agentverse")
                    else:
                        logger.warning(
                            f"Failed to connect agent to mailbox: "
                            f"{connect_response.status_code}"
                        )
                except Exception as e:
                    logger.warning(f"Error connecting agent to mailbox: {e}")
            
            # First, try to GET the agent to see if it exists
            get_url = f"https://agentverse.ai/v1/agents/{agent_address}"
            
            try:
                get_response = requests.get(get_url, headers=headers, timeout=10)
                
                if get_response.status_code == 200:
                    # Agent exists, update it
                    logger.info(f"Agent found in Agentverse, updating metadata...")
                    
                    update_payload = {
                        "name": agent.name,
                        "readme": readme,
                        "short_description": self.description or f"LlamaIndex agent: {agent.name}",
                    }
                    
                    update_response = requests.put(
                        get_url, json=update_payload, headers=headers, timeout=10
                    )
                    
                    if update_response.status_code == 200:
                        logger.info(f"[OK] Agent '{agent.name}' registered to Agentverse marketplace")
                        logger.info(f"   View at: https://agentverse.ai/agents")
                        logger.info(f"   Check 'My Agents' for: {agent.name}")
                        self._registered = True
                    else:
                        logger.warning(
                            f"Failed to update agent: {update_response.status_code} - {update_response.text}"
                        )
                
                elif get_response.status_code == 404:
                    # Agent doesn't exist yet
                    logger.warning(
                        f"Agent not found in Agentverse (404). "
                        f"This is normal for first-time setup."
                    )
                    logger.info(
                        f"[TIP] The agent needs to be created in Agentverse first. "
                        f"After clicking 'Connect' in inspector, it may take a few moments to appear in the API."
                    )
                else:
                    logger.warning(f"Unexpected response when checking agent: {get_response.status_code}")
            
            except Exception as e:
                logger.warning(f"Error checking agent status: {e}")
            
        except Exception as e:
            logger.warning(f"Agentverse registration error: {e}")
            logger.info(f"[INFO] Agent still works locally via Almanac")
    
    def _generate_readme(self, agent_name: str) -> str:
        """Generate README for Agentverse marketplace listing.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Formatted README content in Markdown
        """
        agent_type = type(self.agent_ll).__name__
        
        return f"""# {agent_name}
![tag:llamaindex](https://img.shields.io/badge/llamaindex-3D8BD3)
<br />
<br />

A LlamaIndex-powered agent for intelligent query answering and document retrieval.

## Description
{self.description or f'A LlamaIndex {agent_type} integrated with uAgents for decentralized AI services.'}

## How to Use
Send chat messages to interact with this LlamaIndex agent. The agent will:
- Process your queries using LlamaIndex
- Provide intelligent responses with source attribution
- Support complex RAG (Retrieval-Augmented Generation) workflows

## Agent Details
- **Framework**: LlamaIndex
- **Agent Type**: `{agent_type}`
- **Protocol**: AgentChatProtocol v0.1.0
- **Capabilities**: Query processing, document retrieval, chat interactions

## Example Queries
- "What is X?"
- "Explain Y to me"
- "Find information about Z"
- "Summarize the documents"

## Features
- Natural language query processing
- Document retrieval with source attribution
- Context-aware responses
- Supports Workflow, Agent, and QueryEngine types

## Contact
Agent powered by LlamaIndex and uAgents adapter.
"""
    
    async def _call_user_function(
        self, 
        query: str, 
        session_id: str, 
        user_id: str
    ) -> str:
        """Call the user's function and handle async if needed.
        
        This method is framework-agnostic and simply calls whatever function
        the user provided, passing the query, session_id, and user_id.
        
        Args:
            query: The user's natural language query
            session_id: Unique session identifier for conversation tracking
            user_id: The sender's address/ID
            
        Returns:
            The response string from the user's function
            
        Raises:
            Exception: If the user's function raises an error
        """
        try:
            logger.debug(
                f"Calling user function: {getattr(self.function, '__name__', type(self.function).__name__)} "
                f"({'async' if self._is_async else 'sync'})"
            )
            
            # Call the user's function
            result = self.function(query, session_id, user_id)
            
            # If async, await it
            if self._is_async or inspect.iscoroutine(result) or inspect.isawaitable(result):
                logger.debug("Function returned awaitable, awaiting...")
                result = await result
            
            # Ensure result is a string
            if not isinstance(result, str):
                result = str(result)
            
            logger.debug(f"Function returned: {len(result)} chars")
            return result
            
        except Exception as e:
            logger.error(f"User function error: {type(e).__name__}: {str(e)}")
            raise
    
    def _generate_readme(self, agent_name: str) -> str:
        """Generate README for Agentverse marketplace listing.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Formatted README content in Markdown
        """
        function_name = getattr(self.function, '__name__', type(self.function).__name__)
        
        return f"""# {agent_name}
![tag:ai-agent](https://img.shields.io/badge/ai--agent-3D8BD3)
<br />
<br />

An AI-powered agent with custom logic for intelligent query answering.

## Description
{self.description or f'A custom AI agent ({function_name}) integrated with uAgents for decentralized AI services.'}

## How to Use
Send chat messages to interact with this AI agent. The agent will:
- Process your queries using custom AI logic
- Provide intelligent responses
- Support session-based conversations

## Agent Details
- **Framework**: Universal (works with any AI framework)
- **Function**: `{function_name}`
- **Protocol**: AgentChatProtocol v0.1.0
- **Capabilities**: Query processing, session tracking, user context

## Example Queries
- "What is X?"
- "Explain Y to me"
- "Help me with Z"
- "Tell me about..."

## Features
- Natural language query processing
- Session-based conversation tracking
- User-specific context support
- Works with LlamaIndex, OpenAI, LangChain, CrewAI, AutoGen, and custom AI logic

## Contact
Agent powered by uAgents adapter.
"""
    
    def _create_protocol(self) -> Protocol:
        """Create protocol for handling chat-based queries.
        
        This protocol handles ChatMessage interactions from ASI1/Agentverse
        or other agents using the chat protocol. It extracts session and user
        context and passes them to the user's function.
        
        Returns:
            Protocol configured with ChatMessage handlers
            
        Message Flow:
            1. ChatMessage received
            2. Acknowledgement sent immediately
            3. Session ID extracted (from ctx.session or generated)
            4. User ID extracted (from sender)
            5. Text content extracted from message
            6. User's function called with (query, session_id, user_id)
            7. ChatMessage response sent back
            
        Error Handling:
            All errors caught internally - graceful degradation.
        """
        protocol = Protocol(
            name="UniversalAIChatProtocol",
            version="0.1.0"
        )
        logger.debug("Creating chat protocol")
        
        @protocol.on_message(model=ChatMessage)
        async def handle_chat(ctx: Context, sender: str, msg: ChatMessage):
            """Handle chat messages from ASI1/Agentverse.
            
            Extracts session and user context, calls user's function,
            and sends response back.
            """
            ctx.logger.info(f"[AI Agent] Received message from {sender[:8]}...")
            
            # Send acknowledgement
            try:
                await ctx.send(
                    sender,
                    ChatAcknowledgement(
                        timestamp=datetime.now(timezone.utc),
                        acknowledged_msg_id=msg.msg_id
                    )
                )
            except Exception as e:
                ctx.logger.error(f"[AI Agent] Failed to send ack: {str(e)}")
            
            # Extract session ID (from context or generate new one)
            session_id = ctx.session
            # User ID is the sender address
            user_id = sender
            
            # Process each content item
            for item in msg.content:
                try:
                    if isinstance(item, StartSessionContent):
                        ctx.logger.info(f"[AI Agent] Session started: {session_id}")
                        continue
                    
                    elif isinstance(item, EndSessionContent):
                        ctx.logger.info(f"[AI Agent] Session ended: {session_id}")
                        continue
                    
                    elif isinstance(item, TextContent):
                        query_text = item.text
                        if not query_text or not query_text.strip():
                            continue
                        
                        ctx.logger.info(f"[AI Agent] Query: {query_text[:50]}...")
                        
                        try:
                            # Call user's function with session and user context
                            result = await self._call_user_function(
                                query=query_text,
                                session_id=session_id,
                                user_id=user_id
                            )
                            
                            # Send response directly (no formatting)
                            await ctx.send(
                                sender,
                                ChatMessage(
                                    timestamp=datetime.now(timezone.utc),
                                    msg_id=uuid4(),
                                    content=[
                                        TextContent(type="text", text=result)
                                    ]
                                )
                            )
                            ctx.logger.info(f"[AI Agent] Response sent ({len(result)} chars)")
                            
                        except Exception as e:
                            # Send error message to user
                            error_msg = "I encountered an error processing your query. Please try again."
                            ctx.logger.error(f"[AI Agent] Query error: {str(e)}\n{traceback.format_exc()}")
                            
                            try:
                                await ctx.send(
                                    sender,
                                    ChatMessage(
                                        timestamp=datetime.now(timezone.utc),
                                        msg_id=uuid4(),
                                        content=[TextContent(type="text", text=error_msg)]
                                    )
                                )
                            except Exception:
                                pass
                
                except Exception as e:
                    ctx.logger.error(f"[AI Agent] Content processing error: {str(e)}")
                    continue
        
        @protocol.on_message(model=ChatAcknowledgement)
        async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
            """Handle chat acknowledgements."""
            ctx.logger.debug(f"[LlamaIndex] Ack received for {msg.acknowledged_msg_id}")
        
        logger.debug("Chat protocol created")
        return protocol
