"""
ChatAdapter — a universal bridge between uAgents and any AI workflow.

Seamlessly integrate frameworks like LangChain, LlamaIndex, CrewAI, OpenAI SDK, AutoGen, 
or your custom logic into uAgents with minimal setup.

Example:
    from uagents_adapter import ChatAdapter
    plugin = ChatAdapter(my_workflow)
    agent.include(plugin.protocol)
    plugin.run(agent)
"""

from importlib import metadata

from .adapter import GenericAdapter

try:
    __version__ = metadata.version(__package__.split(".")[0])
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)


__all__ = [
    "GenericAdapter",
    "__version__",
]