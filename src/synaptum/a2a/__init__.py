"""SYN-44 · Delegar en un agente remoto por A2A.

Solo biblioteca estándar: no hay extra que instalar.

**En el camino gobernado la llamada no sale de aquí.** Va por el proxy del
arnés, que es quien tiene la credencial y quien puede denegar — un control en el
framework es una petición, y un proxy en el camino es una frontera. Esto es para
modo autónomo, y para apuntar a ese proxy.
"""

from .client import A2AClient
from .delegate import RemoteDelegate
from .types import ESPERANDO, TERMINALES, AgentCard, Artifact, Task, TaskState

__all__ = [
    "RemoteDelegate",
    "A2AClient",
    "AgentCard",
    "Task",
    "TaskState",
    "Artifact",
    "TERMINALES",
    "ESPERANDO",
]
