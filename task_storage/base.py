from abc import ABC, abstractmethod
from typing import Dict, Optional, Any


DEFAULT_USER_ID = "default"


class TaskStorage(ABC):
    """
    Abstract base class for task storage implementations.
    Defines the interface for storing and retrieving tasks.
    """

    @abstractmethod
    def create_task(self, task_id: str, task_data: Dict, user_id: str = DEFAULT_USER_ID) -> None:
        """Create a new task with the specified ID and data"""
        pass

    @abstractmethod
    def get_task(self, task_id: str, user_id: str = DEFAULT_USER_ID) -> Optional[Dict]:
        """Get a task by ID"""
        pass

    @abstractmethod
    def update_task(self, task_id: str, update_data: Dict, user_id: str = DEFAULT_USER_ID) -> None:
        """Update a task with new data"""
        pass

    @abstractmethod
    def delete_task(self, task_id: str, user_id: str = DEFAULT_USER_ID) -> bool:
        """Delete a task by ID"""
        pass

    @abstractmethod
    def list_tasks(self, user_id: str = DEFAULT_USER_ID, page: int = 1, per_page: int = 100) -> Dict:
        """List all tasks for a user with pagination"""
        pass

    @abstractmethod
    def task_exists(self, task_id: str, user_id: str = DEFAULT_USER_ID) -> bool:
        """Check if a task exists"""
        pass

    @abstractmethod
    def update_task_status(self, task_id: str, status: str, user_id: str = DEFAULT_USER_ID) -> None:
        """Update a task's status"""
        pass

    @abstractmethod
    def add_task_step(self, task_id: str, step_data: Dict, user_id: str = DEFAULT_USER_ID) -> None:
        """Add a step to a task's execution history"""
        pass

    @abstractmethod
    def add_task_media(self, task_id: str, media_data: Dict, user_id: str = DEFAULT_USER_ID) -> None:
        """Add media information to a task"""
        pass

    @abstractmethod
    def get_task_agent(self, task_id: str, user_id: str = DEFAULT_USER_ID) -> Any:
        """Get the agent instance associated with a task"""
        pass

    @abstractmethod
    def set_task_agent(self, task_id: str, agent: Any, user_id: str = DEFAULT_USER_ID) -> None:
        """Set the agent instance for a task"""
        pass

    @abstractmethod
    def set_task_output(self, task_id: str, output: str, user_id: str = DEFAULT_USER_ID) -> None:
        """Set the output result of a task"""
        pass

    @abstractmethod
    def set_task_error(self, task_id: str, error: str, user_id: str = DEFAULT_USER_ID) -> None:
        """Set the error message for a task"""
        pass

    @abstractmethod
    def mark_task_finished(self, task_id: str, user_id: str = DEFAULT_USER_ID, 
                          status: str = "finished") -> None:
        """Mark a task as finished with timestamp"""
        pass 