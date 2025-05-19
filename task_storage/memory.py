from typing import Dict, Optional, Any
from datetime import datetime, UTC
import logging

from task_storage.base import TaskStorage, DEFAULT_USER_ID

logger = logging.getLogger("browser-use-bridge")

class InMemoryTaskStorage(TaskStorage):
    """
    In-memory implementation of TaskStorage.
    Stores tasks in memory with user segregation.
    """

    def __init__(self):
        """Initialize with empty tasks dictionary segregated by user"""
        # Top-level dictionary is keyed by user_id
        # Each user has a dictionary of tasks keyed by task_id
        self._tasks: Dict[str, Dict[str, Dict]] = {}

    def create_task(self, task_id: str, task_data: Dict, user_id: str = DEFAULT_USER_ID) -> None:
        """Create a new task with the specified ID and data"""
        # Ensure user exists in storage
        if user_id not in self._tasks:
            self._tasks[user_id] = {}
        
        # Store the task
        self._tasks[user_id][task_id] = task_data

    def get_task(self, task_id: str, user_id: str = DEFAULT_USER_ID) -> Optional[Dict]:
        """Get a task by ID"""
        if user_id not in self._tasks or task_id not in self._tasks[user_id]:
            return None
        
        # Return a copy of the task data to prevent accidental modifications
        # Exclude agent to avoid serialization issues
        task_data = self._tasks[user_id][task_id]
        result = {k: v for k, v in task_data.items() if k != "agent"}
        return result

    def update_task(self, task_id: str, update_data: Dict, user_id: str = DEFAULT_USER_ID) -> None:
        """Update a task with new data"""
        if not self.task_exists(task_id, user_id):
            raise KeyError(f"Task {task_id} not found for user {user_id}")
        
        # Update the task data
        self._tasks[user_id][task_id].update(update_data)

    def delete_task(self, task_id: str, user_id: str = DEFAULT_USER_ID) -> bool:
        """Delete a task by ID"""
        if not self.task_exists(task_id, user_id):
            return False
        
        del self._tasks[user_id][task_id]
        return True

    def list_tasks(self, user_id: str = DEFAULT_USER_ID, page: int = 1, per_page: int = 100) -> Dict:
        """List all tasks for a user with pagination"""
        if user_id not in self._tasks:
            return {"tasks": [], "total": 0, "page": page, "per_page": per_page}
        
        # Get all tasks for the user
        user_tasks = self._tasks[user_id]
        
        # Sort tasks by created_at timestamp (newest first)
        sorted_tasks = sorted(
            user_tasks.items(),
            key=lambda x: x[1].get("created_at", ""),
            reverse=True
        )
        
        # Paginate results
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_tasks = sorted_tasks[start_idx:end_idx]
        
        # Format task summaries
        task_list = []
        for task_id, task_data in paginated_tasks:
            task_summary = {
                "id": task_id,
                "status": task_data.get("status", "unknown"),
                "task": task_data.get("task", ""),
                "created_at": task_data.get("created_at", ""),
                "finished_at": task_data.get("finished_at"),
                "live_url": task_data.get("live_url", f"/live/{task_id}"),
            }
            task_list.append(task_summary)
        
        return {
            "tasks": task_list,
            "total": len(user_tasks),
            "page": page,
            "per_page": per_page
        }

    def task_exists(self, task_id: str, user_id: str = DEFAULT_USER_ID) -> bool:
        """Check if a task exists"""
        return user_id in self._tasks and task_id in self._tasks[user_id]

    def update_task_status(self, task_id: str, status: str, user_id: str = DEFAULT_USER_ID) -> None:
        """Update a task's status"""
        if not self.task_exists(task_id, user_id):
            raise KeyError(f"Task {task_id} not found for user {user_id}")
        
        self._tasks[user_id][task_id]["status"] = status

    def add_task_step(self, task_id: str, step_data: Dict, user_id: str = DEFAULT_USER_ID) -> None:
        """Add a step to a task's execution history"""
        if not self.task_exists(task_id, user_id):
            raise KeyError(f"Task {task_id} not found for user {user_id}")
        
        task = self._tasks[user_id][task_id]
        
        # Initialize steps array if not present
        if "steps" not in task:
            task["steps"] = []
        
        task["steps"].append(step_data)
        logger.info(f"Added step {step_data.get('step')} for task {task_id}")

    def add_task_media(self, task_id: str, media_data: Dict, user_id: str = DEFAULT_USER_ID) -> None:
        """Add media information to a task"""
        if not self.task_exists(task_id, user_id):
            raise KeyError(f"Task {task_id} not found for user {user_id}")
        
        task = self._tasks[user_id][task_id]
        
        # Initialize media list if not present
        if "media" not in task:
            task["media"] = []
        
        task["media"].append(media_data)

    def get_task_agent(self, task_id: str, user_id: str = DEFAULT_USER_ID) -> Any:
        """Get the agent instance associated with a task"""
        if not self.task_exists(task_id, user_id):
            return None
        
        return self._tasks[user_id][task_id].get("agent")

    def set_task_agent(self, task_id: str, agent: Any, user_id: str = DEFAULT_USER_ID) -> None:
        """Set the agent instance for a task"""
        if not self.task_exists(task_id, user_id):
            raise KeyError(f"Task {task_id} not found for user {user_id}")
        
        self._tasks[user_id][task_id]["agent"] = agent

    def set_task_output(self, task_id: str, output: str, user_id: str = DEFAULT_USER_ID) -> None:
        """Set the output result of a task"""
        if not self.task_exists(task_id, user_id):
            raise KeyError(f"Task {task_id} not found for user {user_id}")
        
        self._tasks[user_id][task_id]["output"] = output

    def set_task_error(self, task_id: str, error: str, user_id: str = DEFAULT_USER_ID) -> None:
        """Set the error message for a task"""
        if not self.task_exists(task_id, user_id):
            raise KeyError(f"Task {task_id} not found for user {user_id}")
        
        self._tasks[user_id][task_id]["error"] = error

    def mark_task_finished(self, task_id: str, user_id: str = DEFAULT_USER_ID, 
                          status: str = "finished") -> None:
        """Mark a task as finished with timestamp"""
        if not self.task_exists(task_id, user_id):
            raise KeyError(f"Task {task_id} not found for user {user_id}")
        
        task = self._tasks[user_id][task_id]
        task["status"] = status
        task["finished_at"] = datetime.now(UTC).isoformat() + "Z" 